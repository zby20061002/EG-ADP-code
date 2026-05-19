import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from tqdm import tqdm
from collections import defaultdict
import tempfile
import shutil
from skimage.metrics import structural_similarity as ssim

# ==================== 配置 ====================
# FER-2013 测试集路径（请修改为你的实际路径）
test_root = r"F:\python\2024218729zby_ML\EG-ADP\data\FER-2013\test"
emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 调试开关
SAVE_DEBUG_IMAGES = True          # 保存加噪前后对比图
DEBUG_IMAGE_DIR = "./debug_egadp" # 保存路径
MAX_DEBUG_SAMPLES = 20            # 最多保存样本数

# MediaPipe 模型文件名（与脚本放在同一目录）
LANDMARKER_MODEL_FILENAME = "face_landmarker.task"

# ==================== EG-ADP 类（使用 MediaPipe）====================
class EG_ADP:
    def __init__(self, delta=1e-5):
        self.delta = delta
        self.epsilon_core = 50.0      # 原来 12 → 50（噪声很小）
        self.epsilon_pos = 40.0       # 原来 10 → 40
        self.epsilon_neu = 30.0       # 原来 6 → 30
        self.epsilon_neg = 20.0       # 原来 3 → 20

        # 初始化 MediaPipe FaceLandmarker
        self._init_mediapipe()

    def _init_mediapipe(self):
        """初始化 MediaPipe FaceLandmarker（CPU 模式，低阈值）"""
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, LANDMARKER_MODEL_FILENAME)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe 模型文件未找到: {model_path}\n"
                f"请从以下地址下载并放置到脚本所在目录：\n"
                f"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path,
            delegate=mp.tasks.BaseOptions.Delegate.CPU
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        # MediaPipe 468 点模型中，眼睛和嘴唇的关键点索引
        self.LEFT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 173]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398]
        self.LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321]

    def __del__(self):
        """释放 MediaPipe 资源"""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

    # --- 差分隐私噪声标准差 ---
    def _noise_std(self, epsilon, sensitivity=255.0):
        # 高斯机制：σ = (Δf * √(2·ln(1.25/δ))) / ε
        return (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / epsilon

    # --- 根据情绪标签获取非核心区隐私预算 ---
    def _get_epsilon_nc(self, emotion_label):
        # emotion_label: 0 angry, 1 disgust, 2 fear, 3 happy, 4 neutral, 5 sad, 6 surprise
        if emotion_label == 3:      # happy
            return self.epsilon_pos
        elif emotion_label == 4:    # neutral
            return self.epsilon_neu
        else:                       # 所有负面情绪
            return self.epsilon_neg

    # --- 创建核心区域掩膜（使用 MediaPipe 关键点）---
    def _create_core_mask(self, rgb_image, expand=10):
        h, w = rgb_image.shape[:2]
        # 将图像转换为 MediaPipe 输入格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            # 未检测到人脸，返回全零掩膜（后续将跳过加噪）
            return np.zeros((h, w), dtype=np.uint8)

        # 收集眼睛和嘴唇的关键点坐标
        pts = []
        landmarks = detection_result.face_landmarks[0]
        for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES + self.LIP_INDICES:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts)
        mask = cv2.fillPoly(np.zeros((h, w), dtype=np.uint8), [hull], 255)

        # 膨胀以扩大保护区域
        kernel = np.ones((expand, expand), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    # --- 对单张图像施加 EG-ADP 噪声 ---
    def protect_image(self, image, emotion_label):
        if image is None or image.size == 0:
            return image

        # 确保是 uint8 RGB
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        eps_nc = self._get_epsilon_nc(emotion_label)
        sigma_core = self._noise_std(self.epsilon_core)
        sigma_nc = self._noise_std(eps_nc)

        mask = self._create_core_mask(image)
        if mask.sum() == 0:
            # 无人脸，不做任何处理（可改为全图加噪，此处保持原样）
            return image

        mask_float = mask / 255.0
        noise_core = np.random.normal(0, sigma_core, image.shape)
        noise_nc   = np.random.normal(0, sigma_nc,   image.shape)

        noisy = image.astype(np.float32) + noise_core * mask_float[..., np.newaxis] + \
                noise_nc * (1 - mask_float[..., np.newaxis])
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy


# ==================== 辅助函数 ====================
def get_deepface_emotion(image_path):
    """使用 DeepFace 预测单张图片的情感，返回情感字符串（七类之一）"""
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=True)
        if result and isinstance(result, list) and len(result) > 0:
            return result[0]['dominant_emotion']
    except Exception:
        pass
    return None


def compute_ssim_stats(original_dir, protected_dir, emotion_folders):
    """计算原始图像与保护后图像的 SSIM，返回均值、标准差、最小值、最大值"""
    ssim_list = []
    for emotion in emotion_folders:
        src_dir = os.path.join(original_dir, emotion)
        dst_dir = os.path.join(protected_dir, emotion)
        if not os.path.isdir(src_dir):
            continue
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            if not os.path.exists(dst_path):
                continue
            img_orig = cv2.imread(src_path)
            img_prot = cv2.imread(dst_path)
            if img_orig is None or img_prot is None:
                continue
            gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            gray_prot = cv2.cvtColor(img_prot, cv2.COLOR_BGR2GRAY)
            score = ssim(gray_orig, gray_prot)
            ssim_list.append(score)
    if len(ssim_list) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return np.mean(ssim_list), np.std(ssim_list), np.min(ssim_list), np.max(ssim_list)


# ==================== 主实验流程 ====================
def main():
    print("=" * 60)
    print("EG-ADP 实验 (使用 MediaPipe 人脸关键点检测)")
    print("=" * 60)

    # 创建调试目录
    if SAVE_DEBUG_IMAGES:
        os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
        print(f"调试图片将保存至: {DEBUG_IMAGE_DIR}")

    # 1. 加载测试集路径及标签
    test_images = []
    for idx, emotion in enumerate(emotion_folders):
        emotion_dir = os.path.join(test_root, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"警告: 目录不存在 {emotion_dir}, 跳过该类别")
            continue
        for fname in os.listdir(emotion_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append((os.path.join(emotion_dir, fname), idx, emotion))
    print(f"加载测试图片总数: {len(test_images)}")
    if len(test_images) == 0:
        print("错误: 没有找到图片, 请检查 test_root 路径")
        return

    # 2. 初始化 EG-ADP
    eg = EG_ADP()

    # 3. Baseline 准确率（原始图片）
    print("\n--- 评估 Baseline 准确率 ---")
    baseline_correct = 0
    baseline_total = 0
    baseline_confusion = defaultdict(int)
    for img_path, true_idx, true_emotion in tqdm(test_images, desc="Baseline"):
        pred_emo = get_deepface_emotion(img_path)
        if pred_emo is None:
            continue
        if pred_emo in emotion_folders:
            pred_idx = emotion_folders.index(pred_emo)
            if pred_idx == true_idx:
                baseline_correct += 1
            else:
                baseline_confusion[(true_idx, pred_idx)] += 1
        baseline_total += 1
    baseline_acc = baseline_correct / baseline_total if baseline_total else 0
    print(f"Baseline Accuracy: {baseline_acc:.4f} ({baseline_correct}/{baseline_total})")

    # 4. 创建临时目录存放加噪图片
    protected_root = tempfile.mkdtemp(prefix="eg_adp_test_")
    print(f"\n创建临时目录存放加噪图片: {protected_root}")
    for emotion in emotion_folders:
        os.makedirs(os.path.join(protected_root, emotion), exist_ok=True)

    # 5. 对测试集施加 EG-ADP 噪声并评估
    print("\n--- 施加 EG-ADP 噪声并评估 ---")
    noisy_correct = 0
    noisy_total = 0
    noisy_confusion = defaultdict(int)
    debug_saved = 0

    for img_path, true_idx, true_emotion in tqdm(test_images, desc="EG-ADP"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        protected_img = eg.protect_image(img_rgb, true_idx)

        # 保存加噪后图片
        fname = os.path.basename(img_path)
        protected_path = os.path.join(protected_root, true_emotion, fname)
        protected_img_bgr = cv2.cvtColor(protected_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(protected_path, protected_img_bgr)

        # 保存调试样本（原始+加噪对比）
        if SAVE_DEBUG_IMAGES and debug_saved < MAX_DEBUG_SAMPLES:
            orig_save = os.path.join(DEBUG_IMAGE_DIR, f"orig_{true_emotion}_{fname}")
            cv2.imwrite(orig_save, img)
            noisy_save = os.path.join(DEBUG_IMAGE_DIR, f"noisy_{true_emotion}_{fname}")
            cv2.imwrite(noisy_save, protected_img_bgr)
            debug_saved += 1

        # 对加噪图片进行情感分析
        pred_emo = get_deepface_emotion(protected_path)
        if pred_emo is None:
            continue
        if pred_emo in emotion_folders:
            pred_idx = emotion_folders.index(pred_emo)
            if pred_idx == true_idx:
                noisy_correct += 1
            else:
                noisy_confusion[(true_idx, pred_idx)] += 1
        noisy_total += 1

    noisy_acc = noisy_correct / noisy_total if noisy_total else 0
    print(f"EG-ADP Acc:      {noisy_acc:.4f} ({noisy_correct}/{noisy_total})")
    print(f"Accuracy Drop:   {baseline_acc - noisy_acc:.4f} ({(baseline_acc - noisy_acc)*100:.2f}%)")

    # 6. 混淆情况
    if noisy_confusion:
        print("\n--- 易混淆类别 (真实 -> 预测, 计数) ---")
        sorted_conf = sorted(noisy_confusion.items(), key=lambda kv: kv[1], reverse=True)[:5]
        for (true_idx, pred_idx), cnt in sorted_conf:
            print(f"  {emotion_folders[true_idx]} -> {emotion_folders[pred_idx]} : {cnt}")

    # 7. 隐私度量 (SSIM)
    mean_ssim, std_ssim, min_ssim, max_ssim = compute_ssim_stats(test_root, protected_root, emotion_folders)
    print(f"\n--- 隐私度量 (SSIM) ---")
    print(f"Mean SSIM: {mean_ssim:.6f}  Std: {std_ssim:.6f}")
    print(f"Min SSIM: {min_ssim:.6f}   Max SSIM: {max_ssim:.6f}")
    if mean_ssim > 0.99:
        print("⚠️  警告: SSIM 过高 (>0.99)，噪声可能未生效！请检查人脸检测或噪声参数。")
    else:
        print("✅ SSIM 正常，隐私保护模块已生效。")

    # 8. 清理临时目录
    shutil.rmtree(protected_root)
    if SAVE_DEBUG_IMAGES:
        print(f"\n调试图片已保存至 {DEBUG_IMAGE_DIR}，请手动检查加噪效果。")
    print("\n实验结束。")


if __name__ == "__main__":
    main()