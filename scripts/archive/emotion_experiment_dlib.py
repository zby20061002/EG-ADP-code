import os
import cv2
import numpy as np
import torch
import random
import shutil
import tempfile
import urllib.request
import bz2
from tqdm import tqdm
import dlib
from facenet_pytorch import InceptionResnetV1
from collections import defaultdict
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ==================== 配置 ====================
test_root = r"F:\python\2024218729zby_ML\EG-ADP\data\FER-2013\test"
emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

SAVE_DEBUG_IMAGES = True
DEBUG_IMAGE_DIR = "./debug_egadp"
MAX_DEBUG_SAMPLES = 20
ERROR_RATES = [0.0, 0.2]          # 预判错误率实验
UPSCALE_SIZE = 192                # 放大后尺寸，保证 dlib 对小脸有效

# ==================== 全局模型加载 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("加载情感评估模型 (vit-face-expression)...")
model_name = "trpakov/vit-face-expression"
processor = AutoImageProcessor.from_pretrained(model_name)
emotion_model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
emotion_model.eval()

print("加载 FaceNet 重识别模型...")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==================== 工具函数 ====================
def get_emotion_label(img_rgb):
    """使用 vit-face-expression 预测情感，返回索引 0~6"""
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return pred

def get_facenet_embedding(img_rgb):
    """提取 FaceNet 嵌入"""
    img = cv2.resize(img_rgb, (160, 160))
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        emb = facenet(tensor)
    return emb.cpu().numpy().flatten()

def corrupt_label(true_label, error_rate=0.2):
    if random.random() < error_rate:
        others = [l for l in range(7) if l != true_label]
        return random.choice(others)
    return true_label

# ==================== dlib 模型 ====================
DLIB_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
LOCAL_DLIB_PATH = r"F:\Major_Innovation_Project\PROJECT\resource\Dlib_face_recognition_from_camera-master\data\data_dlib\shape_predictor_68_face_landmarks.dat"
if os.path.exists(LOCAL_DLIB_PATH):
    DLIB_MODEL_PATH = LOCAL_DLIB_PATH
else:
    if not os.path.exists(DLIB_MODEL_PATH):
        print("正在下载 dlib 68 点模型...")
        DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_path = DLIB_MODEL_PATH + ".bz2"
        urllib.request.urlretrieve(DLIB_MODEL_URL, bz2_path)
        with bz2.open(bz2_path, 'rb') as f_in:
            with open(DLIB_MODEL_PATH, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(bz2_path)

# ==================== EG-ADP 类（dlib 版） ====================
class EG_ADP:
    def __init__(self, delta=1e-5):
        self.delta = delta
        self.epsilon_core = 50.0
        self.epsilon_pos  = 40.0
        self.epsilon_neu  = 30.0
        self.epsilon_neg  = 20.0

        # 情绪到正/中/负的映射
        self.pos_set = {emotion_folders.index('happy')}
        self.neu_set = {emotion_folders.index('neutral'),
                        emotion_folders.index('surprise')}
        self.neg_set = {emotion_folders.index('angry'),
                        emotion_folders.index('disgust'),
                        emotion_folders.index('fear'),
                        emotion_folders.index('sad')}

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

    def _noise_std(self, epsilon):
        return np.sqrt(2 * np.log(1.25 / self.delta)) / epsilon

    def _get_epsilon_nc(self, emotion_label):
        if emotion_label in self.pos_set:
            return self.epsilon_pos
        elif emotion_label in self.neu_set:
            return self.epsilon_neu
        else:
            return self.epsilon_neg

    def _create_core_mask(self, img_rgb):
        """返回眼睛+嘴唇区域掩膜，同时返回是否成功检脸"""
        h, w = img_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 0)
        found = False
        if len(faces) > 0:
            found = True
            face = faces[0]
            landmarks = self.predictor(gray, face)

            # 左眼 (36-41)，右眼 (42-47)，嘴 (48-67)
            left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            mouth_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

            for pts in [left_eye_pts, right_eye_pts, mouth_pts]:
                hull = cv2.convexHull(pts.astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 1.0)

            mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)

        return mask, found

    def protect_image(self, img_rgb, emotion_label):
        img = img_rgb.astype(np.float32) / 255.0
        mask, face_found = self._create_core_mask(img_rgb)
        sigma_core = self._noise_std(self.epsilon_core)
        sigma_nc = self._noise_std(self._get_epsilon_nc(emotion_label))
        noise = np.random.randn(*img.shape).astype(np.float32)
        noise_core = noise * sigma_core
        noise_nc = noise * sigma_nc
        mask_3ch = np.stack([mask]*3, axis=-1)
        combined_noise = mask_3ch * noise_core + (1.0 - mask_3ch) * noise_nc
        noisy_img = img + combined_noise
        noisy_img = np.clip(noisy_img, 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)
        return noisy_img, face_found, mask.mean()

# ==================== 主实验 ====================
def main():
    print("="*60)
    print("EG-ADP 完整实验 (vit-face-expression 强基线 + 预判错误注入 + FaceNet攻击)")
    print("="*60)

    # 收集测试图片
    test_images = []
    for idx, emotion in enumerate(emotion_folders):
        d = os.path.join(test_root, emotion)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append((os.path.join(d, f), idx, emotion))

    print(f"测试图片总数: {len(test_images)}")
    if not test_images:
        print("未找到测试图片，请检查路径！")
        return

    eg = EG_ADP()

    # ---------- Baseline ----------
    print("\n>>> 评估 Baseline (原始图像) ...")
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for path, true_idx, emotion in tqdm(test_images, desc="Baseline"):
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = get_emotion_label(img_rgb)
        if pred == true_idx:
            correct += 1
            class_correct[true_idx] += 1
        class_total[true_idx] += 1
        total += 1
    baseline_acc = correct / total
    print(f"Baseline Accuracy: {baseline_acc:.4f} ({correct}/{total})")
    if baseline_acc < 0.5:
        print("⚠️ 警告：Baseline 准确率异常偏低，请检查数据集标签是否正确！")

    # ---------- 不同预判错误率下的 EG-ADP ----------
    all_results = {}
    for error_rate in ERROR_RATES:
        print(f"\n>>> 预判错误率: {int(error_rate*100)}%")
        noisy_correct = 0
        reid_success = 0
        total = 0
        face_detected_count = 0
        avg_mask_coverage = 0.0
        class_noisy_correct = defaultdict(int)
        class_total_ = defaultdict(int)
        class_reid_success = defaultdict(int)
        debug_count = 0

        tmp_dir = tempfile.mkdtemp(prefix="egadp_")

        for path, true_idx, true_emo in tqdm(test_images, desc="EG-ADP"):
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 放大图像以保证 dlib 检测
            img_upscaled = cv2.resize(img_rgb, (UPSCALE_SIZE, UPSCALE_SIZE), interpolation=cv2.INTER_LINEAR)

            pred_label = corrupt_label(true_idx, error_rate)
            protected, face_found, mask_cov = eg.protect_image(img_upscaled, pred_label)

            if face_found:
                face_detected_count += 1
            avg_mask_coverage += mask_cov

            pred_emo = get_emotion_label(protected)
            if pred_emo == true_idx:
                noisy_correct += 1
                class_noisy_correct[true_idx] += 1
            class_total_[true_idx] += 1

            emb_orig = get_facenet_embedding(img_upscaled)
            emb_prot = get_facenet_embedding(protected)
            cos_sim = np.dot(emb_orig, emb_prot) / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_prot))
            if cos_sim > 0.5:
                reid_success += 1
                class_reid_success[true_idx] += 1

            total += 1

            if SAVE_DEBUG_IMAGES and debug_count < MAX_DEBUG_SAMPLES:
                os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
                fname = os.path.basename(path)
                cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"orig_{true_emo}_{fname}"),
                            cv2.cvtColor(img_upscaled, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"noisy_{true_emo}_{fname}"),
                            cv2.cvtColor(protected, cv2.COLOR_RGB2BGR))
                debug_count += 1

        shutil.rmtree(tmp_dir)

        noisy_acc = noisy_correct / total
        reid_rate = reid_success / total
        face_det_rate = face_detected_count / total
        avg_mask_cov = avg_mask_coverage / total

        all_results[error_rate] = {
            'noisy_acc': noisy_acc,
            'reid_rate': reid_rate,
            'face_det_rate': face_det_rate,
            'avg_mask_cov': avg_mask_cov,
            'class_noisy_correct': dict(class_noisy_correct),
            'class_total': dict(class_total_),
            'class_reid_success': dict(class_reid_success)
        }

        print(f"EG-ADP Accuracy: {noisy_acc:.4f} ({noisy_correct}/{total})")
        print(f"Accuracy Drop: {baseline_acc - noisy_acc:.4f} ({(baseline_acc - noisy_acc)*100:.2f}%)")
        print(f"Re-identification Success Rate: {reid_rate:.4f} ({reid_success}/{total})")
        print(f"人脸检出率: {face_det_rate:.4f} ({face_detected_count}/{total})")
        print(f"平均核心掩膜覆盖率: {avg_mask_cov:.4f}")

    # ==================== 详细结果输出 ====================
    print("\n" + "="*60)
    print("实验结果详细汇总")
    print("="*60)

    print(f"\nBaseline 准确率: {baseline_acc:.4f}")
    print("\n--- 每类 Baseline 准确率 ---")
    for idx, folder in enumerate(emotion_folders):
        total_class = class_total.get(idx, 0)
        corr = class_correct.get(idx, 0)
        acc = corr / total_class if total_class > 0 else 0
        print(f"  {folder:10s}: {acc:.4f} ({corr}/{total_class})")

    for error_rate in ERROR_RATES:
        res = all_results[error_rate]
        print(f"\n--- 预判错误率 = {int(error_rate*100)}% ---")
        print(f"加噪后准确率: {res['noisy_acc']:.4f}")
        print(f"准确率下降: {baseline_acc - res['noisy_acc']:.4f}")
        print(f"重识别成功率: {res['reid_rate']:.4f}")
        print(f"人脸检出率: {res['face_det_rate']:.4f}")
        print(f"平均核心掩膜覆盖率: {res['avg_mask_cov']:.4f}")

        print("  每类加噪后准确率:")
        for idx, folder in enumerate(emotion_folders):
            total_class = res['class_total'].get(idx, 0)
            corr = res['class_noisy_correct'].get(idx, 0)
            acc = corr / total_class if total_class > 0 else 0
            print(f"    {folder:10s}: {acc:.4f} ({corr}/{total_class})")

        print("  每类重识别成功率:")
        for idx, folder in enumerate(emotion_folders):
            total_class = res['class_total'].get(idx, 0)
            reid = res['class_reid_success'].get(idx, 0)
            rate = reid / total_class if total_class > 0 else 0
            print(f"    {folder:10s}: {rate:.4f} ({reid}/{total_class})")

    if SAVE_DEBUG_IMAGES:
        print(f"\n调试图片已保存至 {DEBUG_IMAGE_DIR}")

if __name__ == "__main__":
    main()