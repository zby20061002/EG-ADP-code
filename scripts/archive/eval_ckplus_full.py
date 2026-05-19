import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
import timm
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
import sys
from PIL import Image
from torchvision.transforms import functional as F

# 添加 BiSeNet 路径
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_PATH = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')
EMOTION_MODEL_PATH = "best_emotion_ckplus.pth"
BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"

# 标签映射
code_to_idx = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# 合理噪声配置 (core_sigma, noncore_sigma)，基于0-1归一化图像
# 调试验证: weak/medium/strong/very_strong 均能保持80%以上准确率
EPSILON_CONFIGS = {
    'egadp_c1': (0.001, 0.005),    # very weak
    'egadp_c2': (0.003, 0.02),     # medium
    'egadp_c3': (0.005, 0.04),     # strong
    'egadp_c4': (0.01, 0.08),      # very strong
}

METHODS = ['original'] + list(EPSILON_CONFIGS.keys())

# ==================== 加载模型 ====================
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
state_dict = torch.load(BISENET_WEIGHT, map_location=DEVICE)
bisenet.load_state_dict(state_dict, strict=False)
bisenet.eval().to(DEVICE)

print("加载情感识别模型...")
emotion_model = timm.create_model('resnet18', pretrained=False, num_classes=6)
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model.eval().to(DEVICE)

emo_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("加载 FaceNet...")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ==================== 工具函数 ====================
def predict_emotion(img_rgb):
    """输入 RGB numpy 数组，返回预测的类别索引 0-5"""
    tensor = emo_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return emotion_model(tensor).argmax(1).item()

def get_embedding(img_rgb):
    """提取 FaceNet 嵌入"""
    img = cv2.resize(img_rgb, (160, 160))
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        return facenet(tensor).cpu().numpy().flatten()

def get_core_mask(img_bgr):
    """返回核心区域掩膜 0-1，尺寸与输入相同"""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    with torch.no_grad():
        out = bisenet(tensor)
        if isinstance(out, tuple):
            out = out[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.isin(parsing, list({2, 3, 4, 5, 6, 10, 11, 12, 13})).astype(np.float32)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask

def protect_image(img_rgb, core_sigma, noncore_sigma):
    """对 RGB 图像加噪声，core/noncore 分别控制，返回 uint8 RGB"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask = get_core_mask(img_bgr)
    img = img_rgb.astype(np.float32) / 255.0
    noise = np.random.randn(*img.shape).astype(np.float32)
    core_noise = noise * core_sigma
    noncore_noise = noise * noncore_sigma
    mask_3ch = np.stack([mask] * 3, axis=-1)
    combined = mask_3ch * core_noise + (1 - mask_3ch) * noncore_noise
    protected = np.clip(img + combined, 0, 1) * 255
    return protected.astype(np.uint8)

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[idx]

# ==================== 读取元数据 ====================
df_all = pd.read_csv(META_PATH)
# 测试集 neutral 做 gallery
df_neutral = df_all[(df_all['subset'] == 'test') & (df_all['emotion'].str.lower() == 'neutral')].copy()
# 测试集 peak 做 probe
df_peak_test = df_all[(df_all['subset'] == 'test') & (~df_all['emotion_code'].isna())].copy()
df_peak_test['emotion_code'] = df_peak_test['emotion_code'].astype(int)

print(f"Gallery (neutral): {len(df_neutral)} 张")
print(f"Probe (peak): {len(df_peak_test)} 张")

# 构建 Gallery 嵌入
gallery_embeddings = {}
for _, row in tqdm(df_neutral.iterrows(), desc="Gallery"):
    subject = row['subject']
    img = cv2.imread(row['image_path'])
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gallery_embeddings[subject] = get_embedding(img_rgb)

# ==================== 评估 ====================
results_emotion = {}
results_identity = {}

for method in METHODS:
    print(f"\n===== 评估 {method} =====")
    # 存储生成图像路径（可选），但这里直接内存处理
    emotion_correct = 0
    emotion_total = 0
    genuine_scores = []
    impostor_scores = []
    retrieval_ranks = []

    for _, row in tqdm(df_peak_test.iterrows(), total=len(df_peak_test), desc=method):
        img_path = row['image_path']
        subject = row['subject']
        true_label = code_to_idx[int(row['emotion_code'])]

        # 读取原始 peak 图像
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 生成保护图像（original 方法不加噪）
        if method == 'original':
            protected_rgb = img_rgb
        else:
            core_sigma, noncore_sigma = EPSILON_CONFIGS[method]
            protected_rgb = protect_image(img_rgb, core_sigma, noncore_sigma)

        # 情感评估
        pred = predict_emotion(protected_rgb)
        if pred == true_label:
            emotion_correct += 1
        emotion_total += 1

        # 身份评估
        probe_emb = get_embedding(protected_rgb)
        # 1:1 verification
        if subject in gallery_embeddings:
            genuine_scores.append(np.dot(gallery_embeddings[subject], probe_emb))
        # impostor 与所有其他身份比较
        for other_subj, other_emb in gallery_embeddings.items():
            if other_subj != subject:
                impostor_scores.append(np.dot(other_emb, probe_emb))

        # 1:N identification
        similarities = {s: np.dot(gallery_embeddings[s], probe_emb) for s in gallery_embeddings}
        sorted_subjects = sorted(similarities, key=similarities.get, reverse=True)
        rank = sorted_subjects.index(subject) + 1 if subject in sorted_subjects else -1
        retrieval_ranks.append(rank)

    # 计算情感准确率
    emotion_acc = emotion_correct / emotion_total if emotion_total > 0 else 0
    results_emotion[method] = emotion_acc
    print(f"  情感准确率: {emotion_acc:.4f} ({emotion_correct}/{emotion_total})")

    # 计算身份指标
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        results_identity[method] = {'AUC': 0, 'EER': 0, 'Top-1': 0, 'Top-5': 0}
        continue

    y_true = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    y_score = genuine_scores + impostor_scores
    auc = roc_auc_score(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    top1 = sum(r == 1 for r in retrieval_ranks) / len(retrieval_ranks)
    top5 = sum(r <= 5 for r in retrieval_ranks) / len(retrieval_ranks)

    results_identity[method] = {'AUC': auc, 'EER': eer, 'Top-1': top1, 'Top-5': top5}
    print(f"  AUC: {auc:.4f}, EER: {eer:.4f}, Top-1: {top1:.4f}, Top-5: {top5:.4f}")

# ==================== 汇总表格 ====================
print("\n" + "=" * 60)
print("最终评估汇总")
print("=" * 60)
print(f"{'方法':<15}{'情感Acc':<10}{'AUC':<10}{'EER':<10}{'Top-1':<10}{'Top-5':<10}")
for method in METHODS:
    acc = results_emotion[method]
    auc = results_identity[method]['AUC']
    eer = results_identity[method]['EER']
    top1 = results_identity[method]['Top-1']
    top5 = results_identity[method]['Top-5']
    print(f"{method:<15}{acc:<10.4f}{auc:<10.4f}{eer:<10.4f}{top1:<10.4f}{top5:<10.4f}")