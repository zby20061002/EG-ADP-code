import os
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict
from torchvision import transforms
import timm
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_PATH = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')
EMOTION_MODEL_PATH = "best_emotion_ckplus.pth"          # 仅用于验证，实际上不影响身份测试

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 标签映射（与之前一致）
code_to_idx = {1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# ==================== 加载身份模型 ====================
print("加载 FaceNet...")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def get_embedding(img_rgb):
    img = cv2.resize(img_rgb, (160, 160))
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        return facenet(tensor).cpu().numpy().flatten()

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[idx]

def evaluate_identity(probe_embeddings, subject_list, gallery_embeddings):
    """通用身份评估函数，返回 AUC, EER, Top-1, Top-5"""
    genuine_scores = []
    impostor_scores = []
    retrieval_ranks = []

    for probe_emb, subject in zip(probe_embeddings, subject_list):
        if subject in gallery_embeddings:
            genuine_scores.append(np.dot(gallery_embeddings[subject], probe_emb))
        # impostor: 与其他所有身份比较
        for other_subj, other_emb in gallery_embeddings.items():
            if other_subj != subject:
                impostor_scores.append(np.dot(other_emb, probe_emb))

        # 1:N
        similarities = {s: np.dot(gallery_embeddings[s], probe_emb) for s in gallery_embeddings}
        sorted_subjects = sorted(similarities, key=similarities.get, reverse=True)
        rank = sorted_subjects.index(subject) + 1 if subject in sorted_subjects else -1
        retrieval_ranks.append(rank)

    if not genuine_scores or not impostor_scores:
        return 0, 0, 0, 0

    y_true = [1]*len(genuine_scores) + [0]*len(impostor_scores)
    y_score = genuine_scores + impostor_scores
    auc = roc_auc_score(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    top1 = sum(r==1 for r in retrieval_ranks) / len(retrieval_ranks)
    top5 = sum(r<=5 for r in retrieval_ranks) / len(retrieval_ranks)
    return auc, eer, top1, top5

# ==================== 读取数据 ====================
df_all = pd.read_csv(META_PATH)
df_neutral = df_all[(df_all['subset'] == 'test') & (df_all['emotion'].str.lower() == 'neutral')].copy()
df_peak = df_all[(df_all['subset'] == 'test') & (~df_all['emotion_code'].isna())].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

print(f"Gallery neutral: {len(df_neutral)} 张")
print(f"Probe peak: {len(df_peak)} 张")

# 构建 Gallery 嵌入 (始终不变)
gallery_embeddings = {}
for _, row in tqdm(df_neutral.iterrows(), desc="Building gallery"):
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gallery_embeddings[row['subject']] = get_embedding(img_rgb)

# 原始 probe 信息
original_subjects = []
original_images = []
for _, row in df_peak.iterrows():
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_images.append(img_rgb)
    original_subjects.append(row['subject'])

# ==================== Sanity Check 1: 随机打乱身份标签 ====================
print("\n===== Check 1: 随机打乱 probe 身份标签 =====")
shuffled_subjects = original_subjects.copy()
random.shuffle(shuffled_subjects)
# 提取原始图像的嵌入（不加噪）
probe_embs = [get_embedding(img) for img in tqdm(original_images, desc="Extracting probe")]
auc1, eer1, top1_1, top5_1 = evaluate_identity(probe_embs, shuffled_subjects, gallery_embeddings)
print(f"  打乱后 Top-1: {top1_1:.4f} (期望接近 1/{len(gallery_embeddings)} = {1/len(gallery_embeddings):.4f})")
print(f"  AUC: {auc1:.4f}, EER: {eer1:.4f}")

# ==================== Sanity Check 2: 纯黑图像 / 纯噪声 ====================
print("\n===== Check 2: 纯黑图像 probe =====")
black_images = [np.zeros_like(img) for img in original_images]   # 全0
noise_images = [np.random.randint(0, 256, img.shape, dtype=np.uint8) for img in original_images]  # 纯随机噪声

# 用纯黑图像测试
probe_embs_black = [get_embedding(img) for img in tqdm(black_images, desc="Black probe")]
auc2_black, eer2_black, top1_black, top5_black = evaluate_identity(probe_embs_black, original_subjects, gallery_embeddings)
print(f"  纯黑 Top-1: {top1_black:.4f}, AUC: {auc2_black:.4f}, EER: {eer2_black:.4f}")

# 用纯噪声图像测试
probe_embs_noise = [get_embedding(img) for img in tqdm(noise_images, desc="Noise probe")]
auc2_noise, eer2_noise, top1_noise, top5_noise = evaluate_identity(probe_embs_noise, original_subjects, gallery_embeddings)
print(f"  纯噪声 Top-1: {top1_noise:.4f}, AUC: {auc2_noise:.4f}, EER: {eer2_noise:.4f}")

# ==================== Sanity Check 3: 强马赛克 ====================
print("\n===== Check 3: 强马赛克 probe =====")
def mosaic(img_rgb, block_size=8):
    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (w//block_size, h//block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

mosaic_images = [mosaic(img, block_size=16) for img in original_images]  # 16x16 马赛克
probe_embs_mosaic = [get_embedding(img) for img in tqdm(mosaic_images, desc="Mosaic probe")]
auc3, eer3, top1_3, top5_3 = evaluate_identity(probe_embs_mosaic, original_subjects, gallery_embeddings)
print(f"  强马赛克 Top-1: {top1_3:.4f}, AUC: {auc3:.4f}, EER: {eer3:.4f}")

print("\n===== Sanity Check 完成 =====")
