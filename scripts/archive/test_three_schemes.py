import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
import timm
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_auc_score, roc_curve
from PIL import Image
from torchvision.transforms import functional as F

# 添加 BiSeNet 路径
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_PATH = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')
EMOTION_MODEL_PATH = "best_emotion_ckplus.pth"
BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"

code_to_idx = {1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

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
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

print("加载 FaceNet...")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ==================== 辅助函数 ====================
def predict_emotion(img_rgb):
    tensor = emo_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return emotion_model(tensor).argmax(1).item()

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

# 身份评估函数
def evaluate_identity(probe_embs, subjects, gallery_embs):
    genuine, impostor, ranks = [], [], []
    for emb, subj in zip(probe_embs, subjects):
        if subj in gallery_embs:
            genuine.append(np.dot(gallery_embs[subj], emb))
        for other_subj, other_emb in gallery_embs.items():
            if other_subj != subj:
                impostor.append(np.dot(other_emb, emb))
        sims = {s: np.dot(gallery_embs[s], emb) for s in gallery_embs}
        sorted_subjs = sorted(sims, key=sims.get, reverse=True)
        rank = sorted_subjs.index(subj) + 1 if subj in sorted_subjs else -1
        ranks.append(rank)
    if not genuine or not impostor:
        return None, None, 0, 0
    y_true = [1]*len(genuine) + [0]*len(impostor)
    y_score = genuine + impostor
    auc = roc_auc_score(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    top1 = sum(r==1 for r in ranks) / len(ranks)
    top5 = sum(r<=5 for r in ranks) / len(ranks)
    return auc, eer, top1, top5

# 获取 BiSeNet 解析图
def get_parsing(img_rgb):
    h, w = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb).resize((512,512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    with torch.no_grad():
        out = bisenet(tensor)
        if isinstance(out, tuple): out = out[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return parsing

# 根据 BiSeNet 标签生成三分区掩膜
def get_three_region_masks(parsing):
    # BiSeNet 标签参考 common.py ATTRIBUTES (索引从1开始)
    # 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 6: eye_g (眼镜)
    # 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth, 12: u_lip, 13: l_lip
    # 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
    # 我们定义：
    # 表情关键区：嘴唇(11,12,13)、嘴角、眉毛(2,3)
    # 身份关键区：眼睛(4,5)、鼻子(10)、可能加上鼻梁区域(10)
    # 非核心区：脸颊、额头、下颌等皮肤区域(skin=1)、耳朵、脖子、头发、背景等
    emotion_labels = {2,3,11,12,13}      # 眉毛、嘴唇
    identity_labels = {4,5,10}           # 眼睛、鼻子
    # 非核心区包含其余所有标签，但我们用排除法生成掩膜
    h, w = parsing.shape
    emotion_mask = np.isin(parsing, list(emotion_labels)).astype(np.float32)
    identity_mask = np.isin(parsing, list(identity_labels)).astype(np.float32)
    # 非核心区 = 1 - emotion_mask - identity_mask，但注意可能有重叠（实际上不会重叠）
    noncore_mask = 1.0 - emotion_mask - identity_mask
    # 稍微膨胀一下核心区，避免边界太硬
    kernel = np.ones((3,3), np.uint8)
    emotion_mask = cv2.dilate(emotion_mask, kernel, iterations=1)
    identity_mask = cv2.dilate(identity_mask, kernel, iterations=1)
    noncore_mask = 1.0 - emotion_mask - identity_mask
    noncore_mask = np.clip(noncore_mask, 0, 1)
    return emotion_mask, identity_mask, noncore_mask

# 三分区加噪
def apply_tri_noise(img_rgb, emo_sigma, id_sigma, nc_sigma):
    parsing = get_parsing(img_rgb)
    emo_mask, id_mask, nc_mask = get_three_region_masks(parsing)
    img = img_rgb.astype(np.float32)/255.0
    noise = np.random.randn(*img.shape).astype(np.float32)
    # 各区域加权
    result = img.copy()
    for c in range(3):
        result[:,:,c] += emo_mask * noise[:,:,c] * emo_sigma
        result[:,:,c] += id_mask * noise[:,:,c] * id_sigma
        result[:,:,c] += nc_mask * noise[:,:,c] * nc_sigma
    result = np.clip(result, 0, 1)*255
    return result.astype(np.uint8)

# 非核心区结构性扰动
def apply_structured_noncore(img_rgb, core_sigma, noncore_type, noncore_param):
    parsing = get_parsing(img_rgb)
    emo_mask, id_mask, _ = get_three_region_masks(parsing)
    core_mask = emo_mask + id_mask
    core_mask = np.clip(core_mask, 0, 1)
    noncore_mask = 1.0 - core_mask
    img = img_rgb.astype(np.float32)/255.0
    # 核心区加轻噪声
    noise = np.random.randn(*img.shape).astype(np.float32)
    core_img = img + core_mask[:,:,np.newaxis] * noise * core_sigma
    core_img = np.clip(core_img, 0, 1)*255
    core_img = core_img.astype(np.uint8)
    # 非核心区处理
    if noncore_type == 'mosaic':
        block = noncore_param
        h, w = img_rgb.shape[:2]
        small = cv2.resize(img_rgb, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
        nc_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif noncore_type == 'blur':
        ksize = noncore_param
        nc_img = cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)
    elif noncore_type == 'blur_noise':
        ksize = noncore_param
        blur = cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)
        nc_img = blur.astype(np.float32)/255.0 + np.random.randn(*img.shape).astype(np.float32)*0.02
        nc_img = np.clip(nc_img, 0, 1)*255
        nc_img = nc_img.astype(np.uint8)
    elif noncore_type == 'mosaic_noise':
        block = noncore_param
        h, w = img_rgb.shape[:2]
        small = cv2.resize(img_rgb, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        nc_img = mosaic.astype(np.float32)/255.0 + np.random.randn(*img.shape).astype(np.float32)*0.02
        nc_img = np.clip(nc_img, 0, 1)*255
        nc_img = nc_img.astype(np.uint8)
    else:
        raise ValueError(f"Unknown noncore_type: {noncore_type}")
    # 合成：核心区用 core_img，非核心区用 nc_img
    combined = np.zeros_like(img_rgb)
    for c in range(3):
        combined[:,:,c] = core_mask * core_img[:,:,c] + noncore_mask * nc_img[:,:,c]
    return combined.astype(np.uint8)

# 强马赛克 baseline
def strong_mosaic(img_rgb, block=16):
    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# ==================== 数据准备 ====================
df_all = pd.read_csv(META_PATH)
df_neutral = df_all[(df_all['subset']=='test') & (df_all['emotion'].str.lower()=='neutral')].copy()
df_peak = df_all[(df_all['subset']=='test') & (~df_all['emotion_code'].isna())].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

print(f"Gallery (neutral): {len(df_neutral)} 张")
print(f"Probe (peak): {len(df_peak)} 张")

# 构建 gallery 嵌入
gallery_embs = {}
for _, row in tqdm(df_neutral.iterrows(), desc="Gallery"):
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gallery_embs[row['subject']] = get_embedding(img_rgb)

# 准备 probe 原始图像和标签
probe_imgs = []
probe_subjects = []
probe_true_labels = []
for _, row in df_peak.iterrows():
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    probe_imgs.append(img_rgb)
    probe_subjects.append(row['subject'])
    probe_true_labels.append(code_to_idx[int(row['emotion_code'])])

# ==================== 测试函数 ====================
def test_method(method_name, process_func):
    print(f"\n--- 测试 {method_name} ---")
    emo_correct = 0
    emo_total = 0
    pro_embs = []
    for img, subj, true_label in tqdm(zip(probe_imgs, probe_subjects, probe_true_labels), total=len(probe_imgs), desc=method_name):
        protected = process_func(img)
        pred = predict_emotion(protected)
        if pred == true_label:
            emo_correct += 1
        emo_total += 1
        pro_embs.append(get_embedding(protected))
    emo_acc = emo_correct / emo_total if emo_total else 0
    _, _, top1, _ = evaluate_identity(pro_embs, probe_subjects, gallery_embs)
    print(f"  情感 Acc: {emo_acc:.4f}, 身份 Top-1: {top1:.4f}")
    return emo_acc, top1

# ==================== 执行测试 ====================
results = {}

# 方案1：三分区 EG-ADP
configs_p1 = {
    'P1': (0.003, 0.020, 0.050),
    'P2': (0.005, 0.040, 0.080),
    'P3': (0.005, 0.060, 0.120),
    'P4': (0.008, 0.080, 0.150),
}
for name, (es, ids, ncs) in configs_p1.items():
    key = f"TriEG-ADP_{name}"
    results[key] = test_method(key, lambda img, es=es, ids=ids, ncs=ncs: apply_tri_noise(img, es, ids, ncs))

# 方案2：非核心区结构性扰动（这里测试几种典型组合）
struct_configs = [
    ('Struct_mosaic', 'mosaic', 16, 0.005),
    ('Struct_blur', 'blur', 21, 0.005),
    ('Struct_blur_noise', 'blur_noise', 21, 0.005),
    ('Struct_mosaic_noise', 'mosaic_noise', 16, 0.005),
]
for name, nc_type, nc_param, core_sig in struct_configs:
    results[name] = test_method(name, lambda img, nt=nc_type, np_=nc_param, cs=core_sig: apply_structured_noncore(img, cs, nt, np_))

# 方案3：强马赛克 baseline
results['Strong Mosaic'] = test_method('Strong Mosaic', lambda img: strong_mosaic(img, 16))

# 原始 baseline
results['Original'] = test_method('Original', lambda img: img)

# ==================== 输出汇总 ====================
print("\n" + "="*60)
print("最终对比结果")
print("="*60)
print(f"{'方法':<20}{'情感Acc':<10}{'身份Top-1':<10}")
for method, (acc, top1) in results.items():
    print(f"{method:<20}{acc:<10.4f}{top1:<10.4f}")