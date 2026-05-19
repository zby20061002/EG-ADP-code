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

sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ==================== 基础配置 ====================
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

# BiSeNet 解析
def get_parsing(img_rgb):
    h, w = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    with torch.no_grad():
        out = bisenet(tensor)
        if isinstance(out, tuple): out = out[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    return cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

# 区域掩膜定义
def get_region_masks(parsing):
    # 表情区：眉毛(2,3), 上嘴唇(12), 下嘴唇(13), 嘴部(11)
    emotion_labels = {2, 3, 11, 12, 13}
    # 眼鼻区：左眼(4), 右眼(5), 鼻子(10)
    id_labels = {4, 5, 10}
    emo_mask = np.isin(parsing, list(emotion_labels)).astype(np.float32)
    id_mask = np.isin(parsing, list(id_labels)).astype(np.float32)
    # 轻微膨胀使边界融合
    kernel = np.ones((3,3), np.uint8)
    emo_mask = cv2.dilate(emo_mask, kernel, iterations=1)
    id_mask = cv2.dilate(id_mask, kernel, iterations=1)
    # 非核心区
    nc_mask = 1.0 - emo_mask - id_mask
    nc_mask = np.clip(nc_mask, 0, 1)
    return emo_mask, id_mask, nc_mask

# 处理函数工厂
def noise_processor(sigma):
    def func(img_rgb):
        img = img_rgb.astype(np.float32) / 255.0
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        return (np.clip(img + noise, 0, 1) * 255).astype(np.uint8)
    return func

def blur_processor(ksize):
    def func(img_rgb):
        return cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)
    return func

def mosaic_processor(block):
    def func(img_rgb):
        h, w = img_rgb.shape[:2]
        small = cv2.resize(img_rgb, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return func

def keep_original(img_rgb):
    return img_rgb.copy()

# 通用区域处理
def apply_region_processing(img_rgb, parsing, proc_emo, proc_id, proc_nc):
    emo_mask, id_mask, nc_mask = get_region_masks(parsing)
    img_emo = proc_emo(img_rgb) if proc_emo else img_rgb
    img_id = proc_id(img_rgb) if proc_id else img_rgb
    img_nc = proc_nc(img_rgb) if proc_nc else img_rgb
    result = emo_mask[...,None] * img_emo + id_mask[...,None] * img_id + nc_mask[...,None] * img_nc
    return result.astype(np.uint8)

# ==================== 数据准备 ====================
df_all = pd.read_csv(META_PATH)
df_neutral = df_all[(df_all['subset']=='test') & (df_all['emotion'].str.lower()=='neutral')].copy()
df_peak = df_all[(df_all['subset']=='test') & (~df_all['emotion_code'].isna())].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

print(f"Gallery (neutral): {len(df_neutral)} 张")
print(f"Probe (peak): {len(df_peak)} 张")

# 构建 gallery
gallery_embs = {}
for _, row in tqdm(df_neutral.iterrows(), desc="Gallery"):
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gallery_embs[row['subject']] = get_embedding(img_rgb)

# 收集 probe
probe_imgs, probe_subjects, probe_true_labels = [], [], []
for _, row in df_peak.iterrows():
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    probe_imgs.append(img_rgb)
    probe_subjects.append(row['subject'])
    probe_true_labels.append(code_to_idx[int(row['emotion_code'])])

# ==================== 方案配置 ====================
# 每个方案由三个区域的处理器（函数或None表示保留原样）构成
methods = {}

# 方法组 A：眼鼻模糊 + 非核心马赛克
methods['A1'] = {'emo': noise_processor(0.003), 'id': blur_processor(11), 'nc': mosaic_processor(16)}
methods['A2'] = {'emo': noise_processor(0.003), 'id': blur_processor(21), 'nc': mosaic_processor(16)}
methods['A3'] = {'emo': noise_processor(0.005), 'id': blur_processor(31), 'nc': mosaic_processor(16)}
methods['A4'] = {'emo': keep_original,            'id': blur_processor(21), 'nc': mosaic_processor(12)}

# 方法组 B：眼鼻马赛克 + 非核心模糊
methods['B1'] = {'emo': noise_processor(0.003), 'id': mosaic_processor(8),  'nc': blur_processor(21)}
methods['B2'] = {'emo': noise_processor(0.003), 'id': mosaic_processor(12), 'nc': blur_processor(21)}
methods['B3'] = {'emo': noise_processor(0.005), 'id': mosaic_processor(8),  'nc': mosaic_processor(12)}
methods['B4'] = {'emo': keep_original,            'id': mosaic_processor(8),  'nc': lambda img: (np.clip(cv2.GaussianBlur(img, (21,21), 0).astype(np.float32)/255.0 + np.random.randn(*img.shape).astype(np.float32)*0.02, 0, 1)*255).astype(np.uint8)}

# 方法组 C：遮挡型 baseline
methods['C_EyeBlur']    = {'emo': keep_original, 'id': blur_processor(21),   'nc': keep_original}
methods['C_NoseBlur']   = {'emo': keep_original, 'id': lambda img: apply_region_single(img, get_parsing(img), 'nose', blur_processor(21)), 'nc': keep_original}  # 需要特殊处理
# 对于仅鼻子模糊，我们需要单独处理鼻子区域，这里用一个更通用的办法：对全图应用模糊，但只替换鼻子区域。
# 为了方便，直接写一个针对C组的处理函数。
def nose_blur(img_rgb):
    parsing = get_parsing(img_rgb)
    nose_mask = np.isin(parsing, [10]).astype(np.float32)
    nose_mask = cv2.dilate(nose_mask, np.ones((3,3),np.uint8), iterations=1)
    blur_img = cv2.GaussianBlur(img_rgb, (21,21), 0)
    return (nose_mask[...,None] * blur_img + (1-nose_mask[...,None]) * img_rgb).astype(np.uint8)

methods['C_NoseBlur'] = {'emo': keep_original, 'id': nose_blur, 'nc': keep_original}  # 这里把鼻子视为id区，但实际上C组仅需要处理鼻子，所以用id区函数，但id区包含眼睛和鼻子，我们只想要鼻子，需要单独函数。
# 重新定义：对于C组，我们简单使用区域处理，但id处理器只作用于鼻子，为此需在apply_region_processing中不能使用统一的id处理器，但我们可以把鼻子掩膜作为id掩膜传入？这样就需要能自定义掩膜，有点复杂。我们改为在apply_region_processing中传入自定义掩膜，但为简化，直接为C_NoseBlur写特殊处理：
def nose_blur_only(img_rgb):
    parsing = get_parsing(img_rgb)
    mask = np.isin(parsing, [10]).astype(np.float32)
    mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=1)
    blur = cv2.GaussianBlur(img_rgb, (21,21), 0)
    return (mask[...,None] * blur + (1-mask[...,None]) * img_rgb).astype(np.uint8)

methods['C_NoseBlur'] = {'emo': keep_original, 'id': nose_blur_only, 'nc': keep_original}
methods['C_EyeNoseBlur'] = {'emo': keep_original, 'id': blur_processor(21), 'nc': keep_original}
def mouthkeep_facemosaic(img_rgb):
    parsing = get_parsing(img_rgb)
    emo_mask, _, nc_mask = get_region_masks(parsing)  # 表情区保留
    # 把眼鼻区也归入非核心区一起马赛克
    mosaic = mosaic_processor(16)(img_rgb)
    id_mask = np.isin(parsing, [4,5,10]).astype(np.float32)
    total_nc = nc_mask + id_mask
    total_nc = np.clip(total_nc, 0, 1)
    return (emo_mask[...,None] * img_rgb + total_nc[...,None] * mosaic).astype(np.uint8)

methods['C_MouthKeep_FaceMosaic'] = {'emo': keep_original, 'id': lambda img: img, 'nc': mosaic_processor(16)}  # 但这样眼鼻区保留了，需要自定义，还是直接写个函数。
# 直接写函数
def mouthkeep_facemosaic_v2(img_rgb):
    parsing = get_parsing(img_rgb)
    # 表情区保留原图，其余马赛克
    emo_mask, _, _ = get_region_masks(parsing)
    mosaic = mosaic_processor(16)(img_rgb)
    return (emo_mask[...,None] * img_rgb + (1-emo_mask[...,None]) * mosaic).astype(np.uint8)
methods['C_MouthKeep_FaceMosaic'] = mouthkeep_facemosaic_v2  # 特殊，不需要三个处理器

# Strong Mosaic 和 Original 直接引用
def strong_mosaic(img): return mosaic_processor(16)(img)
methods['Strong Mosaic'] = strong_mosaic
methods['Original'] = keep_original

# 为了方便评估，定义一个统一调用接口
def process_image(img_rgb, method):
    if callable(method):
        return method(img_rgb)  # 直接是函数（如Strong Mosaic, Original）
    # 否则是一个字典，包含emo/id/nc处理器，需要解析
    parsing = get_parsing(img_rgb)
    proc_emo = method.get('emo', keep_original)
    proc_id = method.get('id', keep_original)
    proc_nc = method.get('nc', keep_original)
    return apply_region_processing(img_rgb, parsing, proc_emo, proc_id, proc_nc)

# ==================== 评估循环 ====================
results = []
for name, method in methods.items():
    print(f"\n--- 测试 {name} ---")
    emo_correct = 0
    emo_total = 0
    pro_embs = []
    for img, subj, true_label in tqdm(zip(probe_imgs, probe_subjects, probe_true_labels), total=len(probe_imgs), desc=name):
        protected = process_image(img, method)
        pred = predict_emotion(protected)
        if pred == true_label:
            emo_correct += 1
        emo_total += 1
        pro_embs.append(get_embedding(protected))
    emo_acc = emo_correct / emo_total if emo_total else 0
    _, _, top1, _ = evaluate_identity(pro_embs, probe_subjects, gallery_embs)
    results.append((name, emo_acc, top1))
    print(f"  情感 Acc: {emo_acc:.4f}, 身份 Top-1: {top1:.4f}")

# 打印汇总
print("\n" + "="*60)
print("最终对比结果")
print("="*60)
print(f"{'方法':<25}{'情感Acc':<10}{'身份Top-1':<10}")
for name, acc, top1 in results:
    print(f"{name:<25}{acc:<10.4f}{top1:<10.4f}")