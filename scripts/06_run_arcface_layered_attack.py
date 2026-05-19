import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from torchvision.transforms import functional as F
from sklearn.metrics import roc_auc_score, roc_curve
import insightface
from insightface.app import FaceAnalysis

# 添加 BiSeNet 路径
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ==================== 配置 ====================
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
INSIGHTFACE_ROOT = r'F:\python\models\insightface'
PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_CSV = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')
BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"
FAIR_RESULTS_CSV = "5fold_fair_baseline_results.csv"  # 可选，用于权衡图
OUTPUT_DIR = "./final_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 加载模型 ====================
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
bisenet.load_state_dict(__import__('torch').load(BISENET_WEIGHT, map_location=DEVICE), strict=False)
bisenet.eval().to(DEVICE)

print("加载 ArcFace...")
arcface_app = FaceAnalysis(name='antelopev2', root=INSIGHTFACE_ROOT,
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
arcface_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(224,224))

# ==================== 工具函数 ====================
def get_parsing(img_rgb):
    h, w = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb).resize((512,512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    with __import__('torch').no_grad():
        out = bisenet(tensor)
        out = out[0] if isinstance(out, tuple) else out
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    return cv2.resize(parsing.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

def blur_processor(ksize):
    if ksize % 2 == 0: ksize += 1
    def func(img_rgb): return cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)
    return func

def mosaic_processor(block):
    def func(img_rgb):
        h,w = img_rgb.shape[:2]
        small = cv2.resize(img_rgb, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
    return func

keep_original = lambda img: img.copy()

def apply_ours(img_rgb):
    parsing = get_parsing(img_rgb)
    brow = np.isin(parsing, [2,3]).astype(np.float32)
    mouth = np.isin(parsing, [11,12,13]).astype(np.float32)
    brow = cv2.dilate(brow, np.ones((3,3),np.uint8), iterations=1)
    mouth = cv2.dilate(mouth, np.ones((11,11),np.uint8), iterations=1)
    emo_mask = np.clip(brow + mouth, 0, 1)
    id_mask = np.isin(parsing, [4,5,10]).astype(np.float32)
    id_mask = cv2.dilate(id_mask, np.ones((3,3),np.uint8), iterations=1)
    nc_mask = np.clip(1.0 - emo_mask - id_mask, 0, 1)
    img_emo = keep_original(img_rgb)
    img_id = blur_processor(17)(img_rgb)
    img_nc = mosaic_processor(14)(img_rgb)
    return (emo_mask[...,None]*img_emo + id_mask[...,None]*img_id + nc_mask[...,None]*img_nc).astype(np.uint8)

# 各种保护方法
protect_methods = {
    'Original': keep_original,
    'Gaussian Blur': lambda img: blur_processor(21)(img),
    'Mosaic b8': lambda img: mosaic_processor(8)(img),
    'Mosaic b12': lambda img: mosaic_processor(12)(img),
    'Strong Mosaic': lambda img: mosaic_processor(16)(img),
    'Ours': apply_ours
}

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[idx]

# ==================== 数据准备 ====================
df = pd.read_csv(META_CSV)
df_peak = df[~df['emotion_code'].isna()].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

# === 关键修复：仅使用原始 test split 的 subject ===
test_subjects = set(df[df['subset'] == 'test']['subject'].unique())
print(f"原始 test subjects 数量: {len(test_subjects)}")

# Gallery：仅使用 test subjects 的 neutral 图像
df_neutral_test = df[(df['emotion'].str.lower() == 'neutral') & (df['subject'].isin(test_subjects))]
gallery_embs = {}
for subj, group in df_neutral_test.groupby('subject'):
    row = group.iloc[0]
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gallery_embs[subj] = arcface_app.get(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))[0].normed_embedding
print(f"Gallery 样本数: {len(gallery_embs)}")

# Probe：只使用 test subjects 的 peak 图像
df_probe = df_peak[df_peak['subject'].isin(test_subjects)].copy()
print(f"Probe 样本数: {len(df_probe)}")

# ==================== 实验1：ArcFace 检测率统计 ====================
print("\n===== 实验1：ArcFace 检测率统计 =====")
results_detection = defaultdict(lambda: {'total':0, 'detected':0})

for _, row in tqdm(df_probe.iterrows(), total=len(df_probe), desc="检测率统计"):
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for method_name, apply_fn in protect_methods.items():
        protected = apply_fn(img_rgb)
        img_bgr = cv2.cvtColor(protected, cv2.COLOR_RGB2BGR)
        faces = arcface_app.get(img_bgr)
        results_detection[method_name]['total'] += 1
        if faces:
            results_detection[method_name]['detected'] += 1

detection_table = []
for method_name, stats in results_detection.items():
    rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
    detection_table.append({'Method': method_name, 'Detection Rate': f"{rate:.4f}"})

df_detection = pd.DataFrame(detection_table)
print(df_detection.to_string(index=False))
df_detection.to_csv(os.path.join(OUTPUT_DIR, "arcface_detection_rate.csv"), index=False)

# ==================== 实验2：Detected-Only 识别结果（修复后） ====================
print("\n===== 实验2：ArcFace Detected-Only 识别结果 =====")
results_detected = defaultdict(list)

for method_name, apply_fn in protect_methods.items():
    print(f"处理 {method_name}...")
    for _, row in tqdm(df_probe.iterrows(), total=len(df_probe), desc=f"{method_name}"):
        subject = row['subject']
        img = cv2.imread(row['image_path'])
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        protected = apply_fn(img_rgb)
        img_bgr = cv2.cvtColor(protected, cv2.COLOR_RGB2BGR)
        faces = arcface_app.get(img_bgr)
        if faces:
            emb = faces[0].normed_embedding
            if subject in gallery_embs:  # 确保有 gallery
                results_detected[method_name].append((subject, emb))

table_detected = []
for method_name, samples in results_detected.items():
    if not samples:
        table_detected.append({'Method': method_name, 'Detected_Count': 0, 'Top-1_detected': 0, 'AUC_detected': 0, 'EER_detected': 0})
        continue

    genuine, impostor, ranks = [], [], []
    for subject, probe_emb in samples:
        genuine.append(np.dot(gallery_embs[subject], probe_emb))
        for other_subj, other_emb in gallery_embs.items():
            if other_subj != subject:
                impostor.append(np.dot(other_emb, probe_emb))
        sims = {s: np.dot(gallery_embs[s], probe_emb) for s in gallery_embs}
        sorted_subjs = sorted(sims, key=sims.get, reverse=True)
        rank = sorted_subjs.index(subject) + 1
        ranks.append(rank)

    y_true = [1]*len(genuine) + [0]*len(impostor)
    y_score = genuine + impostor
    auc = roc_auc_score(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    top1 = sum(r == 1 for r in ranks) / len(ranks)

    table_detected.append({
        'Method': method_name,
        'Detected_Count': len(samples),
        'Top-1_detected': f"{top1:.4f}",
        'AUC_detected': f"{auc:.4f}",
        'EER_detected': f"{eer:.4f}"
    })

df_detected = pd.DataFrame(table_detected)
print("\n" + df_detected.to_string(index=False))
df_detected.to_csv(os.path.join(OUTPUT_DIR, "arcface_detected_only.csv"), index=False)

# 合并分层表
detection_csv = os.path.join(OUTPUT_DIR, "arcface_detection_rate.csv")
if os.path.exists(detection_csv):
    df_det_rate = pd.read_csv(detection_csv)
    df_merged = pd.merge(df_det_rate, df_detected, on='Method', how='left')
    print("\n===== 分层结果总表 =====")
    print(df_merged.to_string(index=False))
    df_merged.to_csv(os.path.join(OUTPUT_DIR, "arcface_layered_results.csv"), index=False)

# ==================== 实验3：隐私-效用权衡图 ====================
print("\n===== 实验3：隐私-效用权衡图 =====")
if os.path.exists(FAIR_RESULTS_CSV):
    df_fair = pd.read_csv(FAIR_RESULTS_CSV)
    methods = []
    for _, row in df_fair.iterrows():
        method_full = row['Method']
        method_name = method_full.replace('_Fair', '')
        acc = float(row['Acc'].split('±')[0]) if '±' in str(row['Acc']) else float(row['Acc'])
        macro_f1 = float(row['Macro-F1'].split('±')[0]) if '±' in str(row['Macro-F1']) else float(row['Macro-F1'])
        top1_fn = float(row['Top-1_FaceNet'].split('±')[0]) if '±' in str(row['Top-1_FaceNet']) else float(row['Top-1_FaceNet'])
        top1_af = float(row['Top-1_ArcFace'].split('±')[0]) if '±' in str(row['Top-1_ArcFace']) else float(row['Top-1_ArcFace'])
        methods.append({
            'Method': method_name,
            'Macro-F1': macro_f1,
            'FaceNet Top-1': top1_fn,
            'ArcFace Top-1': top1_af
        })
    df_plot = pd.DataFrame(methods)
else:
    df_plot = pd.DataFrame([
        {'Method': 'Original', 'Macro-F1': 0.88, 'FaceNet Top-1': 1.0, 'ArcFace Top-1': 1.0},
        {'Method': 'Gaussian Blur', 'Macro-F1': 0.86, 'FaceNet Top-1': 0.99, 'ArcFace Top-1': 1.0},
        {'Method': 'Mosaic b8', 'Macro-F1': 0.40, 'FaceNet Top-1': 0.77, 'ArcFace Top-1': 0.04},
        {'Method': 'Mosaic b12', 'Macro-F1': 0.24, 'FaceNet Top-1': 0.34, 'ArcFace Top-1': 0.04},
        {'Method': 'Strong Mosaic', 'Macro-F1': 0.19, 'FaceNet Top-1': 0.09, 'ArcFace Top-1': 0.04},
        {'Method': 'Ours', 'Macro-F1': 0.88, 'FaceNet Top-1': 0.66, 'ArcFace Top-1': 0.47}
    ])

plt.figure(figsize=(8,6))
for _, row in df_plot.iterrows():
    plt.scatter(row['FaceNet Top-1'], row['Macro-F1'], label=row['Method'], s=100)
    plt.text(row['FaceNet Top-1']+0.01, row['Macro-F1']+0.01, row['Method'], fontsize=9)
plt.xlabel('Identity Top-1 (FaceNet)')
plt.ylabel('Expression Macro-F1')
plt.title('Privacy-Utility Trade-off (FaceNet)')
plt.grid(alpha=0.3)
plt.xlim(0, 1.1)
plt.ylim(0, 1.0)
plt.savefig(os.path.join(OUTPUT_DIR, "tradeoff_facenet.png"), dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,6))
for _, row in df_plot.iterrows():
    plt.scatter(row['ArcFace Top-1'], row['Macro-F1'], label=row['Method'], s=100)
    plt.text(row['ArcFace Top-1']+0.01, row['Macro-F1']+0.01, row['Method'], fontsize=9)
plt.xlabel('Identity Top-1 (ArcFace)')
plt.ylabel('Expression Macro-F1')
plt.title('Privacy-Utility Trade-off (ArcFace)')
plt.grid(alpha=0.3)
plt.xlim(0, 1.1)
plt.ylim(0, 1.0)
plt.savefig(os.path.join(OUTPUT_DIR, "tradeoff_arcface.png"), dpi=150, bbox_inches='tight')
plt.close()

# ==================== 实验4：可视化 ====================
print("\n===== 实验4：生成可视化对比图 =====")
np.random.seed(42)
samples = df_probe.sample(min(5, len(df_probe)), random_state=42)
method_list = ['Original', 'Mask Overlay', 'Gaussian Blur', 'Mosaic b12', 'Strong Mosaic', 'Ours']

for idx, (_, row) in enumerate(samples.iterrows()):
    img_path = row['image_path']
    img = cv2.imread(img_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    parsing = get_parsing(img_rgb)
    
    fig, axes = plt.subplots(1, len(method_list), figsize=(18, 3))
    for ax, method in zip(axes, method_list):
        if method == 'Original':
            ax.imshow(img_rgb)
        elif method == 'Mask Overlay':
            overlay = img_rgb.copy()
            brow = np.isin(parsing, [2,3])
            mouth = np.isin(parsing, [11,12,13])
            eye = np.isin(parsing, [4,5,10])
            overlay[brow | mouth] = [0, 255, 0]
            overlay[eye] = [255, 165, 0]
            noncore = ~(brow | mouth | eye)
            overlay[noncore] = [0, 0, 255]
            ax.imshow(overlay)
        elif method == 'Ours':
            ax.imshow(apply_ours(img_rgb))
        else:
            apply_fn = protect_methods[method]
            ax.imshow(apply_fn(img_rgb))
        ax.set_title(method, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"visualization_sample{idx+1}.png"), dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n所有结果已保存至 {OUTPUT_DIR}")