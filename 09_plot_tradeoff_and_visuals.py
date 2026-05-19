import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
import sys
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet
import insightface
from insightface.app import FaceAnalysis

# ==================== 配置 ====================
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
INSIGHTFACE_ROOT = r'F:\python\models\insightface'
PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_CSV = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')
BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"
FAIR_RESULTS_CSV = "5fold_fair_baseline_results.csv"  # 前序实验生成的结果文件
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

# ==================== 实验1：ArcFace 检测率统计 ====================
print("\n===== 实验1：ArcFace 检测率统计 =====")
df_meta = pd.read_csv(META_CSV)
df_peak = df_meta[~df_meta['emotion_code'].isna()].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

# 使用全部测试数据（可随机采样加快速度，这里用全部）
results_detection = defaultdict(lambda: {'total':0, 'detected':0})

for _, row in tqdm(df_peak.iterrows(), total=len(df_peak), desc="检测率统计"):
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

# ==================== 实验2：隐私-效用权衡图 ====================
print("\n===== 实验2：隐私-效用权衡图 =====")
# 从 fair baseline 结果中读取数据（假设文件已存在）
if os.path.exists(FAIR_RESULTS_CSV):
    df_fair = pd.read_csv(FAIR_RESULTS_CSV)
    # 提取方法名和需要的指标（格式如 "Method Fair"）
    methods = []
    for _, row in df_fair.iterrows():
        method_full = row['Method']
        # 从 "Original_Fair" 提取 "Original"
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
    # 若前序CSV不存在，使用硬编码的示例数据（请根据实际值替换）
    print("警告：未找到 fair baseline CSV，使用示例数据绘图。")
    df_plot = pd.DataFrame([
        {'Method': 'Original', 'Macro-F1': 0.88, 'FaceNet Top-1': 1.0, 'ArcFace Top-1': 1.0},
        {'Method': 'Gaussian Blur', 'Macro-F1': 0.86, 'FaceNet Top-1': 0.99, 'ArcFace Top-1': 1.0},
        {'Method': 'Mosaic b8', 'Macro-F1': 0.40, 'FaceNet Top-1': 0.77, 'ArcFace Top-1': 0.04},
        {'Method': 'Mosaic b12', 'Macro-F1': 0.24, 'FaceNet Top-1': 0.34, 'ArcFace Top-1': 0.04},
        {'Method': 'Strong Mosaic', 'Macro-F1': 0.19, 'FaceNet Top-1': 0.09, 'ArcFace Top-1': 0.04},
        {'Method': 'Ours', 'Macro-F1': 0.88, 'FaceNet Top-1': 0.66, 'ArcFace Top-1': 0.47}
    ])

# 绘制 FaceNet 权衡图
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

# 绘制 ArcFace 权衡图
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

print(f"权衡图已保存至 {OUTPUT_DIR}")

# ==================== 实验3：可视化 ====================
print("\n===== 实验3：生成可视化对比图 =====")
# 随机选取 5 个测试样本（固定种子可复现）
np.random.seed(42)
all_test = df_meta[df_meta['subset'] == 'test']  # 使用原划分的测试集
peak_test = all_test[~all_test['emotion_code'].isna()].copy()
samples = peak_test.sample(min(5, len(peak_test)), random_state=42)

method_list = ['Original', 'Mask Overlay', 'Gaussian Blur', 'Mosaic b12', 'Strong Mosaic', 'Ours']

for idx, (_, row) in enumerate(samples.iterrows()):
    img_path = row['image_path']
    img = cv2.imread(img_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    parsing = get_parsing(img_rgb)  # 语义解析图
    
    fig, axes = plt.subplots(1, len(method_list), figsize=(18, 3))
    for ax, method in zip(axes, method_list):
        if method == 'Original':
            ax.imshow(img_rgb)
        elif method == 'Mask Overlay':
            # 生成彩色掩膜叠加图
            overlay = img_rgb.copy()
            brow = np.isin(parsing, [2,3])
            mouth = np.isin(parsing, [11,12,13])
            eye = np.isin(parsing, [4,5,10])
            overlay[brow | mouth] = [0, 255, 0]    # 绿色：表情保留区
            overlay[eye] = [255, 165, 0]            # 橙色：身份敏感区
            noncore = ~(brow | mouth | eye)
            overlay[noncore] = [0, 0, 255]          # 蓝色：非核心扰动区
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

print(f"可视化图片已保存至 {OUTPUT_DIR}")
print("所有实验完成。")