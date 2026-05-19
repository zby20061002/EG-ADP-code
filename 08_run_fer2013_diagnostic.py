import os
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from PIL import Image
from torchvision.transforms import functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
import sys
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FER2013 测试集路径
TEST_ROOT = r"F:\python\2024218729zby_ML\EG-ADP\data\FER-2013\test"
BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"
OUTPUT_CSV = "fer2013_adaptive_results.csv"

EMOTION_MAP = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'neutral': 4, 'sad': 5, 'surprise': 6
}

# ==================== 加载 BiSeNet ====================
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
bisenet.load_state_dict(torch.load(BISENET_WEIGHT, map_location=DEVICE), strict=False)
bisenet.eval().to(DEVICE)

# ==================== 加载情感模型 ====================
print("加载情感模型...")
model_name = "trpakov/vit-face-expression"
processor = AutoImageProcessor.from_pretrained(model_name)
emotion_model = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE).eval()

def predict_emotion(img_rgb):
    inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

# ==================== 自适应保护函数 ====================
def get_parsing(img_rgb):
    """获取语义解析图（先放大再送入 BiSeNet，再缩放回原图尺寸）"""
    h, w = img_rgb.shape[:2]
    # 放大到 512 附近
    scale = max(1, 512 // max(h, w))
    img_large = cv2.resize(img_rgb, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    pil = Image.fromarray(img_large).resize((512, 512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    with torch.no_grad():
        out = bisenet(tensor)
        out = out[0] if isinstance(out, tuple) else out
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    # 缩放回原图尺寸
    parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return parsing

def apply_ours_adaptive(img_rgb, blur_k, mosaic_block):
    """
    使用指定的模糊核和马赛克块大小进行保护
    blur_k: 高斯模糊核大小 (奇数)
    mosaic_block: 马赛克块大小
    """
    parsing = get_parsing(img_rgb)
    # 表情区：眉毛、嘴唇
    brow = np.isin(parsing, [2, 3]).astype(np.float32)
    mouth = np.isin(parsing, [11, 12, 13]).astype(np.float32)
    brow = cv2.dilate(brow, np.ones((3, 3), np.uint8), iterations=1)
    mouth = cv2.dilate(mouth, np.ones((5, 5), np.uint8), iterations=1)
    emo_mask = np.clip(brow + mouth, 0, 1)
    # 眼鼻区
    id_mask = np.isin(parsing, [4, 5, 10]).astype(np.float32)
    id_mask = cv2.dilate(id_mask, np.ones((3, 3), np.uint8), iterations=1)
    # 非核心区
    nc_mask = np.clip(1.0 - emo_mask - id_mask, 0, 1)

    # 各区域处理
    img_emo = img_rgb.copy()
    # 眼鼻模糊
    if blur_k % 2 == 0:
        blur_k += 1
    img_id = cv2.GaussianBlur(img_rgb, (blur_k, blur_k), 0)
    # 非核心马赛克
    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (w // mosaic_block, h // mosaic_block), interpolation=cv2.INTER_LINEAR)
    img_nc = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    result = (emo_mask[..., None] * img_emo +
              id_mask[..., None] * img_id +
              nc_mask[..., None] * img_nc).astype(np.uint8)
    return result

# ==================== 收集图像 ====================
print("扫描 FER2013 测试集...")
test_images = []  # (path, label_idx)
if not os.path.isdir(TEST_ROOT):
    print(f"错误：目录 {TEST_ROOT} 不存在！")
    exit(1)

subdirs = [d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))]
folder_to_idx = {}
for sub in subdirs:
    sub_lower = sub.lower()
    if sub_lower in EMOTION_MAP:
        folder_to_idx[sub] = EMOTION_MAP[sub_lower]

if not folder_to_idx:
    print("未找到有效情绪文件夹！")
    exit(1)

for folder, label_idx in folder_to_idx.items():
    folder_path = os.path.join(TEST_ROOT, folder)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append((os.path.join(folder_path, fname), label_idx))

print(f"测试图片总数: {len(test_images)}")
if len(test_images) == 0:
    print("错误：未找到任何图片。")
    exit(1)

# ==================== 评估不同配置 ====================
# 定义配置：名称, 模糊核, 马赛克块
configs = [
    ('Original', None, None),  # 不处理
    ('F1 (blur=3, mosaic=3)', 3, 3),
    ('F2 (blur=5, mosaic=3)', 5, 3),
    ('F3 (blur=5, mosaic=4)', 5, 4),
    ('F4 (blur=7, mosaic=4)', 7, 4),
]

results = []

for name, blur_k, mosaic_block in configs:
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for path, true_idx in tqdm(test_images, desc=f"Evaluating {name}"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # 转为 RGB（三通道）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if name == 'Original':
            protected = img_rgb
        else:
            protected = apply_ours_adaptive(img_rgb, blur_k, mosaic_block)
        pred = predict_emotion(protected)
        all_preds.append(pred)
        all_labels.append(true_idx)
        if pred == true_idx:
            correct += 1
        total += 1
    acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    results.append({'Method': name, 'Acc': f"{acc:.4f}", 'Macro-F1': f"{macro_f1:.4f}"})
    print(f"{name}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}")

# 保存结果
df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))
df.to_csv(OUTPUT_CSV, index=False)
print(f"结果已保存至 {OUTPUT_CSV}")