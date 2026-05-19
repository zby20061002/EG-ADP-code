import os
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms
import timm
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random
import sys

# 添加 BiSeNet 路径
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 数据路径
DATA_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_PATH = os.path.join(DATA_DIR, 'metadata', 'samples.csv')
EMOTION_MODEL_PATH = "best_emotion_ckplus.pth"   # 确保该文件存在
BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"

# 输出目录
OUTPUT_DIR = "./debug_ckplus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 情感标签映射
code_to_idx = {1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
idx2emo = {v:k for k,v in code_to_idx.items()}
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
EMO2IDX = {e:i for i,e in enumerate(EMOTIONS)}

# 诊断样本数
NUM_SAMPLES = 10

# 我们将使用简化的噪声强度（sigma）替代 ε，方便调试
# (core_sigma, noncore_sigma) 基于 0-1 归一化图像
DEBUG_SIGMAS = {
    'original': (0.0, 0.0),
    'weak':     (0.001, 0.005),
    'medium':   (0.003, 0.02),
    'strong':   (0.005, 0.04),
    'very_strong': (0.01, 0.08),
}

# ==================== 加载模型 ====================
print("加载模型...")

# BiSeNet
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
state_dict = torch.load(BISENET_WEIGHT, map_location=DEVICE)
bisenet.load_state_dict(state_dict, strict=False)
bisenet = bisenet.to(DEVICE).eval()

# 情感模型
emotion_model = timm.create_model('resnet18', pretrained=False, num_classes=6)
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model = emotion_model.to(DEVICE).eval()

emo_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# FaceNet
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
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        return facenet(tensor).cpu().numpy().flatten()

def get_core_mask(img_bgr):
    """返回 0-1 float32 的核心区域掩膜，尺寸与输入相同"""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    with torch.no_grad():
        out = bisenet(tensor)
        if isinstance(out, tuple): out = out[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.isin(parsing, list({2,3,4,5,6,10,11,12,13})).astype(np.float32)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
    return mask

def add_noise_protect(img_rgb, core_sigma, noncore_sigma):
    """使用给定的 sigma（基于 0-1 图像）添加噪声"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask = get_core_mask(img_bgr)
    img = img_rgb.astype(np.float32) / 255.0
    noise = np.random.randn(*img.shape).astype(np.float32)
    # 核心区域用 core_sigma，非核心用 noncore_sigma
    core_noise = noise * core_sigma
    noncore_noise = noise * noncore_sigma
    mask_3ch = np.stack([mask]*3, axis=-1)
    combined = mask_3ch * core_noise + (1 - mask_3ch) * noncore_noise
    protected = np.clip(img + combined, 0, 1) * 255
    return protected.astype(np.uint8), mask

def psnr(img1, img2):
    """计算 PSNR，输入 0-255 uint8 图像"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def ssim(img1, img2):
    """简化版 SSIM，使用 OpenCV 实现"""
    # 转为灰度计算 SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    C1, C2 = 6.5025, 58.5225  # 常用常数
    mu1 = cv2.GaussianBlur(gray1, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11,11), 1.5)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(gray1**2, (11,11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2**2, (11,11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1*gray2, (11,11), 1.5) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# ==================== 数据准备 ====================
df_all = pd.read_csv(META_PATH)
# 测试集 peak 图像
df_test = df_all[(df_all['subset'] == 'test') & (~df_all['emotion_code'].isna())].copy()
df_test['emotion_code'] = df_test['emotion_code'].astype(int)
# 随机抽样
if len(df_test) > NUM_SAMPLES:
    df_sample = df_test.sample(NUM_SAMPLES, random_state=SEED)
else:
    df_sample = df_test

print(f"共选取 {len(df_sample)} 个测试样本")

# 同时读取所有 test neutral 用于身份相似度
df_neutral = df_all[(df_all['subset'] == 'test') & (df_all['emotion'].str.lower() == 'neutral')].copy()
neutral_embs = {}
for _, row in df_neutral.iterrows():
    subject = row['subject']
    img = cv2.imread(row['image_path'])
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        neutral_embs[subject] = get_embedding(img_rgb)

# ==================== 主诊断循环 ====================
results = []  # 存放每条样本的诊断记录
sample_dirs = []

for idx, (_, row) in enumerate(df_sample.iterrows()):
    subject = row['subject']
    img_path = row['image_path']
    fname = os.path.basename(img_path)
    true_label = code_to_idx[int(row['emotion_code'])]
    
    print(f"\n处理样本 {idx+1}/{len(df_sample)}: {subject} {true_label} ({EMOTIONS[true_label]})")
    
    # 读取原始图像 (BGR -> RGB)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 为当前样本创建专属输出目录
    sample_dir = os.path.join(OUTPUT_DIR, f"{subject}_{fname[:-4]}")
    os.makedirs(sample_dir, exist_ok=True)
    sample_dirs.append(sample_dir)
    
    # 保存原始图像
    cv2.imwrite(os.path.join(sample_dir, "original.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    # 情感预测原图
    pred_orig = predict_emotion(img_rgb)
    emb_orig = get_embedding(img_rgb)
    orig_neutral_sim = np.dot(neutral_embs.get(subject, np.zeros_like(emb_orig)), emb_orig) if subject in neutral_embs else None
    
    # 遍历不同噪声等级
    for method, (core_s, noncore_s) in DEBUG_SIGMAS.items():
        if method == 'original':
            protected = img_rgb
            mask = np.zeros(img_rgb.shape[:2], dtype=np.float32)
        else:
            protected, mask = add_noise_protect(img_rgb, core_s, noncore_s)
        
        # 保存保护图像
        cv2.imwrite(os.path.join(sample_dir, f"{method}.png"), cv2.cvtColor(protected, cv2.COLOR_RGB2BGR))
        
        # 保存掩膜可视化
        mask_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(sample_dir, f"mask_{method}.png"), mask_vis)
        # 掩膜叠加：红色区域表示核心区
        overlay = cv2.addWeighted(cv2.cvtColor(protected, cv2.COLOR_RGB2BGR), 0.7,
                                  cv2.cvtColor(cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB), 0.3, 0)
        cv2.imwrite(os.path.join(sample_dir, f"overlay_{method}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # 计算指标
        psnr_val = psnr(img_rgb, protected) if method != 'original' else np.inf
        ssim_val = ssim(img_rgb, protected) if method != 'original' else 1.0
        core_ratio = mask.mean()
        pix_mean = protected.mean()
        pix_std = protected.std()
        
        # 情感预测
        pred_prot = predict_emotion(protected)
        
        # 身份相似度
        emb_prot = get_embedding(protected)
        if subject in neutral_embs:
            neutral_sim = np.dot(neutral_embs[subject], emb_prot)
        else:
            neutral_sim = None
        
        # 记录
        results.append({
            'sample': f"{subject}_{fname}",
            'method': method,
            'true_label': true_label,
            'pred_orig': pred_orig,
            'pred': pred_prot,
            'core_ratio': core_ratio,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'pixel_mean': pix_mean,
            'pixel_std': pix_std,
            'neutral_sim': neutral_sim,
            'core_sigma': core_s,
            'noncore_sigma': noncore_s,
        })
        
        # 输出简要信息
        print(f"  {method}: acc={pred_prot==true_label}, PSNR={psnr_val:.1f}, SSIM={ssim_val:.3f}, core_ratio={core_ratio:.3f}, sim={neutral_sim:.3f}" if neutral_sim else f"  {method}: acc={pred_prot==true_label}, PSNR={psnr_val:.1f}, SSIM={ssim_val:.3f}")

# ==================== 汇总报告 ====================
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "debug_results.csv"), index=False)

# 输出摘要
print("\n" + "="*50)
print("诊断摘要")
print("="*50)
for method in DEBUG_SIGMAS.keys():
    sub = df_results[df_results['method'] == method]
    if len(sub) == 0: continue
    acc = (sub['pred'] == sub['true_label']).mean()
    avg_psnr = sub['psnr'].mean()
    avg_ssim = sub['ssim'].mean()
    avg_core = sub['core_ratio'].mean()
    avg_sim = sub['neutral_sim'].dropna().mean() if not sub['neutral_sim'].dropna().empty else None
    print(f"\n{method}:")
    print(f"  情感准确率: {acc:.2%}")
    print(f"  平均 PSNR: {avg_psnr:.1f}")
    print(f"  平均 SSIM: {avg_ssim:.3f}")
    print(f"  平均核心区域占比: {avg_core:.3f}")
    if avg_sim is not None:
        print(f"  平均同身份相似度: {avg_sim:.3f}")

print(f"\n详细结果已保存至: {OUTPUT_DIR}")
print("请检查各样本目录下的图像，确认掩膜是否正确覆盖眼、眉、嘴、鼻区域。")