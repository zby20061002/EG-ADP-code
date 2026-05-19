import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F

# 导入 BiSeNet 模型（请确保在 face-parsing 目录下运行，或修改 sys.path）
import sys
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ------- 配置 -------
DATA_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_PATH = os.path.join(DATA_DIR, 'metadata', 'samples.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

OUTPUT_BASE = r"F:\python\2024218729zby_ML\EG-ADP\data\protected_ckplus"
WEIGHT_PATH = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 表情列表（6类，顺序与训练一致）
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}

# EG-ADP 配置
EPSILON_CONFIGS = {
    'egadp_c1': (50, 40, 30, 20),
    'egadp_c2': (60, 40, 25, 15),
    'egadp_c3': (40, 30, 20, 10),
    'egadp_c4': (70, 50, 40, 30),
}

# 输出子目录
METHODS = ['original'] + list(EPSILON_CONFIGS.keys())
for m in METHODS:
    os.makedirs(os.path.join(OUTPUT_BASE, m), exist_ok=True)

# 加载 BiSeNet
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
bisenet.load_state_dict(state_dict, strict=False)
bisenet.eval().to(DEVICE)

# ---------- EG-ADP 类 ----------
class EG_ADP:
    def __init__(self, ec, ep, en, eneg, delta=1e-5):
        self.delta = delta
        self.eps_core, self.eps_pos, self.eps_neu, self.eps_neg = ec, ep, en, eneg
        self.pos_set = {EMO2IDX['happiness']}
        self.neu_set = {EMO2IDX['surprise']}
        self.neg_set = {EMO2IDX[e] for e in ['anger', 'disgust', 'fear', 'sadness']}

    def _noise_std(self, eps):
        return np.sqrt(2 * np.log(1.25 / self.delta)) / eps

    def _get_eps_nc(self, emo):
        if emo in self.pos_set: return self.eps_pos
        elif emo in self.neu_set: return self.eps_neu
        else: return self.eps_neg

    def _core_mask(self, img_bgr):
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
        tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        with torch.no_grad():
            out = bisenet(tensor)
            if isinstance(out, tuple): out = out[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.isin(parsing, list({2, 3, 4, 5, 6, 10, 11, 12, 13})).astype(np.float32)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return mask

    def protect(self, img_rgb, emo):
        img = img_rgb.astype(np.float32) / 255.0
        mask = self._core_mask(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        s_core = self._noise_std(self.eps_core)
        s_nc = self._noise_std(self._get_eps_nc(emo))
        noise = np.random.randn(*img.shape).astype(np.float32)
        combined = np.stack([mask], -1) * (noise * s_core) + (1 - np.stack([mask], -1)) * (noise * s_nc)
        noisy = np.clip(img + combined, 0, 1) * 255
        return noisy.astype(np.uint8)

# ---------- 读取测试集 peak 图像 ----------
df_all = pd.read_csv(META_PATH)
# 取测试集且有情绪标签的行（即 peak 图像）
df_test = df_all[(df_all['subset'] == 'test') & (~df_all['emotion_code'].isna())].copy()
print(f"测试集 peak 图像: {len(df_test)} 张")

# 处理
for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="生成保护图像"):
    img_path = row['image_path']
    emotion_code = int(row['emotion_code'])
    # 映射为 0-5 索引
    label_idx = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}[emotion_code]
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 文件名: 保留原始文件名（例如 S005_001_peak.png）
    fname = os.path.basename(img_path)
    
    # 保存原图 (BGR)
    cv2.imwrite(os.path.join(OUTPUT_BASE, 'original', fname), img)
    
    # 保存各 EG-ADP 配置
    for cfg_name, (ec, ep, en, eneg) in EPSILON_CONFIGS.items():
        eg = EG_ADP(ec, ep, en, eneg)
        protected = eg.protect(img_rgb, label_idx)
        cv2.imwrite(os.path.join(OUTPUT_BASE, cfg_name, fname), cv2.cvtColor(protected, cv2.COLOR_RGB2BGR))

print("保护图像生成完成。")