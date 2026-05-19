import os, sys, cv2, numpy as np, torch, pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
import timm
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, f1_score
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
idx2emo = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise'}
EMOTIONS = list(idx2emo.values())

# ==================== 模型加载 ====================
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
bisenet.load_state_dict(torch.load(BISENET_WEIGHT, map_location=DEVICE), strict=False)
bisenet.eval().to(DEVICE)

print("加载情感模型...")
emotion_model = timm.create_model('resnet18', pretrained=False, num_classes=6)
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model.eval().to(DEVICE)
emo_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

print("加载 FaceNet...")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ==================== 工具函数 ====================
def predict_emotion(img_rgb):
    tensor = emo_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return emotion_model(tensor).argmax(1).item()

def get_embedding(img_rgb):
    img = cv2.resize(img_rgb, (160,160))
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)/255.
    tensor = (tensor-0.5)/0.5
    with torch.no_grad():
        return facenet(tensor).cpu().numpy().flatten()

def get_parsing(img_rgb):
    h,w = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb).resize((512,512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    with torch.no_grad():
        out = bisenet(tensor)
        out = out[0] if isinstance(out, tuple) else out
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    return cv2.resize(parsing.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

def get_region_masks(parsing, emo_dilation=3):
    """统一膨胀版本，眉毛+嘴唇一起膨胀"""
    emo_labels = {2,3,11,12,13}          # 眉毛、嘴唇
    id_labels = {4,5,10}                 # 眼睛、鼻子
    emo_mask = np.isin(parsing, list(emo_labels)).astype(np.float32)
    id_mask = np.isin(parsing, list(id_labels)).astype(np.float32)
    kernel = np.ones((emo_dilation, emo_dilation), np.uint8)
    emo_mask = cv2.dilate(emo_mask, kernel, iterations=1)
    id_mask = cv2.dilate(id_mask, np.ones((3,3),np.uint8), iterations=1)
    nc_mask = np.clip(1.0 - emo_mask - id_mask, 0, 1)
    return emo_mask, id_mask, nc_mask

def get_custom_masks(parsing, brow_dilation=3, mouth_dilation=3):
    """分离眉毛和嘴部膨胀"""
    brow_mask = np.isin(parsing, [2,3]).astype(np.float32)
    mouth_mask = np.isin(parsing, [11,12,13]).astype(np.float32)
    kernel_brow = np.ones((brow_dilation, brow_dilation), np.uint8)
    kernel_mouth = np.ones((mouth_dilation, mouth_dilation), np.uint8)
    brow_mask = cv2.dilate(brow_mask, kernel_brow, iterations=1)
    mouth_mask = cv2.dilate(mouth_mask, kernel_mouth, iterations=1)
    emo_mask = np.clip(brow_mask + mouth_mask, 0, 1)
    id_mask = np.isin(parsing, [4,5,10]).astype(np.float32)
    id_mask = cv2.dilate(id_mask, np.ones((3,3),np.uint8), iterations=1)
    nc_mask = np.clip(1.0 - emo_mask - id_mask, 0, 1)
    return emo_mask, id_mask, nc_mask

def noise_processor(sigma):
    def func(img_rgb):
        img = img_rgb.astype(np.float32)/255.
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        return (np.clip(img + noise, 0, 1)*255).astype(np.uint8)
    return func

def blur_processor(ksize):
    # 确保核大小为奇数
    if ksize % 2 == 0:
        ksize += 1
    def func(img_rgb):
        return cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)
    return func

def mosaic_processor(block):
    def func(img_rgb):
        h,w = img_rgb.shape[:2]
        small = cv2.resize(img_rgb, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
    return func

keep_original = lambda img: img.copy()

def apply_processing_with_masks(img_rgb, emo_mask, id_mask, nc_mask, proc_emo, proc_id, proc_nc):
    img_emo = proc_emo(img_rgb)
    img_id = proc_id(img_rgb)
    img_nc = proc_nc(img_rgb)
    result = emo_mask[...,None]*img_emo + id_mask[...,None]*img_id + nc_mask[...,None]*img_nc
    return result.astype(np.uint8)

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
        return 0,0,0,0
    y_true = [1]*len(genuine) + [0]*len(impostor)
    y_score = genuine + impostor
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    top1 = sum(r==1 for r in ranks)/len(ranks)
    top5 = sum(r<=5 for r in ranks)/len(ranks)
    return auc, eer, top1, top5

# ==================== 数据准备 ====================
df = pd.read_csv(META_PATH)
df_neutral = df[(df['subset']=='test') & (df['emotion'].str.lower()=='neutral')]
df_peak = df[(df['subset']=='test') & (~df['emotion_code'].isna())].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

gallery_embs = {}
for _, row in tqdm(df_neutral.iterrows(), desc="Gallery"):
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gallery_embs[row['subject']] = get_embedding(img_rgb)

probe_data = []
for _, row in df_peak.iterrows():
    img = cv2.imread(row['image_path'])
    if img is None: continue
    probe_data.append((row['subject'], code_to_idx[int(row['emotion_code'])], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

# ==================== 定义配置 ====================
configs = []

# 方案1: 固定 noncore=14, dilation=3, emotion keep, 微调 id blur (16-20) → 实际 kernel 自动转为奇数
for raw_k in [16, 17, 18, 19, 20]:
    # 计算出实际的模糊核大小（奇数）
    actual_k = raw_k if raw_k % 2 == 1 else raw_k + 1
    name = f"N_blur{raw_k}_nc14_d3"   # 保持名称中的原始数字，便于区分
    configs.append({
        'name': name,
        'proc_emo': keep_original,
        'proc_id': blur_processor(raw_k),   # 内部会自动转为奇数
        'proc_nc': mosaic_processor(14),
        'mask_type': 'unified',
        'emo_dilation': 3,
        'brow_dilation': None,
        'mouth_dilation': None
    })

# 方案2: 固定 id blur=17, emotion keep, dilation=3, 微调 noncore mosaic (13-16)
for block in [13, 14, 15, 16]:
    name = f"M_nc{block}_blur17_d3"
    configs.append({
        'name': name,
        'proc_emo': keep_original,
        'proc_id': blur_processor(17),
        'proc_nc': mosaic_processor(block),
        'mask_type': 'unified',
        'emo_dilation': 3,
        'brow_dilation': None,
        'mouth_dilation': None
    })

# 方案3: 分离 brow 和 mouth 膨胀，固定 brow=3, id blur=17, noncore mosaic=14，微调 mouth dilation
for mouth_dil in [7, 9, 11]:
    name = f"S_mouthD{mouth_dil}_brow3_blur17_nc14"
    configs.append({
        'name': name,
        'proc_emo': keep_original,
        'proc_id': blur_processor(17),
        'proc_nc': mosaic_processor(14),
        'mask_type': 'split',
        'brow_dilation': 3,
        'mouth_dilation': mouth_dil
    })

# 方案3 额外一组: brow=3, mouth=9, id blur=18, nc=14
configs.append({
    'name': 'S_mouthD9_brow3_blur18_nc14',
    'proc_emo': keep_original,
    'proc_id': blur_processor(18),   # 18 → 19
    'proc_nc': mosaic_processor(14),
    'mask_type': 'split',
    'brow_dilation': 3,
    'mouth_dilation': 9
})

results = []
for cfg in configs:
    name = cfg['name']
    print(f"\n--- 测试 {name} ---")
    all_preds, all_labels, probe_embs, subjects = [], [], [], []
    for subj, true_label, img_rgb in tqdm(probe_data, desc=name):
        parsing = get_parsing(img_rgb)
        if cfg['mask_type'] == 'unified':
            emo_mask, id_mask, nc_mask = get_region_masks(parsing, emo_dilation=cfg['emo_dilation'])
        else:  # split
            emo_mask, id_mask, nc_mask = get_custom_masks(parsing,
                                                           brow_dilation=cfg['brow_dilation'],
                                                           mouth_dilation=cfg['mouth_dilation'])
        protected = apply_processing_with_masks(img_rgb, emo_mask, id_mask, nc_mask,
                                                cfg['proc_emo'], cfg['proc_id'], cfg['proc_nc'])
        pred = predict_emotion(protected)
        all_preds.append(pred)
        all_labels.append(true_label)
        subjects.append(subj)
        probe_embs.append(get_embedding(protected))

    # 情感指标
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cls_report = classification_report(all_labels, all_preds, target_names=EMOTIONS, output_dict=True, zero_division=0)
    per_class_f1 = {emo: cls_report[emo]['f1-score'] for emo in EMOTIONS}

    # 身份指标
    auc, eer, top1, top5 = evaluate_identity(probe_embs, subjects, gallery_embs)

    results.append({
        'Method': name,
        'Acc': acc, 'Macro-F1': macro_f1,
        **{f'F1_{e}': per_class_f1[e] for e in EMOTIONS},
        'AUC': auc, 'EER': eer, 'Top-1': top1, 'Top-5': top5
    })
    print(f"  Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, Top-1={top1:.4f}, AUC={auc:.4f}")

df_res = pd.DataFrame(results)
print("\n" + "="*100)
print("最终评估结果")
print("="*100)
print(df_res.to_string(index=False))
df_res.to_csv("fine_tune_v2_results.csv", index=False)
print("\n结果已保存至 fine_tune_v2_results.csv")