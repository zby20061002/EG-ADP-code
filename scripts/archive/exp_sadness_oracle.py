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

def blur_processor(ksize):
    if ksize % 2 == 0: ksize += 1
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

probe_data = []    # (subject, true_label, img_rgb)
for _, row in df_peak.iterrows():
    img = cv2.imread(row['image_path'])
    if img is None: continue
    probe_data.append((row['subject'], code_to_idx[int(row['emotion_code'])], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

# ==================== 定义基准与 Sadness 专用参数 ====================
# 基准: S_mouthD11_brow3_blur17_nc14
base_params = {
    'brow_dilation': 3,
    'mouth_dilation': 11,
    'id_blur': 17,
    'nc_mosaic': 14
}

# Sadness 专用变体
sad_variants = {
    'Sad-V1': {'brow_dilation': 5, 'mouth_dilation': 11, 'id_blur': 11, 'nc_mosaic': 14},
    'Sad-V2': {'brow_dilation': 7, 'mouth_dilation': 15, 'id_blur': 13, 'nc_mosaic': 14},
    'Sad-V3': {'brow_dilation': 7, 'mouth_dilation': 15, 'id_blur': 11, 'nc_mosaic': 15},
    'Sad-V4': {'brow_dilation': 3, 'mouth_dilation': 17, 'id_blur': 17, 'nc_mosaic': 14},
}

# 测试方法列表：先 Base，再 Sadness 专用版本
methods = {}
methods['Base'] = {'sadness_params': base_params, 'other_params': base_params}  # 全部用基准

for var_name, sad_params in sad_variants.items():
    methods[var_name] = {
        'sadness_params': sad_params,
        'other_params': base_params
    }

results = []
for method_name, param_set in methods.items():
    print(f"\n--- 测试 {method_name} ---")
    all_preds, all_labels, probe_embs, subjects = [], [], [], []
    for subj, true_label, img_rgb in tqdm(probe_data, desc=method_name):
        # 根据 true_label 选择参数
        if true_label == 4:   # sadness
            params = param_set['sadness_params']
        else:
            params = param_set['other_params']
        
        parsing = get_parsing(img_rgb)
        emo_mask, id_mask, nc_mask = get_custom_masks(parsing,
                                                       brow_dilation=params['brow_dilation'],
                                                       mouth_dilation=params['mouth_dilation'])
        proc_emo = keep_original       # 表情区保留
        proc_id = blur_processor(params['id_blur'])
        proc_nc = mosaic_processor(params['nc_mosaic'])
        protected = apply_processing_with_masks(img_rgb, emo_mask, id_mask, nc_mask, proc_emo, proc_id, proc_nc)

        pred = predict_emotion(protected)
        all_preds.append(pred)
        all_labels.append(true_label)
        subjects.append(subj)
        probe_embs.append(get_embedding(protected))

    # 情感指标
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cls_report = classification_report(all_labels, all_preds, target_names=EMOTIONS, output_dict=True, zero_division=0)
    f1_sadness = cls_report['sadness']['f1-score']
    f1_fear = cls_report['fear']['f1-score']
    f1_disgust = cls_report['disgust']['f1-score']

    # 身份指标
    auc, eer, top1, top5 = evaluate_identity(probe_embs, subjects, gallery_embs)

    results.append({
        'Method': method_name,
        'Acc': acc, 'Macro-F1': macro_f1,
        'F1_sadness': f1_sadness, 'F1_fear': f1_fear, 'F1_disgust': f1_disgust,
        'Top-1': top1, 'AUC': auc, 'EER': eer
    })
    print(f"  Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, F1_sadness={f1_sadness:.4f}, Top-1={top1:.4f}")

# 汇总
df_res = pd.DataFrame(results)
print("\n" + "="*80)
print("Oracle Sadness-specific 实验结果")
print("="*80)
print(df_res.to_string(index=False))
df_res.to_csv("oracle_sadness_results.csv", index=False)
print("\n结果已保存至 oracle_sadness_results.csv")