import os, sys, cv2, torch, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from torchvision.transforms import functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
from facenet_pytorch import InceptionResnetV1

# 添加 BiSeNet 路径
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ==================== 全局配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
FOLDS = 5

PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_CSV = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')

BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"
FINAL_MODEL_DIR = "./final_models_5fold"
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

code_to_idx = {1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
idx2emo = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise'}
EMOTIONS = list(idx2emo.values())

# ==================== 加载 BiSeNet ====================
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
bisenet.load_state_dict(torch.load(BISENET_WEIGHT, map_location=DEVICE), strict=False)
bisenet.eval().to(DEVICE)

# ==================== 掩膜与保护函数 ====================
def get_parsing(img_rgb):
    h, w = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb).resize((512,512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    with torch.no_grad():
        out = bisenet(tensor)
        out = out[0] if isinstance(out, tuple) else out
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    return cv2.resize(parsing.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

def get_custom_masks(parsing, brow_dilation=3, mouth_dilation=3):
    brow_mask = np.isin(parsing, [2,3]).astype(np.float32)
    mouth_mask = np.isin(parsing, [11,12,13]).astype(np.float32)
    if brow_dilation > 0:
        brow_mask = cv2.dilate(brow_mask, np.ones((brow_dilation, brow_dilation), np.uint8), iterations=1)
    if mouth_dilation > 0:
        mouth_mask = cv2.dilate(mouth_mask, np.ones((mouth_dilation, mouth_dilation), np.uint8), iterations=1)
    emo_mask = np.clip(brow_mask + mouth_mask, 0, 1)
    id_mask = np.isin(parsing, [4,5,10]).astype(np.float32)
    id_mask = cv2.dilate(id_mask, np.ones((3,3), np.uint8), iterations=1)
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

def apply_ours_protection(img_rgb):
    """固定最终方法：brow=3, mouth=11, blur=17, mosaic=14"""
    parsing = get_parsing(img_rgb)
    emo_mask, id_mask, nc_mask = get_custom_masks(parsing, brow_dilation=3, mouth_dilation=11)
    proc_emo = keep_original
    proc_id = blur_processor(17)
    proc_nc = mosaic_processor(14)
    result = emo_mask[...,None]*proc_emo(img_rgb) + id_mask[...,None]*proc_id(img_rgb) + nc_mask[...,None]*proc_nc(img_rgb)
    return result.astype(np.uint8)

def apply_strong_mosaic(img_rgb):
    return mosaic_processor(16)(img_rgb)

def apply_gaussian_blur(img_rgb):
    return blur_processor(21)(img_rgb)

# ==================== 情感模型训练与评估 ====================
class EmotionDataset(Dataset):
    def __init__(self, samples, transform=None):
        # samples 可能是三元组 (img, label, subject) 或二元组，统一取前两个
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        img, label = item[0], item[1]
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def train_emotion_model(train_samples, val_samples, class_weights=None, epochs=40):
    train_set = EmotionDataset(train_samples, train_transform)
    val_set = EmotionDataset(val_samples, val_transform)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    
    model = timm.create_model('resnet18', pretrained=True, num_classes=6).to(DEVICE)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_macro_f1 = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(macro_f1)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    return model

def evaluate_emotion_model(model, test_samples):
    model.eval()
    all_preds, all_labels = [], []
    test_set = EmotionDataset(test_samples, val_transform)
    loader = DataLoader(test_set, batch_size=8, shuffle=False)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=EMOTIONS, output_dict=True, zero_division=0)
    per_f1 = {e: report[e]['f1-score'] for e in EMOTIONS}
    return acc, macro_f1, per_f1['sadness'], per_f1['fear'], per_f1['disgust']

# ==================== 身份评估 ====================
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
def get_embedding(img_rgb):
    img = cv2.resize(img_rgb, (160,160))
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)/255.
    tensor = (tensor-0.5)/0.5
    with torch.no_grad():
        return facenet(tensor).cpu().numpy().flatten()

def evaluate_identity(test_samples, gallery_embs):
    """test_samples: 列表，每项 (img_rgb, label_idx, subject)"""
    probe_embs, subjects = [], []
    for img_rgb, _, subj in test_samples:
        probe_embs.append(get_embedding(img_rgb))
        subjects.append(subj)
    genuine, impostor, ranks = [], [], []
    for emb, subj in zip(probe_embs, subjects):
        if subj in gallery_embs:
            genuine.append(np.dot(gallery_embs[subj], emb))
        for other_subj, other_emb in gallery_embs.items():
            if other_subj != subj:
                impostor.append(np.dot(other_emb, emb))
        sims = {s: np.dot(gallery_embs[s], emb) for s in gallery_embs}
        sorted_subjs = sorted(sims, key=sims.get, reverse=True)
        rank = sorted_subjs.index(subj)+1 if subj in sorted_subjs else -1
        ranks.append(rank)
    if not genuine or not impostor:
        return 0,0,0
    y_true = [1]*len(genuine) + [0]*len(impostor)
    y_score = genuine + impostor
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eer = fpr[np.nanargmin(np.abs(fpr - (1-tpr)))]
    top1 = sum(r==1 for r in ranks)/len(ranks)
    return top1, auc, eer

# ==================== 数据准备 ====================
df = pd.read_csv(META_CSV)
df_peak = df[~df['emotion_code'].isna()].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

all_subjects = sorted(df_peak['subject'].unique())
np.random.seed(SEED)
np.random.shuffle(all_subjects)

# ==================== 5-fold 循环 ====================
results = defaultdict(list)

for fold in range(FOLDS):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{FOLDS}")
    print('='*50)
    
    test_start = int(fold * len(all_subjects) / FOLDS)
    test_end = int((fold + 1) * len(all_subjects) / FOLDS)
    test_subjects = set(all_subjects[test_start:test_end])
    remaining = [s for s in all_subjects if s not in test_subjects]
    np.random.shuffle(remaining)
    val_size = max(1, int(0.2 * len(remaining)))
    val_subjects = set(remaining[:val_size])
    train_subjects = set(remaining[val_size:])
    
    train_df = df_peak[df_peak['subject'].isin(train_subjects)]
    val_df = df_peak[df_peak['subject'].isin(val_subjects)]
    test_df = df_peak[df_peak['subject'].isin(test_subjects)]
    
    def load_samples(subset_df, apply_protect=None, return_subject=False):
        samples = []
        for _, row in subset_df.iterrows():
            path = row['image_path']
            img = cv2.imread(path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = code_to_idx[int(row['emotion_code'])]
            subj = row['subject']
            if apply_protect is not None:
                img_rgb = apply_protect(img_rgb)
            if return_subject:
                samples.append((img_rgb, label, subj))
            else:
                samples.append((img_rgb, label))
        return samples
    
    # 训练/验证样本（二元组）
    train_orig = load_samples(train_df)
    val_orig = load_samples(val_df)
    train_ours = load_samples(train_df, apply_ours_protection)
    val_ours = load_samples(val_df, apply_ours_protection)
    
    # 测试样本：情感评估用二元组，身份评估用三元组
    test_orig_eval = load_samples(test_df)                 # 二元组
    test_ours_eval = load_samples(test_df, apply_ours_protection)
    test_mosaic_eval = load_samples(test_df, apply_strong_mosaic)
    test_blur_eval = load_samples(test_df, apply_gaussian_blur)
    
    test_orig_id = load_samples(test_df, return_subject=True)       # 三元组
    test_ours_id = load_samples(test_df, apply_ours_protection, return_subject=True)
    test_mosaic_id = load_samples(test_df, apply_strong_mosaic, return_subject=True)
    test_blur_id = load_samples(test_df, apply_gaussian_blur, return_subject=True)
    
    # Gallery
    df_neutral_test = df[(df['subset']=='test') & (df['emotion'].str.lower()=='neutral') & (df['subject'].isin(test_subjects))]
    gallery_embs = {}
    for _, row in df_neutral_test.iterrows():
        img = cv2.imread(row['image_path'])
        if img is None: continue
        gallery_embs[row['subject']] = get_embedding(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 类别权重
    labels = [s[1] for s in train_orig]
    counts = Counter(labels)
    class_counts = np.array([counts.get(i, 1) for i in range(6)])
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * 6
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    # 训练
    print("训练 Clean 模型...")
    model_clean = train_emotion_model(train_orig, val_orig, class_weights)
    print("训练 Mixed 模型...")
    train_mixed = train_orig + train_ours
    val_mixed = val_orig + val_ours
    model_mixed = train_emotion_model(train_mixed, val_mixed, class_weights)
    
    # 评估
    eval_configs = [
        ('Original', test_orig_eval, test_orig_id, model_clean, 'Clean'),
        ('Original', test_orig_eval, test_orig_id, model_mixed, 'Mixed'),
        ('Ours', test_ours_eval, test_ours_id, model_clean, 'Clean'),
        ('Ours', test_ours_eval, test_ours_id, model_mixed, 'Mixed'),
        ('Strong Mosaic', test_mosaic_eval, test_mosaic_id, model_mixed, 'Mixed'),
        ('Gaussian Blur', test_blur_eval, test_blur_id, model_mixed, 'Mixed'),
    ]
    
    for test_name, test_eval, test_id, model, model_tag in eval_configs:
        acc, macro_f1, f1_sad, f1_fear, f1_dis = evaluate_emotion_model(model, test_eval)
        top1, auc, eer = evaluate_identity(test_id, gallery_embs)
        key = f"{test_name}_{model_tag}"
        results[key].append({
            'Acc': acc, 'Macro-F1': macro_f1,
            'F1_sadness': f1_sad, 'F1_fear': f1_fear, 'F1_disgust': f1_dis,
            'Top-1': top1, 'AUC': auc, 'EER': eer
        })
        print(f"  {test_name} ({model_tag}): Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, SadF1={f1_sad:.4f}, Top1={top1:.4f}")

# ==================== 汇总 ====================
print("\n" + "="*80)
print("5-fold 最终结果")
print("="*80)
table_rows = []
for key, metrics in results.items():
    arr = {k: np.array([m[k] for m in metrics]) for k in metrics[0]}
    mean_std = {k: f"{arr[k].mean():.4f}±{arr[k].std():.4f}" for k in arr}
    table_rows.append({
        'Method': key,
        'Acc': mean_std['Acc'],
        'Macro-F1': mean_std['Macro-F1'],
        'F1_sadness': mean_std['F1_sadness'],
        'F1_fear': mean_std['F1_fear'],
        'F1_disgust': mean_std['F1_disgust'],
        'Top-1': mean_std['Top-1'],
        'AUC': mean_std['AUC'],
        'EER': mean_std['EER']
    })

df_final = pd.DataFrame(table_rows)
print(df_final.to_string(index=False))
df_final.to_csv("5fold_final_results.csv", index=False)
print("\n结果已保存至 5fold_final_results.csv")