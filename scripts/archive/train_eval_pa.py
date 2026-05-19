import os, sys, cv2, torch, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from torchvision.transforms import functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
from facenet_pytorch import InceptionResnetV1

sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ==================== 全局配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_CSV = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')
PA_DATA_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_pa"

BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"
CLEAN_MODEL_PATH = "best_emotion_ckplus.pth"
PA_MODEL_DIR = "./pa_models"
os.makedirs(PA_MODEL_DIR, exist_ok=True)

code_to_idx = {1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
idx2emo = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise'}
EMOTIONS = list(idx2emo.values())

PROTECTED_VERSIONS = {
    'original': None,
    'base':    {'brow':3, 'mouth':11, 'blur':17, 'mosaic':14},
    'n_blur17':{'brow':3, 'mouth':11, 'blur':17, 'mosaic':14},  # 实际与 base 相同，为了保持命名保留
    'm_nc15':  {'brow':3, 'mouth':11, 'blur':17, 'mosaic':15},
    'strong_mosaic': 'mosaic16'
}

# ==================== 加载 BiSeNet ====================
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
bisenet.load_state_dict(torch.load(BISENET_WEIGHT, map_location=DEVICE), strict=False)
bisenet.eval().to(DEVICE)

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

def apply_protection(img_rgb, version):
    if version == 'original':
        return img_rgb.copy()
    if version == 'strong_mosaic':
        return mosaic_processor(16)(img_rgb)
    params = PROTECTED_VERSIONS[version]
    parsing = get_parsing(img_rgb)
    emo_mask, id_mask, nc_mask = get_custom_masks(parsing,
                                                   brow_dilation=params['brow'],
                                                   mouth_dilation=params['mouth'])
    proc_emo = keep_original
    proc_id = blur_processor(params['blur'])
    proc_nc = mosaic_processor(params['mosaic'])
    result = emo_mask[...,None]*proc_emo(img_rgb) + id_mask[...,None]*proc_id(img_rgb) + nc_mask[...,None]*proc_nc(img_rgb)
    return result.astype(np.uint8)

# ==================== 生成数据 ====================
def generate_pa_data():
    if os.path.exists(PA_DATA_DIR):
        print("Protected-aware 数据已存在，跳过生成。")
        return
    print("生成 Protected-aware 数据...")
    df = pd.read_csv(META_CSV)
    df_peak = df[~df['emotion_code'].isna()].copy()
    df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)
    subsets = ['train', 'val', 'test']
    versions = ['original', 'base', 'n_blur17', 'm_nc15']
    new_rows = []
    for subset in subsets:
        subset_df = df_peak[df_peak['subset'] == subset]
        for ver in versions:
            out_dir = os.path.join(PA_DATA_DIR, subset, ver)
            os.makedirs(out_dir, exist_ok=True)
            for _, row in tqdm(subset_df.iterrows(), desc=f"{subset}/{ver}", total=len(subset_df)):
                img_path = row['image_path']
                img = cv2.imread(img_path)
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                protected = apply_protection(img_rgb, ver)
                out_name = os.path.basename(img_path)
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, cv2.cvtColor(protected, cv2.COLOR_RGB2BGR))
                new_row = row.to_dict()
                new_row['image_path'] = out_path
                new_row['version'] = ver
                new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows)
    meta_dir = os.path.join(PA_DATA_DIR, 'metadata')
    os.makedirs(meta_dir, exist_ok=True)
    new_df.to_csv(os.path.join(meta_dir, 'pa_samples.csv'), index=False)
    print("Protected-aware 数据生成完毕。")

class EmotionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        label = code_to_idx[int(row['emotion_code'])]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img_rgb)
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

def train_one_model(train_df, val_df, model_name, class_weights=None):
    train_set = EmotionDataset(train_df, train_transform)
    val_set = EmotionDataset(val_df, val_transform)
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
    for epoch in range(40):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
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
            torch.save(model.state_dict(), os.path.join(PA_MODEL_DIR, f"{model_name}.pth"))
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}")
    print(f"{model_name} 最佳 Macro-F1: {best_macro_f1:.4f}")
    model.load_state_dict(torch.load(os.path.join(PA_MODEL_DIR, f"{model_name}.pth")))
    return model

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
def get_embedding(img_rgb):
    img = cv2.resize(img_rgb, (160,160))
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)/255.
    tensor = (tensor-0.5)/0.5
    with torch.no_grad():
        return facenet(tensor).cpu().numpy().flatten()

def evaluate_emotion_dynamic(model, test_info, version):
    model.eval()
    all_preds, all_labels = [], []
    for img_rgb, label, _ in test_info:
        if version != 'original':
            img_rgb = apply_protection(img_rgb, version)
        tensor = val_transform(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(tensor).argmax(1).item()
        all_preds.append(pred)
        all_labels.append(label)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=EMOTIONS, output_dict=True, zero_division=0)
    per_f1 = {e: report[e]['f1-score'] for e in EMOTIONS}
    return acc, macro_f1, per_f1

def evaluate_identity_dynamic(test_info, version, gallery_embs):
    probe_embs, subjects = [], []
    for img_rgb, _, subj in test_info:
        if version != 'original':
            img_rgb = apply_protection(img_rgb, version)
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

# ==================== 主流程 ====================
if __name__ == "__main__":
    generate_pa_data()
    pa_csv = os.path.join(PA_DATA_DIR, 'metadata', 'pa_samples.csv')
    df_pa = pd.read_csv(pa_csv)
    df_pa['label_idx'] = df_pa['emotion_code'].map(code_to_idx)

    df_orig = pd.read_csv(META_CSV)
    df_neutral = df_orig[(df_orig['subset']=='test') & (df_orig['emotion'].str.lower()=='neutral')]
    gallery_embs = {}
    print("构建 Gallery 嵌入...")
    for _, row in tqdm(df_neutral.iterrows()):
        img = cv2.imread(row['image_path'])
        if img is None: continue
        gallery_embs[row['subject']] = get_embedding(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    df_train_orig = df_pa[(df_pa['subset']=='train') & (df_pa['version']=='original')]
    df_val_orig   = df_pa[(df_pa['subset']=='val') & (df_pa['version']=='original')]
    df_train_base = df_pa[(df_pa['subset']=='train') & (df_pa['version']=='base')]
    df_val_base   = df_pa[(df_pa['subset']=='val') & (df_pa['version']=='base')]

    labels = df_train_orig['label_idx'].values
    counts = Counter(labels)
    class_counts = np.array([counts[i] for i in range(6)])
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * 6
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print("\n=== 加载 Clean-trained baseline ===")
    model_clean = timm.create_model('resnet18', pretrained=False, num_classes=6)
    model_clean.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=DEVICE))
    model_clean.to(DEVICE).eval()

    print("\n=== 训练 Protected-only model ===")
    model_prot_only = train_one_model(df_train_base, df_val_base, "protected_only", class_weights)

    print("\n=== 训练 Mixed model ===")
    df_train_mixed = pd.concat([df_train_orig, df_train_base], ignore_index=True)
    df_val_mixed = pd.concat([df_val_orig, df_val_base], ignore_index=True)
    model_mixed = train_one_model(df_train_mixed, df_val_mixed, "mixed", class_weights)

    print("\n=== 全面评估 ===")
    df_test_peak_orig = df_orig[(df_orig['subset']=='test') & (~df_orig['emotion_code'].isna())].copy()
    df_test_peak_orig['emotion_code'] = df_test_peak_orig['emotion_code'].astype(int)
    test_info = []
    for _, row in df_test_peak_orig.iterrows():
        img = cv2.imread(row['image_path'])
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_info.append((img_rgb, code_to_idx[int(row['emotion_code'])], row['subject']))

    versions = ['original', 'base', 'n_blur17', 'm_nc15', 'strong_mosaic']
    results = []
    for ver in versions:
        top1, auc, eer = evaluate_identity_dynamic(test_info, ver, gallery_embs)
        for model_name, model in [('Clean', model_clean), ('Protected-only', model_prot_only), ('Mixed', model_mixed)]:
            acc, macro_f1, per_f1 = evaluate_emotion_dynamic(model, test_info, ver)
            results.append({
                'Model': model_name, 'Version': ver,
                'Acc': acc, 'Macro-F1': macro_f1,
                'F1_sadness': per_f1['sadness'], 'F1_fear': per_f1['fear'], 'F1_disgust': per_f1['disgust'],
                'Top-1': top1, 'AUC': auc, 'EER': eer
            })
            print(f"{model_name} on {ver}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, SadF1={per_f1['sadness']:.4f}, Top1={top1:.4f}")

    df_res = pd.DataFrame(results)
    print("\n===== 最终结果 =====")
    print(df_res.to_string(index=False))
    df_res.to_csv("pa_training_results.csv", index=False)