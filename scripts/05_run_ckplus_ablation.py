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
import insightface
from insightface.app import FaceAnalysis

# ========== 缓存路径 ==========
os.environ['HF_HOME'] = r'F:\python\cache\huggingface'
os.environ['TORCH_HOME'] = r'F:\python\cache\torch'
INSIGHTFACE_ROOT = r'F:\python\models\insightface'
os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)

sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

# ==================== 全局配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
FOLDS = 5

PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_CSV = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')
BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"

code_to_idx = {1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
EMOTIONS = ['anger','disgust','fear','happiness','sadness','surprise']

# ==================== 加载 BiSeNet ====================
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
bisenet.load_state_dict(torch.load(BISENET_WEIGHT, map_location=DEVICE), strict=False)
bisenet.eval().to(DEVICE)

# ==================== 加载 ArcFace ====================
print("加载 ArcFace...")
arcface_app = FaceAnalysis(name='antelopev2', root=INSIGHTFACE_ROOT,
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
arcface_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(224,224))

def get_arcface_embedding(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    faces = arcface_app.get(img_bgr)
    if faces:
        return faces[0].normed_embedding
    return np.zeros(512, dtype=np.float32)

# ==================== 通用组件 ====================
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

# 标准语义掩膜
def semantic_masks(parsing):
    brow = np.isin(parsing, [2,3]).astype(np.float32)
    mouth = np.isin(parsing, [11,12,13]).astype(np.float32)
    brow = cv2.dilate(brow, np.ones((3,3),np.uint8), iterations=1)
    mouth = cv2.dilate(mouth, np.ones((11,11),np.uint8), iterations=1)
    emo_mask = np.clip(brow + mouth, 0, 1)
    id_mask = np.isin(parsing, [4,5,10]).astype(np.float32)
    id_mask = cv2.dilate(id_mask, np.ones((3,3),np.uint8), iterations=1)
    nc_mask = np.clip(1.0 - emo_mask - id_mask, 0, 1)
    return emo_mask, id_mask, nc_mask

# 随机掩膜（同等面积，非语义）
def random_masks(parsing, target_area=0.12):
    h,w = parsing.shape
    mask = np.zeros((h,w), dtype=np.float32)
    # 选取随机中心，生成高斯权重，取阈值达到目标面积
    # 简单方法：随机一个矩形区域，使其面积接近target_area
    for _ in range(100):
        cx, cy = np.random.randint(w), np.random.randint(h)
        rw, rh = int(np.sqrt(target_area)*w), int(np.sqrt(target_area)*h)
        x1, y1 = max(0, cx-rw//2), max(0, cy-rh//2)
        x2, y2 = min(w, cx+rw//2), min(h, cy+rh//2)
        if (x2-x1)*(y2-y1) > target_area*w*h*0.8:
            mask[y1:y2, x1:x2] = 1.0
            break
    # 确保有区域
    if mask.sum() < 10:
        mask[100:200, 100:200] = 1.0
    # 将随机区域作为“表情区”，其余作为眼鼻区和非核心区拆分
    # 为简化，我们直接生成两个随机mask，一个代表表情区（保留），一个代表眼鼻区（模糊），非核心马赛克
    emo_mask = mask
    # 随机眼鼻区：另外随机一块区域，占比较小
    mask2 = np.zeros_like(mask)
    for _ in range(100):
        cx, cy = np.random.randint(w), np.random.randint(h)
        rw2, rh2 = int(0.05*w), int(0.05*h)
        x1,y1 = max(0,cx-rw2//2), max(0,cy-rh2//2)
        x2,y2 = min(w,cx+rw2//2), min(h,cy+rh2//2)
        if (x2-x1)*(y2-y1) > 0.02*w*h:
            mask2[y1:y2, x1:x2] = 1.0
            break
    id_mask = mask2
    nc_mask = np.clip(1.0 - emo_mask - id_mask, 0, 1)
    return emo_mask, id_mask, nc_mask

# 保护函数生成器
def make_protector(emo_fn, id_fn, nc_fn, use_semantic=True):
    def protector(img_rgb):
        parsing = get_parsing(img_rgb)
        if use_semantic:
            emo_mask, id_mask, nc_mask = semantic_masks(parsing)
        else:
            emo_mask, id_mask, nc_mask = random_masks(parsing)
        img_emo = emo_fn(img_rgb)
        img_id = id_fn(img_rgb)
        img_nc = nc_fn(img_rgb)
        return (emo_mask[...,None]*img_emo + id_mask[...,None]*img_id + nc_mask[...,None]*img_nc).astype(np.uint8)
    return protector

# 定义消融方法及其保护函数
# 注意：以下均使用语义掩膜，除非标注Random
ablation_methods = {
    'Original': keep_original,  # 特殊：直接返回原图
    'Ours_full': make_protector(keep_original, blur_processor(17), mosaic_processor(14), use_semantic=True),
    'No_identity_blur': make_protector(keep_original, keep_original, mosaic_processor(14), use_semantic=True),
    'No_noncore_mosaic': make_protector(keep_original, blur_processor(17), keep_original, use_semantic=True),
    'Emotion_only_keep': make_protector(keep_original, mosaic_processor(14), mosaic_processor(14), use_semantic=True), # 眼鼻也马赛克
    'Random_mask_Ours': make_protector(keep_original, blur_processor(17), mosaic_processor(14), use_semantic=False),
}

# ==================== 情感模型训练与评估 ====================
class EmotionDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, label = self.samples[idx][:2]
        if self.transform: img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((256,256)), transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(5),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224,224)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def train_emotion_model(train_samples, val_samples, class_weights=None, epochs=40):
    train_set = EmotionDataset(train_samples, train_transform)
    val_set = EmotionDataset(val_samples, val_transform)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    model = timm.create_model('resnet18', pretrained=True, num_classes=6).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) if class_weights is not None else nn.CrossEntropyLoss()
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
def get_facenet_embedding(img_rgb):
    img = cv2.resize(img_rgb, (160,160))
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)/255.
    tensor = (tensor-0.5)/0.5
    with torch.no_grad(): return facenet(tensor).cpu().numpy().flatten()

def evaluate_identity(test_samples, gallery_embs, model_type='facenet'):
    valid_samples = [(img, subj) for img, _, subj in test_samples if subj in gallery_embs]
    if not valid_samples: return 0,0,0
    emb_fn = get_arcface_embedding if model_type == 'arcface' else get_facenet_embedding
    probe_embs, subjects = [], []
    for img_rgb, subj in valid_samples:
        probe_embs.append(emb_fn(img_rgb))
        subjects.append(subj)
    genuine, impostor, ranks = [], [], []
    for emb, subj in zip(probe_embs, subjects):
        genuine.append(np.dot(gallery_embs[subj], emb))
        for other_subj, other_emb in gallery_embs.items():
            if other_subj != subj: impostor.append(np.dot(other_emb, emb))
        sims = {s: np.dot(gallery_embs[s], emb) for s in gallery_embs}
        sorted_subjs = sorted(sims, key=sims.get, reverse=True)
        rank = sorted_subjs.index(subj) + 1
        ranks.append(rank)
    if not genuine or not impostor: return 0,0,0
    y_true = [1]*len(genuine) + [0]*len(impostor)
    y_score = genuine + impostor
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eer = fpr[np.nanargmin(np.abs(fpr - (1-tpr)))]
    top1 = sum(r==1 for r in ranks) / len(ranks)
    return top1, auc, eer

# ==================== 数据准备 ====================
df = pd.read_csv(META_CSV)
df_peak = df[~df['emotion_code'].isna()].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)
all_subjects = sorted(df_peak['subject'].unique())
np.random.seed(SEED)
np.random.shuffle(all_subjects)

# ==================== 5折循环 ====================
results = defaultdict(list)

for fold in range(FOLDS):
    print(f"\n{'='*50}\nFold {fold+1}/{FOLDS}\n{'='*50}")
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

    def load_samples(subset_df, apply_fn=None, return_subject=False):
        samples = []
        for _, row in subset_df.iterrows():
            path = row['image_path']
            img = cv2.imread(path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = code_to_idx[int(row['emotion_code'])]
            subj = row['subject']
            if apply_fn is not None: img_rgb = apply_fn(img_rgb)
            if return_subject: samples.append((img_rgb, label, subj))
            else: samples.append((img_rgb, label))
        return samples

    train_orig = load_samples(train_df)
    val_orig = load_samples(val_df)

    # Gallery
    df_neutral_test = df[(df['emotion'].str.lower()=='neutral') & (df['subject'].isin(test_subjects))].copy()
    gallery_facenet, gallery_arcface = {}, {}
    for subj, group in df_neutral_test.groupby('subject'):
        row = group.iloc[0]
        img = cv2.imread(row['image_path'])
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gallery_facenet[subj] = get_facenet_embedding(img_rgb)
        gallery_arcface[subj] = get_arcface_embedding(img_rgb)

    labels = [s[1] for s in train_orig]
    counts = Counter(labels)
    class_counts = np.array([counts.get(i,1) for i in range(6)])
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * 6
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # 为每个消融方法训练专用模型（除Original外，其余均用mixed策略）
    for method_name, protect_fn in ablation_methods.items():
        if method_name == 'Original':
            model = train_emotion_model(train_orig, val_orig, class_weights)
        else:
            train_prot = load_samples(train_df, protect_fn)
            val_prot = load_samples(val_df, protect_fn)
            model = train_emotion_model(train_orig + train_prot, val_orig + val_prot, class_weights)

        # 评估自己的保护图像
        test_fn = protect_fn if method_name != 'Original' else keep_original
        test_eval = load_samples(test_df, test_fn)
        test_id = load_samples(test_df, test_fn, return_subject=True)

        acc, macro_f1, f1_sad, f1_fear, f1_dis = evaluate_emotion_model(model, test_eval)
        top1_fn, _, _ = evaluate_identity(test_id, gallery_facenet, 'facenet')
        top1_af, _, _ = evaluate_identity(test_id, gallery_arcface, 'arcface')

        results[method_name].append({
            'Acc': acc, 'Macro-F1': macro_f1,
            'F1_sadness': f1_sad, 'F1_fear': f1_fear, 'F1_disgust': f1_dis,
            'Top-1_FN': top1_fn, 'Top-1_AF': top1_af
        })
        print(f"  {method_name}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, SadF1={f1_sad:.4f}, FN_Top1={top1_fn:.4f}, AF_Top1={top1_af:.4f}")

# 汇总
print("\n" + "="*80)
print("消融实验结果")
print("="*80)
table_rows = []
for method, metrics in results.items():
    arr = {k: np.array([m[k] for m in metrics]) for k in metrics[0]}
    mean_std = {k: f"{arr[k].mean():.4f}±{arr[k].std():.4f}" for k in arr}
    table_rows.append({
        'Method': method,
        'Acc': mean_std['Acc'],
        'Macro-F1': mean_std['Macro-F1'],
        'F1_sadness': mean_std['F1_sadness'],
        'Top-1_FaceNet': mean_std['Top-1_FN'],
        'Top-1_ArcFace': mean_std['Top-1_AF']
    })
df_final = pd.DataFrame(table_rows)
print(df_final.to_string(index=False))
df_final.to_csv("5fold_ablation_results.csv", index=False)
print("\n结果已保存至 5fold_ablation_results.csv")