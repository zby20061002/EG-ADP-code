import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# ------- 配置 -------
DATA_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
META_PATH = os.path.join(DATA_DIR, 'metadata', 'samples.csv')
BATCH_SIZE = 8
LR_HEAD = 1e-3
LR_FT = 1e-4
EPOCHS_HEAD = 10
EPOCHS_FT = 40
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# 原始CK+编码 → 0~5索引
code_to_idx = {1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
idx2emo = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise'}

torch.manual_seed(SEED)
np.random.seed(SEED)

# ------- 数据集 -------
class CKPlusDataset(Dataset):
    def __init__(self, df, transform=None, use_peak_only=True):
        self.transform = transform
        self.samples = []
        for _, row in df.iterrows():
            if use_peak_only and pd.isna(row['emotion_code']):
                continue
            path = row['image_path']
            code = int(row['emotion_code'])
            label = code_to_idx[code]          # 映射为0-5
            self.samples.append((path, label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

# 温和增强
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
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

# 读取元数据
df_all = pd.read_csv(META_PATH)
df_peak = df_all[~df_all['emotion_code'].isna()].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

# 按 subset 分割
df_train_val = df_peak[df_peak['subset'].isin(['train', 'val'])]
df_test = df_peak[df_peak['subset'] == 'test']

# 从 train/val 中构建 5-fold 交叉验证
subjects = df_train_val['subject'].unique()
np.random.shuffle(subjects)

k_folds = 5
fold_accs = []
fold_f1s = []

for fold in range(k_folds):
    # 划分
    test_start = int(fold * len(subjects) / k_folds)
    test_end = int((fold + 1) * len(subjects) / k_folds)
    val_subjects = set(subjects[test_start:test_end])
    train_subjects = set(subjects) - val_subjects
    
    df_train = df_train_val[df_train_val['subject'].isin(train_subjects)]
    df_val = df_train_val[df_train_val['subject'].isin(val_subjects)]
    
    print(f"\n===== Fold {fold+1} =====")
    print(f"训练集: {len(df_train)} 张, 验证集: {len(df_val)} 张")
    
    train_set = CKPlusDataset(df_train, train_transform, use_peak_only=True)
    val_set = CKPlusDataset(df_val, val_transform, use_peak_only=True)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # 模型
    model = timm.create_model('resnet18', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # 阶段一：冻结骨干
    for n, p in model.named_parameters():
        p.requires_grad = ('fc' in n or 'classifier' in n)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=1e-4)
    for ep in range(EPOCHS_HEAD):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
    
    # 阶段二：解冻全模型
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FT)
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    
    for ep in range(EPOCHS_FT):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        val_loss = val_loss / len(val_set)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_macro_f1
            best_epoch = ep + 1
        
        if (ep + 1) % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch {ep+1:2d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Macro-F1: {val_macro_f1:.4f}")
    
    fold_accs.append(best_val_acc)
    fold_f1s.append(best_val_f1)
    print(f"Fold {fold+1} 最佳: Acc={best_val_acc:.4f}, Macro-F1={best_val_f1:.4f} (epoch {best_epoch})")

print(f"\n{k_folds}-fold 交叉验证结果:")
print(f"平均准确率: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
print(f"平均 Macro-F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

# ===== 训练最终模型（使用全部 train+val 数据）并保存 =====
print("\n===== 训练最终模型（全数据） =====")

# 合并训练集和验证集
df_final_train = df_train_val.copy()

final_train_set = CKPlusDataset(df_final_train, train_transform, use_peak_only=True)
final_train_loader = DataLoader(final_train_set, batch_size=BATCH_SIZE, shuffle=True)

# 重新初始化模型
final_model = timm.create_model('resnet18', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()

# 阶段一：训练分类头
for n, p in final_model.named_parameters():
    p.requires_grad = ('fc' in n or 'classifier' in n)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, final_model.parameters()), lr=LR_HEAD, weight_decay=1e-4)
for ep in range(EPOCHS_HEAD):
    final_model.train()
    for imgs, labels in final_train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(final_model(imgs), labels)
        loss.backward()
        optimizer.step()

# 阶段二：解冻全模型
for p in final_model.parameters():
    p.requires_grad = True
optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR_FT, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FT)
for ep in range(EPOCHS_FT):
    final_model.train()
    for imgs, labels in final_train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(final_model(imgs), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

# 保存最终模型
torch.save(final_model.state_dict(), "best_emotion_ckplus.pth")
print("最终模型已保存为 best_emotion_ckplus.pth")