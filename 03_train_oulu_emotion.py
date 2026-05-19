import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ------- 配置 -------
PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_oulu"
BATCH_SIZE = 32
LR_HEAD = 1e-3          # 分类头学习率
LR_FT = 1e-4            # 微调学习率
EPOCHS_HEAD = 5
EPOCHS_FT = 20
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# 标签映射
emo_str2idx = {
    'Anger': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happiness': 3,
    'Sadness': 4,
    'Surprise': 5
}
idx2emo = {v: k for k, v in emo_str2idx.items()}

torch.manual_seed(SEED)
np.random.seed(SEED)

# ------- 数据集 -------
class OuluDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for subj in os.listdir(root_dir):
            subj_dir = os.path.join(root_dir, subj)
            if not os.path.isdir(subj_dir):
                continue
            for fname in os.listdir(subj_dir):
                parts = fname.split('_')
                if len(parts) < 2:
                    continue
                emo = parts[0]
                if emo in emo_str2idx:
                    self.samples.append((os.path.join(subj_dir, fname), emo_str2idx[emo]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

# 训练数据增强
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据
train_set = OuluDataset(os.path.join(PROCESSED_DIR, 'train'), train_transform)
val_set = OuluDataset(os.path.join(PROCESSED_DIR, 'val'), val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"训练集样本数: {len(train_set)}")
print(f"验证集样本数: {len(val_set)}")
print("训练集类别分布:", Counter([label for _, label in train_set.samples]))
print("验证集类别分布:", Counter([label for _, label in val_set.samples]))

# ------- 模型 -------
model = timm.create_model('resnet18', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 早停相关
best_val_loss = float('inf')
best_val_acc = 0.0
patience = 5
early_stop_counter = 0

# 保存历史
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

def validate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc, all_preds, all_labels

# ===== 阶段一：只训练分类头 =====
print("\n===== 阶段一：训练分类头 =====")
for name, param in model.named_parameters():
    if "fc" not in name and "classifier" not in name:
        param.requires_grad = False

# 确保分类头可训练
for name, param in model.named_parameters():
    if "fc" in name or "classifier" in name:
        param.requires_grad = True

optimizer_head = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=1e-4)

for epoch in range(EPOCHS_HEAD):
    model.train()
    train_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Head Epoch {epoch+1}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer_head.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_head.step()
        train_loss += loss.item() * imgs.size(0)

    avg_train_loss = train_loss / len(train_set)
    val_loss, val_acc, _, _ = validate(model, val_loader)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Head Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ===== 阶段二：微调完整网络 =====
print("\n===== 阶段二：微调完整网络 =====")
# 解冻所有层
for param in model.parameters():
    param.requires_grad = True

optimizer_ft = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=3, factor=0.5)

best_val_acc_epoch = 0
best_val_loss_epoch = 0

for epoch in range(EPOCHS_FT):
    model.train()
    train_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"FT Epoch {epoch+1}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer_ft.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()
        train_loss += loss.item() * imgs.size(0)

    avg_train_loss = train_loss / len(train_set)
    val_loss, val_acc, preds, labels = validate(model, val_loader)
    scheduler.step(val_loss)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"FT Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer_ft.param_groups[0]['lr']:.2e}")

    # 保存最佳 val_loss 模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_emotion_model_oulu_loss.pth")
        best_val_loss_epoch = epoch + 1
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # 保存最佳 val_acc 模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_emotion_model_oulu_acc.pth")
        best_val_acc_epoch = epoch + 1

    # 早停
    if early_stop_counter >= patience:
        print(f"验证损失连续 {patience} 轮未下降，提前停止训练。")
        break

print(f"\n训练完成。最佳验证损失: {best_val_loss:.4f} (第 {best_val_loss_epoch} 轮)")
print(f"最佳验证准确率: {best_val_acc:.4f} (第 {best_val_acc_epoch} 轮)")

# ===== 最终评估 =====
# 载入最佳准确率模型做全面报告
model.load_state_dict(torch.load("best_emotion_model_oulu_acc.pth"))
_, _, final_preds, final_labels = validate(model, val_loader)

print("\n分类报告 (基于最佳准确率模型):")
print(classification_report(
    final_labels, final_preds,
    target_names=[idx2emo[i] for i in range(NUM_CLASSES)],
    digits=4
))

print("混淆矩阵:")
print(confusion_matrix(final_labels, final_preds))