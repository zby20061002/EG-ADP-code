import os
import numpy as np
from collections import defaultdict

# ==================== 路径设置 ====================
BASE = r"F:\python\2024218729zby_ML\EG-ADP\data\CK+"
IMAGES_DIR = os.path.join(BASE, "cohn-kanade-images")
EMOTION_DIR = os.path.join(BASE, "Emotion")

# ==================== 表情标签映射 ====================
# CK+ 标签: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
EMO_MAP = {
    0: 'neutral',
    1: 'anger',
    2: 'contempt',
    3: 'disgust',
    4: 'fear',
    5: 'happiness',
    6: 'sadness',
    7: 'surprise'
}

# 我们最终使用的6类（排除 neutral 和 contempt）
VALID_EMOTIONS = {'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'}

# ==================== 扫描数据 ====================
print("=" * 60)
print("CK+ 数据集探测报告")
print("=" * 60)

# 1. 扫描图像目录
print("\n[1] 扫描图像目录...")
subjects = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]
print(f"  发现 {len(subjects)} 个被试 (Subject)")

total_sequences = 0
subject_sequence_count = defaultdict(int)

for subject in subjects:
    subject_dir = os.path.join(IMAGES_DIR, subject)
    sequences = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
    subject_sequence_count[subject] = len(sequences)
    total_sequences += len(sequences)

print(f"  总序列数: {total_sequences}")
print(f"  每个被试平均序列数: {total_sequences / len(subjects):.1f}")
print(f"  序列数范围: {min(subject_sequence_count.values())} ~ {max(subject_sequence_count.values())}")

# 2. 扫描表情标签目录
print("\n[2] 扫描表情标签目录...")
emotion_subjects = [d for d in os.listdir(EMOTION_DIR) if os.path.isdir(os.path.join(EMOTION_DIR, d))]
print(f"  有表情标签的被试数: {len(emotion_subjects)}")

labeled_sequences = []
emotion_distribution = defaultdict(int)
sequence_details = {}

for subject in emotion_subjects:
    subject_dir = os.path.join(EMOTION_DIR, subject)
    for seq in os.listdir(subject_dir):
        seq_dir = os.path.join(subject_dir, seq)
        if not os.path.isdir(seq_dir):
            continue
        # 查找 _emotion.txt 文件
        for fname in os.listdir(seq_dir):
            if fname.endswith('_emotion.txt'):
                fpath = os.path.join(seq_dir, fname)
                with open(fpath, 'r') as f:
                    content = f.read().strip()
                    try:
                        emo_code = int(float(content))
                        emo_name = EMO_MAP.get(emo_code, 'unknown')
                        sequence_details[(subject, seq)] = {
                            'code': emo_code,
                            'name': emo_name,
                            'valid': emo_name in VALID_EMOTIONS
                        }
                        emotion_distribution[emo_name] += 1
                        labeled_sequences.append((subject, seq, emo_name))
                    except:
                        pass
                break  # 每个序列只有一个标签文件

print(f"  有表情标签的序列数: {len(labeled_sequences)}")
print(f"\n  表情标签分布:")
print(f"  {'表情':<15} {'数量':<8} {'是否使用'}")
print(f"  {'-'*35}")
for emo_name in ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']:
    count = emotion_distribution.get(emo_name, 0)
    status = "✅ 使用" if emo_name in VALID_EMOTIONS else "❌ 排除"
    if count > 0:
        print(f"  {emo_name:<15} {count:<8} {status}")

valid_count = sum(emotion_distribution[e] for e in VALID_EMOTIONS)
print(f"\n  可用序列总数 (6类): {valid_count}")

# 3. 检查图像与标签的对应关系
print("\n[3] 验证图像-标签对应关系...")
missing_images = []
peak_frame_info = {}

for subject, seq, emo_name in labeled_sequences:
    if emo_name not in VALID_EMOTIONS:
        continue
    
    seq_img_dir = os.path.join(IMAGES_DIR, subject, seq)
    if not os.path.exists(seq_img_dir):
        missing_images.append((subject, seq))
        continue
    
    frames = sorted([f for f in os.listdir(seq_img_dir) if f.endswith('.png')])
    if not frames:
        missing_images.append((subject, seq))
        continue
    
    # 记录峰值帧（最后一帧）
    peak_frame_info[(subject, seq)] = {
        'total_frames': len(frames),
        'peak_frame': frames[-1],
        'emotion': emo_name
    }

print(f"  可用的峰值帧数: {len(peak_frame_info)}")
if missing_images:
    print(f"  ⚠️ 有 {len(missing_images)} 个序列缺少图像文件")
    for s, seq in missing_images[:5]:
        print(f"    - Subject {s}, Sequence {seq}")

# 4. 输出数据集划分建议
print("\n[4] 数据集划分建议...")
all_subjects_with_labels = list(set(s[0] for s in peak_frame_info.keys()))
print(f"  有可用标签的被试数: {len(all_subjects_with_labels)}")

# 随机划分（固定种子以确保可复现）
np.random.seed(42)
shuffled = np.random.permutation(all_subjects_with_labels)
n_train = int(len(shuffled) * 0.7)
n_val = int(len(shuffled) * 0.1)

train_subjects = set(shuffled[:n_train])
val_subjects = set(shuffled[n_train:n_train + n_val])
test_subjects = set(shuffled[n_train + n_val:])

print(f"  建议划分:")
print(f"    训练集: {len(train_subjects)} 人")
print(f"    验证集: {len(val_subjects)} 人")
print(f"    测试集: {len(test_subjects)} 人")

# 统计各划分中的样本数
train_count = sum(1 for (s, _), _ in peak_frame_info.items() if s in train_subjects)
val_count = sum(1 for (s, _), _ in peak_frame_info.items() if s in val_subjects)
test_count = sum(1 for (s, _), _ in peak_frame_info.items() if s in test_subjects)
print(f"    训练集样本: {train_count}")
print(f"    验证集样本: {val_count}")
print(f"    测试集样本: {test_count}")

# 5. 样本图像尺寸检测
print("\n[5] 检测图像尺寸...")
import cv2
sample_sizes = []
for (subject, seq), info in list(peak_frame_info.items())[:10]:
    img_path = os.path.join(IMAGES_DIR, subject, seq, info['peak_frame'])
    img = cv2.imread(img_path)
    if img is not None:
        sample_sizes.append(img.shape[:2])

if sample_sizes:
    avg_h = np.mean([s[0] for s in sample_sizes])
    avg_w = np.mean([s[1] for s in sample_sizes])
    print(f"  采样 {len(sample_sizes)} 张图像，平均尺寸: {avg_w:.0f} × {avg_h:.0f}")

print("\n" + "=" * 60)
print("探测完成！")
print("=" * 60)