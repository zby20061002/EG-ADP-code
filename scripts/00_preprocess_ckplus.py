import os, cv2, shutil, random, numpy as np, torch, pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ------- 配置 -------
BASE = r"F:\python\2024218729zby_ML\EG-ADP\data\CK+"
IMAGES_DIR = os.path.join(BASE, "cohn-kanade-images")
EMOTION_DIR = os.path.join(BASE, "Emotion")
OUTPUT_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 标签映射（6类）
EMO_MAP = {1: 'anger', 3: 'disgust', 4: 'fear', 5: 'happiness', 6: 'sadness', 7: 'surprise'}
VALID_EMOTIONS = set(EMO_MAP.values())

# 清空输出目录
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建子目录
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'metadata'), exist_ok=True)

# 人脸检测器 (post_process=False 直接输出 0-255 图像)
mtcnn = MTCNN(
    image_size=224,
    margin=40,
    keep_all=False,
    post_process=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# -------- 读取标签并收集可用序列 --------
labeled_seqs = []  # (subject, seq, emotion_code, emotion_name)
for subject in os.listdir(EMOTION_DIR):
    subj_emo_dir = os.path.join(EMOTION_DIR, subject)
    if not os.path.isdir(subj_emo_dir):
        continue
    for seq in os.listdir(subj_emo_dir):
        seq_dir = os.path.join(subj_emo_dir, seq)
        if not os.path.isdir(seq_dir):
            continue
        for fname in os.listdir(seq_dir):
            if fname.endswith('_emotion.txt'):
                with open(os.path.join(seq_dir, fname), 'r') as f:
                    code = int(float(f.read().strip()))
                    if code in EMO_MAP:
                        labeled_seqs.append((subject, seq, code, EMO_MAP[code]))
                break

print(f"共有 {len(labeled_seqs)} 个可用序列（6类表情）")

# 提取 neutral 和 peak 帧路径
data_entries = []  # (subject, seq, emotion_code, emotion_name, neutral_path, peak_path)
for subject, seq, code, emotion in labeled_seqs:
    seq_img_dir = os.path.join(IMAGES_DIR, subject, seq)
    if not os.path.exists(seq_img_dir):
        continue
    frames = sorted([f for f in os.listdir(seq_img_dir) if f.endswith('.png')])
    if len(frames) < 2:  # 至少要有 neutral 和 peak
        continue
    neutral_frame = frames[0]
    peak_frame = frames[-1]
    data_entries.append((subject, seq, code, emotion,
                         os.path.join(seq_img_dir, neutral_frame),
                         os.path.join(seq_img_dir, peak_frame)))

print(f"有效序列数（含neutral+peak）: {len(data_entries)}")

# -------- 划分身份 (subject-independent) --------
all_subjects = sorted(list(set([d[0] for d in data_entries])))
random.shuffle(all_subjects)
n_train = int(len(all_subjects) * 0.7)
n_val = int(len(all_subjects) * 0.1)
split = {
    'train': set(all_subjects[:n_train]),
    'val':   set(all_subjects[n_train:n_train+n_val]),
    'test':  set(all_subjects[n_train+n_val:])
}
print(f"训练: {len(split['train'])}人, 验证: {len(split['val'])}人, 测试: {len(split['test'])}人")

# -------- 处理并保存图像 --------
def save_face(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        face = mtcnn(img_rgb)  # post_process=False，输出 0-255 的 PIL Image 或 tensor
        if face is None:
            return False
        # face 可能是 PIL Image 或 tensor，统一转 numpy
        if isinstance(face, torch.Tensor):
            face = face.permute(1, 2, 0).cpu().numpy()
            # 如果 MTCNN 内部已经做了 0-255，直接转 uint8
            face = np.clip(face, 0, 255).astype(np.uint8)
        elif hasattr(face, 'convert'):  # PIL Image
            face = np.array(face)
            if face.ndim == 2:  # 灰度
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
            elif face.shape[2] == 3:
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        else:
            return False
        # 确保 BGR 存储
        if face.shape[2] == 3:
            cv2.imwrite(save_path, face)
        else:
            cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"失败: {img_path}, error={e}")
        return False

# 保存 neutral 和 peak，并记录 metadata
metadata_rows = []
success_count = 0

for subject, seq, code, emotion, neutral_path, peak_path in tqdm(data_entries, desc="预处理"):
    # 确定子集
    subset = 'train' if subject in split['train'] else ('val' if subject in split['val'] else 'test')
    
    # 保存 neutral
    neutral_save = os.path.join(OUTPUT_DIR, 'images', f"{subject}_{seq}_neutral.png")
    if save_face(neutral_path, neutral_save):
        metadata_rows.append([subject, seq, subset, 'neutral', neutral_save, None])
        success_count += 1
    
    # 保存 peak
    peak_save = os.path.join(OUTPUT_DIR, 'images', f"{subject}_{seq}_peak.png")
    if save_face(peak_path, peak_save):
        metadata_rows.append([subject, seq, subset, emotion, peak_save, code])
        success_count += 1

# 保存 metadata CSV
df_meta = pd.DataFrame(metadata_rows, columns=['subject', 'sequence', 'subset', 'emotion', 'image_path', 'emotion_code'])
df_meta.to_csv(os.path.join(OUTPUT_DIR, 'metadata', 'samples.csv'), index=False)

# 保存 fold 划分信息
fold_info = []
for subject in all_subjects:
    if subject in split['train']:
        fold_info.append([subject, 'train'])
    elif subject in split['val']:
        fold_info.append([subject, 'val'])
    else:
        fold_info.append([subject, 'test'])
df_folds = pd.DataFrame(fold_info, columns=['subject', 'subset'])
df_folds.to_csv(os.path.join(OUTPUT_DIR, 'metadata', 'subject_folds.csv'), index=False)

print(f"\n成功保存 {success_count} 张图像。")
print(f"请立即检查 {OUTPUT_DIR}/images/ 中的图像是否正常（颜色、裁切、关键部位完整）。")