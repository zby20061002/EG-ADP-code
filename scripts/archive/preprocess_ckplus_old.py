import os, cv2, shutil, random, numpy as np, torch
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ------- 配置 -------
BASE = r"F:\python\2024218729zby_ML\EG-ADP\data\CK+"
IMAGES_DIR = os.path.join(BASE, "cohn-kanade-images")
EMOTION_DIR = os.path.join(BASE, "Emotion")
OUTPUT_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 标签映射（仅保留六类）
EMO_MAP = {1: 'anger', 3: 'disgust', 4: 'fear', 5: 'happiness', 6: 'sadness', 7: 'surprise'}
VALID_EMOTIONS = set(EMO_MAP.values())

# 创建输出目录结构
for subset in ['train', 'val', 'test']:
    for emo in VALID_EMOTIONS:
        os.makedirs(os.path.join(OUTPUT_DIR, subset, emo), exist_ok=True)

# 人脸检测器
mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

# -------- 读取标签并收集可用序列 --------
labeled_seqs = []  # (subject, seq, emotion_name)
for subject in os.listdir(EMOTION_DIR):
    subj_emo_dir = os.path.join(EMOTION_DIR, subject)
    if not os.path.isdir(subj_emo_dir):
        continue
    for seq in os.listdir(subj_emo_dir):
        seq_dir = os.path.join(subj_emo_dir, seq)
        if not os.path.isdir(seq_dir):
            continue
        # 找表情标签文件
        for fname in os.listdir(seq_dir):
            if fname.endswith('_emotion.txt'):
                with open(os.path.join(seq_dir, fname), 'r') as f:
                    code = int(float(f.read().strip()))
                    if code in EMO_MAP:
                        labeled_seqs.append((subject, seq, EMO_MAP[code]))
                break

print(f"共有 {len(labeled_seqs)} 个可用序列（6类表情）")

# 提取峰值帧路径
peak_data = []
for subject, seq, emotion in labeled_seqs:
    seq_img_dir = os.path.join(IMAGES_DIR, subject, seq)
    if not os.path.exists(seq_img_dir):
        continue
    frames = sorted([f for f in os.listdir(seq_img_dir) if f.endswith('.png')])
    if not frames:
        continue
    peak_frame = frames[-1]
    peak_data.append((subject, seq, emotion, os.path.join(seq_img_dir, peak_frame)))

print(f"可用峰值帧数: {len(peak_data)}")

# -------- 划分身份 --------
all_subjects = sorted(list(set([d[0] for d in peak_data])))
random.shuffle(all_subjects)
n_train = 74
n_val = 10
split = {
    'train': set(all_subjects[:n_train]),
    'val': set(all_subjects[n_train:n_train+n_val]),
    'test': set(all_subjects[n_train+n_val:])
}
print(f"训练: {len(split['train'])}人, 验证: {len(split['val'])}人, 测试: {len(split['test'])}人")

# -------- 处理并保存 --------
def process_and_save(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        face = mtcnn(img_rgb, save_path=None)
        if face is not None:
            face = face.permute(1, 2, 0).cpu().numpy()
            face = (face * 255).astype(np.uint8)
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, face_bgr)
            return True
    except:
        pass
    return False

count = 0
for subject, seq, emotion, src_path in tqdm(peak_data, desc="预处理"):
    # 确定子集
    if subject in split['train']:
        subset = 'train'
    elif subject in split['val']:
        subset = 'val'
    elif subject in split['test']:
        subset = 'test'
    else:
        continue
    dst_path = os.path.join(OUTPUT_DIR, subset, emotion, f"{subject}_{seq}.png")
    if process_and_save(src_path, dst_path):
        count += 1

print(f"成功保存 {count} 张图像")