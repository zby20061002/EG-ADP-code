import os, sys, cv2, shutil, random, numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
# ------- 设置 -------
DATA_ROOT = r"E:\image\Oulu-CASIA\data\Oulu_CASIA_NIR_VIS\VL\Strong"
OUTPUT_ROOT = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_oulu"
SEED = 42
random.seed(SEED)

emotion_folders = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
# 映射到小写或保持原样，后续训练用数字标签 0-5
emo2idx = {e:i for i,e in enumerate(emotion_folders)}

# 读取所有身份 ID
identities = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
random.shuffle(identities)
n_train, n_val = 60, 10
split = {
    'train': identities[:n_train],
    'val': identities[n_train:n_train+n_val],
    'test': identities[n_train+n_val:]
}

# 创建输出目录
for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_ROOT, subset), exist_ok=True)

# 人脸检测器（MTCNN）
mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

def process_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None: return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        face = mtcnn(img_rgb, save_path=None)
        if face is not None:
            face = face.permute(1,2,0).numpy()
            face = (face * 255).astype(np.uint8)
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, face_bgr)
            return True
    except:
        pass
    return False

# 遍历处理
total_proc = 0
for subset, subs in split.items():
    for subj in tqdm(subs, desc=f'Processing {subset}'):
        subj_dir = os.path.join(DATA_ROOT, subj)
        for emo in emotion_folders:
            emo_dir = os.path.join(subj_dir, emo)
            if not os.path.exists(emo_dir): continue
            for fname in os.listdir(emo_dir):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    src = os.path.join(emo_dir, fname)
                    dst_dir = os.path.join(OUTPUT_ROOT, subset, subj)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, f'{emo}_{fname}')
                    if process_image(src, dst):
                        total_proc += 1

print(f'成功处理图像总数: {total_proc}')