import os, sys, cv2, torch, numpy as np, pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from torchvision import transforms
import timm
from PIL import Image
from torchvision.transforms import functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ==================== 路径与配置 ====================
sys.path.insert(0, r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing")
from models.bisenet import BiSeNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROCESSED_DIR = r"F:\python\2024218729zby_ML\EG-ADP\data\processed_ckplus_v2"
META_CSV = os.path.join(PROCESSED_DIR, 'metadata', 'samples.csv')
BISENET_WEIGHT = r"F:\python\2024218729zby_ML\EG-ADP\src\face-parsing\weights\resnet18.pt"
EMOTION_MODEL_PATH = "best_emotion_ckplus.pth"   # 请确认文件存在

# ==================== 标签映射 ====================
code_to_idx = {1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
idx2emo = {0:'anger',1:'disgust',2:'fear',3:'happiness',4:'sadness',5:'surprise'}
EMOTIONS = list(idx2emo.values())

# ==================== 加载模型 ====================
print("加载 BiSeNet...")
bisenet = BiSeNet(num_classes=19, backbone_name='resnet18')
bisenet.load_state_dict(torch.load(BISENET_WEIGHT, map_location=DEVICE), strict=False)
bisenet.eval().to(DEVICE)

print("加载情感模型...")
emotion_model = timm.create_model('resnet18', pretrained=False, num_classes=6)
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model.eval().to(DEVICE)
emo_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224,224)),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

print("加载 FaceNet...")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

print("加载 ArcFace (antelopev2)...")
arcface_app = FaceAnalysis(name='antelopev2', root=r'F:\python\models\insightface',
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
arcface_app.prepare(ctx_id=0, det_size=(224,224))

print("加载额外身份模型 (buffalo_l)...")
extra_app = FaceAnalysis(name='buffalo_l', root=r'F:\python\models\insightface',
                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
extra_app.prepare(ctx_id=0, det_size=(224,224))

# ==================== 工具函数 ====================
def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-12)

def get_parsing(img_rgb):
    h,w = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb).resize((512,512), Image.BILINEAR)
    tensor = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
    tensor = F.normalize(tensor, [0.485,0.456,0.406],[0.229,0.224,0.225])
    with torch.no_grad():
        out = bisenet(tensor)
        out = out[0] if isinstance(out, tuple) else out
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
    return cv2.resize(parsing.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

def apply_ours(img_rgb):
    parsing = get_parsing(img_rgb)
    brow = np.isin(parsing,[2,3]).astype(np.float32)
    mouth = np.isin(parsing,[11,12,13]).astype(np.float32)
    brow = cv2.dilate(brow, np.ones((3,3),np.uint8),1)
    mouth = cv2.dilate(mouth, np.ones((11,11),np.uint8),1)
    emo_mask = np.clip(brow+mouth,0,1)
    id_mask = np.isin(parsing,[4,5,10]).astype(np.float32)
    id_mask = cv2.dilate(id_mask, np.ones((3,3),np.uint8),1)
    nc_mask = np.clip(1.0-emo_mask-id_mask,0,1)
    img_emo = img_rgb.copy()
    img_id = cv2.GaussianBlur(img_rgb, (17,17),0)
    img_nc = cv2.resize(cv2.resize(img_rgb, (img_rgb.shape[1]//14, img_rgb.shape[0]//14), interpolation=cv2.INTER_LINEAR),
                        (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (emo_mask[...,None]*img_emo + id_mask[...,None]*img_id + nc_mask[...,None]*img_nc).astype(np.uint8)

def get_facenet_embedding(img_rgb):
    img = cv2.resize(img_rgb, (160,160))
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)/255.
    tensor = (tensor-0.5)/0.5
    with torch.no_grad():
        emb = facenet(tensor).cpu().numpy().flatten()
    return l2norm(emb)

def arcface_detect_and_extract(img_rgb, app):
    faces = app.get(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if faces:
        return faces[0]
    return None

def arcface_embed_from_aligned(aligned_rgb_112, app):
    aligned_bgr = cv2.cvtColor(aligned_rgb_112, cv2.COLOR_RGB2BGR)
    feat = app.models['recognition'].get_feat(aligned_bgr).flatten()
    return l2norm(feat)

def predict_emotion(img_rgb):
    tensor = emo_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return emotion_model(tensor).argmax(1).item()

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0

def evaluate_identity(probe_embs, probe_subjects, gallery_embs):
    genuine, impostor, ranks = [], [], []
    for emb, subj in zip(probe_embs, probe_subjects):
        if subj not in gallery_embs:
            continue
        genuine.append(np.dot(gallery_embs[subj], emb))
        for other_subj, other_emb in gallery_embs.items():
            if other_subj != subj:
                impostor.append(np.dot(other_emb, emb))
        sims = {s: np.dot(gallery_embs[s], emb) for s in gallery_embs}
        sorted_subjs = sorted(sims, key=sims.get, reverse=True)
        rank = sorted_subjs.index(subj) + 1
        ranks.append(rank)
    if not genuine or not impostor:
        return 0,0,0
    y_true = [1]*len(genuine) + [0]*len(impostor)
    y_score = genuine + impostor
    auc = roc_auc_score(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    top1 = sum(r==1 for r in ranks)/len(ranks)
    return top1, auc, eer

# ---- 新增：Laplace 噪声工具 ----
def laplace_noise(image, sigma):
    noise = np.random.laplace(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def semantic_laplace(img_rgb, sigma_emo, sigma_id, sigma_nc):
    parsing = get_parsing(img_rgb)
    brow = np.isin(parsing,[2,3]).astype(np.float32); mouth = np.isin(parsing,[11,12,13]).astype(np.float32)
    brow = cv2.dilate(brow, np.ones((3,3),np.uint8),1)
    mouth = cv2.dilate(mouth, np.ones((11,11),np.uint8),1)
    emo_mask = np.clip(brow+mouth,0,1)
    id_mask = np.isin(parsing,[4,5,10]).astype(np.float32)
    id_mask = cv2.dilate(id_mask, np.ones((3,3),np.uint8),1)
    nc_mask = np.clip(1.0-emo_mask-id_mask,0,1)
    noise_emo = np.random.laplace(0, sigma_emo, img_rgb.shape).astype(np.float32)
    noise_id  = np.random.laplace(0, sigma_id,  img_rgb.shape).astype(np.float32)
    noise_nc  = np.random.laplace(0, sigma_nc,  img_rgb.shape).astype(np.float32)
    img_f = img_rgb.astype(np.float32)
    result = emo_mask[...,None]* (img_f + noise_emo) + \
             id_mask[...,None]*  (img_f + noise_id)  + \
             nc_mask[...,None]*  (img_f + noise_nc)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# ---- DeepPrivacy ----
try:
    from deep_privacy import build_anonymizer
    deep_privacy_anonymizer = build_anonymizer()
    DEEP_PRIVACY_AVAILABLE = True
except ImportError:
    DEEP_PRIVACY_AVAILABLE = False
    deep_privacy_anonymizer = None
    print("DeepPrivacy 不可用，将跳过其基线。")

def deep_privacy_anonymize(img_rgb):
    if not DEEP_PRIVACY_AVAILABLE:
        return img_rgb
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    anonymized = deep_privacy_anonymizer(img_bgr)
    return cv2.cvtColor(anonymized, cv2.COLOR_BGR2RGB)

# ==================== 保护方法字典 ====================
protect_methods = {
    'Original': lambda img: img,
    'Gaussian Blur': lambda img: cv2.GaussianBlur(img, (21,21),0),
    'Mosaic b12': lambda img: cv2.resize(cv2.resize(img, (img.shape[1]//12, img.shape[0]//12), interpolation=cv2.INTER_LINEAR),
                                         (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST),
    'Strong Mosaic': lambda img: cv2.resize(cv2.resize(img, (img.shape[1]//16, img.shape[0]//16), interpolation=cv2.INTER_LINEAR),
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST),
    'EG-ADP': apply_ours,
    'Laplace-DP-σ5':  lambda img: laplace_noise(img, sigma=5),
    'Laplace-DP-σ10': lambda img: laplace_noise(img, sigma=10),
    'Laplace-DP-σ20': lambda img: laplace_noise(img, sigma=20),
    'Laplace-DP-σ30': lambda img: laplace_noise(img, sigma=30),
    'Semantic-Laplace-low':    lambda img: semantic_laplace(img, sigma_emo=3, sigma_id=10, sigma_nc=20),
    'Semantic-Laplace-medium': lambda img: semantic_laplace(img, sigma_emo=5, sigma_id=15, sigma_nc=30),
    'Semantic-Laplace-high':   lambda img: semantic_laplace(img, sigma_emo=8, sigma_id=25, sigma_nc=45),
}
if DEEP_PRIVACY_AVAILABLE:
    protect_methods['DeepPrivacy'] = deep_privacy_anonymize

# ==================== 数据准备 ====================
df = pd.read_csv(META_CSV)
df_peak = df[~df['emotion_code'].isna()].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

test_subjects = set(df[df['subset']=='test']['subject'].unique())
df_test = df_peak[df_peak['subject'].isin(test_subjects)]

df_neutral = df[(df['emotion'].str.lower()=='neutral') & (df['subject'].isin(test_subjects))]
gallery_facenet, gallery_arcface, gallery_extra = {}, {}, {}
for subj, group in df_neutral.groupby('subject'):
    row = group.iloc[0]
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 修复函数名
    gallery_facenet[subj] = get_facenet_embedding(img_rgb)
    face_af = arcface_detect_and_extract(img_rgb, arcface_app)
    if face_af is not None:
        gallery_arcface[subj] = l2norm(face_af.normed_embedding)
    face_ex = arcface_detect_and_extract(img_rgb, extra_app)
    if face_ex is not None:
        gallery_extra[subj] = l2norm(face_ex.normed_embedding)

# ==================== 评估循环 ====================
summary_rows = []

for method_name, protect_fn in tqdm(protect_methods.items(), desc="Methods"):
    print(f"\n===== {method_name} =====")
    det_data = {'facenet': ([], []), 'arcface': ([], [], []), 'extra': ([], [], [])}
    fixed_data = {'facenet': ([], []), 'arcface': ([], []), 'extra': ([], [])}
    pre_data = {'facenet': ([], []), 'arcface': ([], []), 'extra': ([], [])}

    for _, row in df_test.iterrows():
        img = cv2.imread(row['image_path'])
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        protected = protect_fn(img_rgb)
        subj = row['subject']

        emb_fn = get_facenet_embedding(protected)
        det_data['facenet'][0].append(emb_fn); det_data['facenet'][1].append(subj)
        fixed_data['facenet'][0].append(emb_fn); fixed_data['facenet'][1].append(subj)
        pre_data['facenet'][0].append(emb_fn); pre_data['facenet'][1].append(subj)

        face_af = arcface_detect_and_extract(protected, arcface_app)
        if face_af is not None:
            det_data['arcface'][0].append(l2norm(face_af.normed_embedding)); det_data['arcface'][1].append(subj)
            det_data['arcface'][2].append(1)
        else:
            det_data['arcface'][2].append(0)
        face_ex = arcface_detect_and_extract(protected, extra_app)
        if face_ex is not None:
            det_data['extra'][0].append(l2norm(face_ex.normed_embedding)); det_data['extra'][1].append(subj)
            det_data['extra'][2].append(1)
        else:
            det_data['extra'][2].append(0)

        orig_face = arcface_detect_and_extract(img_rgb, arcface_app)
        if orig_face is not None:
            lm = orig_face.kps
            aligned = face_align.norm_crop(protected, lm, image_size=112)
            emb_af = arcface_embed_from_aligned(aligned, arcface_app)
            fixed_data['arcface'][0].append(emb_af); fixed_data['arcface'][1].append(subj)
            emb_ex = arcface_embed_from_aligned(aligned, extra_app)
            fixed_data['extra'][0].append(emb_ex); fixed_data['extra'][1].append(subj)

        pre_resized = cv2.resize(protected, (112,112))
        emb_af_pre = arcface_embed_from_aligned(pre_resized, arcface_app)
        pre_data['arcface'][0].append(emb_af_pre); pre_data['arcface'][1].append(subj)
        emb_ex_pre = arcface_embed_from_aligned(pre_resized, extra_app)
        pre_data['extra'][0].append(emb_ex_pre); pre_data['extra'][1].append(subj)

    for protocol, data_dict, gallery_dict in [
        ('detector-dependent', det_data, {'facenet':gallery_facenet, 'arcface':gallery_arcface, 'extra':gallery_extra}),
        ('fixed-landmark', fixed_data, {'facenet':gallery_facenet, 'arcface':gallery_arcface, 'extra':gallery_extra}),
        ('pre-aligned', pre_data, {'facenet':gallery_facenet, 'arcface':gallery_arcface, 'extra':gallery_extra})
    ]:
        for model_name in ['facenet','arcface','extra']:
            embs, subs = data_dict[model_name][:2]
            if not embs: continue
            top1, auc, eer = evaluate_identity(embs, subs, gallery_dict[model_name])
            det_rate = None; overall_top1 = top1
            if model_name in ('arcface','extra') and protocol == 'detector-dependent':
                det_flags = data_dict[model_name][2]
                det_rate = sum(det_flags)/len(det_flags) if det_flags else 1.0
                overall_top1 = top1 * det_rate
            summary_rows.append({
                'Method': method_name,
                'Protocol': protocol,
                'Model': model_name,
                'Top-1 (Detected-only)': top1,
                'Overall Top-1': overall_top1,
                'AUC': auc,
                'EER': eer,
                'DetectionRate': det_rate
            })
            print(f"{method_name:20s} | {protocol:20s} | {model_name:8s} | "
                  f"DetTop1={top1:.4f} | OverallTop1={overall_top1:.4f} | "
                  f"AUC={auc:.4f} | EER={eer:.4f}" + (f" | DetRate={det_rate:.4f}" if det_rate else ""))

df_res = pd.DataFrame(summary_rows)
df_res.to_csv("extended_baselines_results.csv", index=False)
print("\n所有扩展基线实验完成，结果保存至 extended_baselines_results.csv")