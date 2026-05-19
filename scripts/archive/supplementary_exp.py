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
EMOTION_MODEL_PATH = "best_emotion_ckplus.pth"      # 混合训练模型

# ==================== 标签映射（6类，无contempt） ====================
# 根据你 CK+ 的 emotion_code 定义
code_to_idx = {
    1: 0,   # anger
    3: 1,   # disgust
    4: 2,   # fear
    5: 3,   # happiness
    6: 4,   # sadness
    7: 5    # surprise
}
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

# 保护方法字典
protect_methods = {
    'Original': lambda img: img,
    'Gaussian Blur': lambda img: cv2.GaussianBlur(img, (21,21),0),
    'Mosaic b12': lambda img: cv2.resize(cv2.resize(img, (img.shape[1]//12, img.shape[0]//12), interpolation=cv2.INTER_LINEAR),
                                         (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST),
    'Strong Mosaic': lambda img: cv2.resize(cv2.resize(img, (img.shape[1]//16, img.shape[0]//16), interpolation=cv2.INTER_LINEAR),
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST),
    'EG-ADP': apply_ours
}

# 嵌入提取函数
def get_facenet_embedding(img_rgb):
    img = cv2.resize(img_rgb, (160,160))
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)/255.
    tensor = (tensor-0.5)/0.5
    with torch.no_grad():
        emb = facenet(tensor).cpu().numpy().flatten()
    return l2norm(emb)

def arcface_detect_and_extract(img_rgb, app):
    """返回检测到的人脸对象（含 kps 和 normed_embedding），失败返回 None"""
    faces = app.get(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if faces:
        return faces[0]
    return None

def arcface_embed_from_aligned(aligned_rgb_112, app):
    """从 112×112 对齐图像直接提取 embedding"""
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

# ==================== 数据准备 ====================
df = pd.read_csv(META_CSV)
df_peak = df[~df['emotion_code'].isna()].copy()
df_peak['emotion_code'] = df_peak['emotion_code'].astype(int)

# 使用原始 test split（可改为 5‑折循环）
test_subjects = set(df[df['subset']=='test']['subject'].unique())
df_test = df_peak[df_peak['subject'].isin(test_subjects)]

# Gallery（中性帧）
df_neutral = df[(df['emotion'].str.lower()=='neutral') & (df['subject'].isin(test_subjects))]
gallery_facenet, gallery_arcface, gallery_extra = {}, {}, {}
for subj, group in df_neutral.groupby('subject'):
    row = group.iloc[0]
    img = cv2.imread(row['image_path'])
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # FaceNet
    gallery_facenet[subj] = get_facenet_embedding(img_rgb)
    # ArcFace
    face_af = arcface_detect_and_extract(img_rgb, arcface_app)
    if face_af is not None:
        gallery_arcface[subj] = l2norm(face_af.normed_embedding)
    # Extra
    face_ex = arcface_detect_and_extract(img_rgb, extra_app)
    if face_ex is not None:
        gallery_extra[subj] = l2norm(face_ex.normed_embedding)

# ==================== 实验 1：Detector‑free 攻击 + 多模型 ====================
print("\n===== Experiment 1: Detector‑free attacks =====")

all_results = []      # 存储详细结果
summary_rows = []     # 汇总输出

for method_name, protect_fn in protect_methods.items():
    print(f"\n--- {method_name} ---")
    # 各类 probe 数据
    det_data = {'facenet': ([], []), 'arcface': ([], [], [], []), 'extra': ([], [], [], [])}
    # arcface/extra: (embs, subjects, detection_flags, rank_info)
    fixed_data = {'facenet': ([], []), 'arcface': ([], []), 'extra': ([], [])}
    pre_data = {'facenet': ([], []), 'arcface': ([], []), 'extra': ([], [])}
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        img = cv2.imread(row['image_path'])
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        protected = protect_fn(img_rgb)
        subj = row['subject']
        
        # ---- detector‑dependent ----
        det_data['facenet'][0].append(get_facenet_embedding(protected))
        det_data['facenet'][1].append(subj)
        
        face_af = arcface_detect_and_extract(protected, arcface_app)
        if face_af is not None:
            det_data['arcface'][0].append(l2norm(face_af.normed_embedding))
            det_data['arcface'][1].append(subj)
            det_data['arcface'][2].append(1)  # detected
        else:
            det_data['arcface'][2].append(0)
        
        face_ex = arcface_detect_and_extract(protected, extra_app)
        if face_ex is not None:
            det_data['extra'][0].append(l2norm(face_ex.normed_embedding))
            det_data['extra'][1].append(subj)
            det_data['extra'][3].append(1)
        else:
            det_data['extra'][3].append(0)
        
        # ---- fixed‑landmark（使用原图关键点） ----
        orig_face_af = arcface_detect_and_extract(img_rgb, arcface_app)
        if orig_face_af is not None:
            lm = orig_face_af.kps
            aligned_prot = face_align.norm_crop(protected, lm, image_size=112)
            emb_af = arcface_embed_from_aligned(aligned_prot, arcface_app)
            fixed_data['arcface'][0].append(emb_af)
            fixed_data['arcface'][1].append(subj)
            
            aligned_prot_ex = face_align.norm_crop(protected, lm, image_size=112)
            emb_ex = arcface_embed_from_aligned(aligned_prot_ex, extra_app)
            fixed_data['extra'][0].append(emb_ex)
            fixed_data['extra'][1].append(subj)
        # FaceNet 无固定框，直接使用整图，但它本身就不需要检测器，因此不区分协议
        fixed_data['facenet'][0].append(get_facenet_embedding(protected))
        fixed_data['facenet'][1].append(subj)
        
        # ---- pre‑aligned（假设输入已对齐，直接 resize） ----
        pre_resized = cv2.resize(protected, (112,112))
        emb_af_pre = arcface_embed_from_aligned(pre_resized, arcface_app)
        pre_data['arcface'][0].append(emb_af_pre)
        pre_data['arcface'][1].append(subj)
        
        emb_ex_pre = arcface_embed_from_aligned(pre_resized, extra_app)
        pre_data['extra'][0].append(emb_ex_pre)
        pre_data['extra'][1].append(subj)
        # FaceNet
        pre_data['facenet'][0].append(get_facenet_embedding(protected))
        pre_data['facenet'][1].append(subj)
    
    # 计算指标
    for protocol, data_dict, gallery_dict in [
        ('detector-dependent', det_data, {'facenet':gallery_facenet, 'arcface':gallery_arcface, 'extra':gallery_extra}),
        ('fixed-landmark', fixed_data, {'facenet':gallery_facenet, 'arcface':gallery_arcface, 'extra':gallery_extra}),
        ('pre-aligned', pre_data, {'facenet':gallery_facenet, 'arcface':gallery_arcface, 'extra':gallery_extra})
    ]:
        for model in ['facenet','arcface','extra']:
            embs, subs = data_dict[model][:2]
            if not embs:
                continue
            top1, auc, eer = evaluate_identity(embs, subs, gallery_dict[model])
            det_rate = None
            if model in ['arcface','extra'] and protocol == 'detector-dependent':
                det_flags = data_dict[model][2]
                det_rate = sum(det_flags)/len(det_flags) if det_flags else 1.0
            summary_rows.append({
                'Method': method_name,
                'Protocol': protocol,
                'Model': model,
                'Top-1': top1,
                'AUC': auc,
                'EER': eer,
                'DetectionRate': det_rate
            })
            print(f"{method_name:12s} | {protocol:20s} | {model:8s} | Top-1={top1:.4f} | AUC={auc:.4f} | EER={eer:.4f}" + (f" | DetRate={det_rate:.4f}" if det_rate else ""))

df_attack = pd.DataFrame(summary_rows)
df_attack.to_csv("detector_free_attacks_v2.csv", index=False)
print("\nExperiment 1 results saved to detector_free_attacks_v2.csv")

# ==================== 实验 2：参数敏感性（固定混合训练模型） ====================
print("\n===== Experiment 2: Parameter sensitivity =====")

def apply_ours_param(img_rgb, blur_k, mosaic_b):
    parsing = get_parsing(img_rgb)
    brow = np.isin(parsing,[2,3]).astype(np.float32)
    mouth = np.isin(parsing,[11,12,13]).astype(np.float32)
    brow = cv2.dilate(brow, np.ones((3,3),np.uint8),1)
    mouth = cv2.dilate(mouth, np.ones((11,11),np.uint8),1)
    emo_mask = np.clip(brow+mouth,0,1)
    id_mask = np.isin(parsing,[4,5,10]).astype(np.float32)
    id_mask = cv2.dilate(id_mask, np.ones((3,3),np.uint8),1)
    nc_mask = np.clip(1.0-emo_mask-id_mask,0,1)
    if blur_k %2 == 0: blur_k+=1
    img_id = cv2.GaussianBlur(img_rgb, (blur_k,blur_k),0)
    img_nc = cv2.resize(cv2.resize(img_rgb, (img_rgb.shape[1]//mosaic_b, img_rgb.shape[0]//mosaic_b), interpolation=cv2.INTER_LINEAR),
                        (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (emo_mask[...,None]*img_rgb + id_mask[...,None]*img_id + nc_mask[...,None]*img_nc).astype(np.uint8)

blur_kernels = [7, 11, 15, 17, 21, 25]
mosaic_blocks = [8, 10, 12, 14, 16, 20]
sens_results = []

# 固定 mosaic=14，变化模糊核
for bk in blur_kernels:
    all_preds, all_labels = [], []
    probe_embs_fixed, probe_subs_fixed = [], []
    probe_embs_det, probe_subs_det = [], []
    det_flags = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        img = cv2.imread(row['image_path'])
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        protected = apply_ours_param(img_rgb, bk, 14)
        # 情感预测
        pred = predict_emotion(protected)
        all_preds.append(pred)
        all_labels.append(code_to_idx[int(row['emotion_code'])])
        # ArcFace fixed-landmark
        orig_face = arcface_detect_and_extract(img_rgb, arcface_app)
        if orig_face is not None:
            aligned_prot = face_align.norm_crop(protected, orig_face.kps, image_size=112)
            emb = arcface_embed_from_aligned(aligned_prot, arcface_app)
            probe_embs_fixed.append(emb)
            probe_subs_fixed.append(row['subject'])
        # ArcFace detector-dependent
        face_prot = arcface_detect_and_extract(protected, arcface_app)
        if face_prot is not None:
            probe_embs_det.append(l2norm(face_prot.normed_embedding))
            probe_subs_det.append(row['subject'])
            det_flags.append(1)
        else:
            det_flags.append(0)
    # 计算指标
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    top1_fixed, _, _ = evaluate_identity(probe_embs_fixed, probe_subs_fixed, gallery_arcface) if probe_embs_fixed else (0,0,0)
    top1_det, _, _ = evaluate_identity(probe_embs_det, probe_subs_det, gallery_arcface) if probe_embs_det else (0,0,0)
    det_rate = np.mean(det_flags) if det_flags else 1.0
    sens_results.append(('blur_kernel', bk, acc, macro_f1, top1_fixed, top1_det, det_rate))
    print(f"blur_k={bk}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, Fixed Top-1={top1_fixed:.4f}, Det Top-1={top1_det:.4f}, DetRate={det_rate:.4f}")

# 固定 blur=17，变化马赛克块
for mb in mosaic_blocks:
    all_preds, all_labels = [], []
    probe_embs_fixed, probe_subs_fixed = [], []
    probe_embs_det, probe_subs_det = [], []
    det_flags = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        img = cv2.imread(row['image_path'])
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        protected = apply_ours_param(img_rgb, 17, mb)
        pred = predict_emotion(protected)
        all_preds.append(pred)
        all_labels.append(code_to_idx[int(row['emotion_code'])])
        orig_face = arcface_detect_and_extract(img_rgb, arcface_app)
        if orig_face is not None:
            aligned_prot = face_align.norm_crop(protected, orig_face.kps, image_size=112)
            emb = arcface_embed_from_aligned(aligned_prot, arcface_app)
            probe_embs_fixed.append(emb)
            probe_subs_fixed.append(row['subject'])
        face_prot = arcface_detect_and_extract(protected, arcface_app)
        if face_prot is not None:
            probe_embs_det.append(l2norm(face_prot.normed_embedding))
            probe_subs_det.append(row['subject'])
            det_flags.append(1)
        else:
            det_flags.append(0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    top1_fixed, _, _ = evaluate_identity(probe_embs_fixed, probe_subs_fixed, gallery_arcface) if probe_embs_fixed else (0,0,0)
    top1_det, _, _ = evaluate_identity(probe_embs_det, probe_subs_det, gallery_arcface) if probe_embs_det else (0,0,0)
    det_rate = np.mean(det_flags) if det_flags else 1.0
    sens_results.append(('mosaic_block', mb, acc, macro_f1, top1_fixed, top1_det, det_rate))
    print(f"mosaic_b={mb}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, Fixed Top-1={top1_fixed:.4f}, Det Top-1={top1_det:.4f}, DetRate={det_rate:.4f}")

df_sens = pd.DataFrame(sens_results, columns=['Parameter','Value','Acc','Macro-F1','FixedCrop_Top1','DetDep_Top1','DetectionRate'])
df_sens.to_csv("sensitivity_analysis_v2.csv", index=False)
print("\nExperiment 2 results saved to sensitivity_analysis_v2.csv")

print("\n所有补充实验完成。")