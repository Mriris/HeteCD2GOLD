import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage import io
from tqdm import tqdm
import warnings
from models.hetecd import hetecd as Net
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
from matplotlib import rcParams

# 抑制警告
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ===== 顶部参数配置 =====
CHECKPOINT = r'checkpoints/gold/trios43/EXP20251024095352'
TEST_DIR = r'D:\0Program\Datasets\241120\Compare\Datas\tsne'
OUTPUT_DIR = r'results/tsne_per_image3'
DEVICE = 'cuda'
NUM_WORKERS = 0  # 单图像处理，使用0避免多进程开销

# t-SNE 与加速参数（可通过命令行覆盖）
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_MAX_ITER = 500  # 降低迭代次数以提速
PCA_DIM = 50       # t-SNE 前先用 PCA 降到 50 维
MAX_PIXELS = 2000  # 每张图用于 t-SNE 的像素上限（分层采样）
RANK_SAMPLE = 3000 # 用于快速排序评分的采样上限
SILHOUETTE_SAMPLE = 1000  # 轮廓系数抽样大小

# 可视化与筛选参数
TOP_N = 10  # 保存可分性最好的前N张图
MIN_SAMPLES_PER_CLASS = 10  # 每个类别至少需要的样本数

# 绘图可读性配置（不影响评分，只影响可视化）
BALANCE_PLOT = True          # 绘图时是否做类别均衡抽样
NEG_POS_RATIO = 3            # 负类与正类绘图比例上限（例如3表示至多3倍）
PLOT_POS_MAX = 800           # 绘图时正类上限
JOINT_TSNE_FOR_PLOT = True   # 为提高可比性，绘图使用"师生合并后一次t-SNE"，再拆分

# 图表元素显示配置
# 可通过命令行参数覆盖：
#   --no-title: 隐藏标题
#   --no-score: 隐藏可分性分数（但保留标题）
#   --no-axis-labels: 隐藏坐标轴标签
#   --no-legend: 隐藏图例
#   --no-grid: 隐藏网格
# 示例：python predict_tsne.py --no-title --no-axis-labels --no-legend --no-grid
SHOW_TITLE = True            # 显示标题（包含模型类型和可分性分数）
SHOW_SCORE_IN_TITLE = False   # 在标题中显示可分性分数
SHOW_AXIS_LABELS = True      # 显示坐标轴标签
SHOW_LEGEND = True           # 显示图例
SHOW_GRID = True             # 显示网格


class FeatureExtractor:
    """使用 forward hook 提取中间特征"""
    def __init__(self):
        self.features = None
    
    def hook(self, module, input, output):
        """Hook 函数，保存模块输出"""
        self.features = output.detach()


class SingleImageDataset(Dataset):
    """单图像数据集，用于逐个处理"""
    def __init__(self, test_dir):
        self.test_dir = test_dir
        
        # 读取图像路径
        img_A_dir = os.path.join(test_dir, 'A')
        img_B_dir = os.path.join(test_dir, 'B')
        img_C_dir = os.path.join(test_dir, 'C')
        
        if not os.path.exists(img_A_dir):
            raise ValueError(f"Dataset directory A does not exist: {img_A_dir}")
        if not os.path.exists(img_B_dir):
            raise ValueError(f"Dataset directory B does not exist: {img_B_dir}")
        if not os.path.exists(img_C_dir):
            raise ValueError(f"Dataset directory C does not exist: {img_C_dir}")
        
        self.img_A_dir = img_A_dir
        self.img_B_dir = img_B_dir
        self.img_C_dir = img_C_dir
        
        self.img_names = sorted([f for f in os.listdir(img_A_dir) if f.endswith(('.png', '.tif', '.jpg'))])
        
        # 检测标签目录
        label_dir = os.path.join(test_dir, 'label')
        self.has_label = os.path.exists(label_dir)
        
        if not self.has_label:
            raise ValueError("Label directory is required for separability analysis")
        
        print(f"Loaded {len(self.img_names)} test samples")
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # 读取 A、B、C 三个图像
        img_A_path = os.path.join(self.img_A_dir, img_name)
        img_B_path = os.path.join(self.img_B_dir, img_name)
        img_C_path = os.path.join(self.img_C_dir, img_name)
        
        img_A = io.imread(img_A_path)
        img_B = io.imread(img_B_path)
        img_C = io.imread(img_C_path)
        
        # 转换为张量
        img_A = torch.from_numpy(img_A).permute(2, 0, 1).float() / 255.0
        img_B = torch.from_numpy(img_B).permute(2, 0, 1).float() / 255.0
        img_C = torch.from_numpy(img_C).permute(2, 0, 1).float() / 255.0
        
        # 读取标签
        label_path = os.path.join(self.test_dir, 'label', img_name)
        label = io.imread(label_path)
        label = (label > 127).astype(np.uint8)
        
        return img_A, img_B, img_C, label, img_name


def normalize_state_dict(state_dict):
    """规范化权重键名"""
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        prefixes_to_remove = [
            '_orig_mod.module.',
            'module._orig_mod.',
            '_orig_mod.',
            'module.',
        ]
        
        for prefix in prefixes_to_remove:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def load_model(checkpoint_path, device='cuda'):
    """Load complete model weights (including teacher and student networks)"""
    if os.path.isdir(checkpoint_path):
        best_path = os.path.join(checkpoint_path, 'best_checkpoint.pth')
        if os.path.exists(best_path):
            checkpoint_path = best_path
            print(f"Found best_checkpoint.pth")
        else:
            last_path = os.path.join(checkpoint_path, 'last_checkpoint.pth')
            if os.path.exists(last_path):
                checkpoint_path = last_path
                print(f"Using last_checkpoint.pth")
            else:
                raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_path}")
    
    model = Net(input_nc=3, output_nc=2).to(device)
    
    print(f"Loading weights: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        epoch_info = checkpoint.get('epoch', 'unknown')
        print(f"Loaded weights from epoch: {epoch_info}")
    else:
        state_dict = checkpoint
    
    state_dict = normalize_state_dict(state_dict)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        print("✓ Weights loaded successfully")
    else:
        if len(missing_keys) > 0:
            print(f"⚠ Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if len(unexpected_keys) > 0:
            print(f"⚠ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    model.use_teacher = True
    model.eval()
    
    return model


def extract_pixel_features(model, img_A, img_B, img_C, label, device='cuda'):
    """
    提取单张图像所有像素点的特征
    
    Returns:
        student_features: [H*W, C] 学生模型特征
        teacher_features: [H*W, C] 教师模型特征
        pixel_labels: [H*W] 每个像素的标签
    """
    model.eval()
    
    # 创建特征提取器
    student_extractor = FeatureExtractor()
    teacher_extractor = FeatureExtractor()
    
    # 注册 hook
    student_hook = model.CD_Decoder.dense_1x.register_forward_hook(student_extractor.hook)
    teacher_hook = model.teacher_decoder.dense_1x.register_forward_hook(teacher_extractor.hook)
    
    # 添加batch维度
    img_A = img_A.unsqueeze(0).to(device)
    img_B = img_B.unsqueeze(0).to(device)
    img_C = img_C.unsqueeze(0).to(device)
    
    with torch.no_grad():
        _ = model(img_A, img_B, img_C)
    
    # 提取特征 [1, C, H, W]
    student_feat = student_extractor.features  # [1, C, H, W]
    teacher_feat = teacher_extractor.features  # [1, C, H, W]
    
    # 移除 batch 维度并转置为 [H*W, C]
    student_feat = student_feat.squeeze(0).permute(1, 2, 0).contiguous()  # [H, W, C]
    teacher_feat = teacher_feat.squeeze(0).permute(1, 2, 0).contiguous()  # [H, W, C]
    
    H, W, C = student_feat.shape
    
    student_feat = student_feat.view(-1, C).cpu().numpy()  # [H*W, C]
    teacher_feat = teacher_feat.view(-1, C).cpu().numpy()  # [H*W, C]
    
    # 调整label大小以匹配特征图
    label_resized = F.interpolate(
        torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float(),
        size=(H, W),
        mode='nearest'
    ).squeeze().numpy().astype(np.uint8)
    
    pixel_labels = label_resized.reshape(-1)  # [H*W]
    
    # 移除 hook
    student_hook.remove()
    teacher_hook.remove()
    
    return student_feat, teacher_feat, pixel_labels


def calculate_separability(features_2d, labels):
    """
    计算特征的可分性（使用轮廓系数）
    
    Returns:
        score: 轮廓系数，范围[-1, 1]，越接近1表示可分性越好
    """
    # 需要至少2个类别
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0
    
    # 抽样以加速 silhouette 计算
    if len(labels) > SILHOUETTE_SAMPLE:
        # 分层抽样
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        n0 = min(len(idx0), SILHOUETTE_SAMPLE // 2)
        n1 = min(len(idx1), SILHOUETTE_SAMPLE - n0)
        sel = np.concatenate([
            np.random.choice(idx0, n0, replace=False) if n0 > 0 else np.array([], dtype=int),
            np.random.choice(idx1, n1, replace=False) if n1 > 0 else np.array([], dtype=int)
        ])
        features_2d = features_2d[sel]
        labels = labels[sel]

    try:
        score = silhouette_score(features_2d, labels)
        return score
    except:
        return -1.0


def visualize_single_image_tsne(student_2d, teacher_2d, labels, img_name, 
                                 student_score, teacher_score, save_path,
                                 perplexity=30, random_state=42):
    """
    Generate separate t-SNE plots for student and teacher models
    
    Args:
        student_2d: [N, 2] Student model 2D features
        teacher_2d: [N, 2] Teacher model 2D features
        labels: [N] Pixel labels
        img_name: Image name
        student_score: Student separability score
        teacher_score: Teacher separability score
        save_path: Base save path (will be split into _student.png and _teacher.png)
    """
    # 可视化前类别均衡抽样，避免极度失衡导致图形"淹没"
    vis_idx = np.arange(len(labels))
    if BALANCE_PLOT:
        idx_pos = np.where(labels == 1)[0]
        idx_neg = np.where(labels == 0)[0]
        if len(idx_pos) > 0 and len(idx_neg) > 0:
            # 限制正类数量
            n_pos = min(len(idx_pos), PLOT_POS_MAX)
            sel_pos = np.random.choice(idx_pos, n_pos, replace=False)
            # 按比例限制负类
            n_neg = min(len(idx_neg), n_pos * NEG_POS_RATIO)
            sel_neg = np.random.choice(idx_neg, n_neg, replace=False)
            vis_idx = np.concatenate([sel_pos, sel_neg])
            np.random.shuffle(vis_idx)

    student_2d = student_2d[vis_idx]
    teacher_2d = teacher_2d[vis_idx]
    labels = labels[vis_idx]

    # 颜色映射
    colors = np.array(['blue' if label == 0 else 'red' for label in labels])
    
    # 生成学生模型图
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.scatter(student_2d[:, 0], student_2d[:, 1], c=colors, alpha=0.5, s=5, edgecolors='none')
    
    # 标题（可选）
    if SHOW_TITLE:
        if SHOW_SCORE_IN_TITLE:
            title = f'Student Model (Optical+SAR)\nSeparability Score: {student_score:.4f}'
        else:
            title = 'Student Model (Optical+SAR)'
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 坐标轴标签（可选）
    if SHOW_AXIS_LABELS:
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # 网格（可选）
    if SHOW_GRID:
        ax.grid(True, alpha=0.3)
    
    # 图例（可选）
    if SHOW_LEGEND:
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Changed'),
            Patch(facecolor='blue', alpha=0.5, label='Unchanged')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=11)
    
    plt.tight_layout()
    student_path = save_path.replace('_tsne.png', '_student.png')
    plt.savefig(student_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成教师模型图
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.scatter(teacher_2d[:, 0], teacher_2d[:, 1], c=colors, alpha=0.5, s=5, edgecolors='none')
    
    # 标题（可选）
    if SHOW_TITLE:
        if SHOW_SCORE_IN_TITLE:
            title = f'Teacher Model (Optical+Optical)\nSeparability Score: {teacher_score:.4f}'
        else:
            title = 'Teacher Model (Optical+Optical)'
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 坐标轴标签（可选）
    if SHOW_AXIS_LABELS:
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # 网格（可选）
    if SHOW_GRID:
        ax.grid(True, alpha=0.3)
    
    # 图例（可选）
    if SHOW_LEGEND:
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Changed'),
            Patch(facecolor='blue', alpha=0.5, label='Unchanged')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=11)
    
    plt.tight_layout()
    teacher_path = save_path.replace('_tsne.png', '_teacher.png')
    plt.savefig(teacher_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_all_images(model, dataset, output_dir, device='cuda', 
                       top_n=10, min_samples=10, perplexity=30):
    """
    处理所有测试图像，生成t-SNE图并选出可分性最好的top-n
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print(f"\nProcessing {len(dataset)} test images...")
    
    for idx in tqdm(range(len(dataset)), desc="Processing"):
        img_A, img_B, img_C, label, img_name = dataset[idx]
        
        # 提取像素级特征
        student_feat, teacher_feat, pixel_labels = extract_pixel_features(
            model, img_A, img_B, img_C, label, device
        )
        
        # 检查每个类别的样本数
        unique, counts = np.unique(pixel_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # 确保两个类别都有足够的样本
        if len(unique) < 2 or min(counts) < min_samples:
            print(f"\nSkipping {img_name}: insufficient samples (class 0: {class_counts.get(0, 0)}, class 1: {class_counts.get(1, 0)})")
            continue
        
        # 为了加速：先限制排序阶段采样像素点数
        if len(pixel_labels) > RANK_SAMPLE:
            # 分层采样，保持类别比例
            indices_0 = np.where(pixel_labels == 0)[0]
            indices_1 = np.where(pixel_labels == 1)[0]

            n0 = int(RANK_SAMPLE * len(indices_0) / len(pixel_labels))
            n1 = RANK_SAMPLE - n0

            sampled_0 = np.random.choice(indices_0, min(n0, len(indices_0)), replace=False)
            sampled_1 = np.random.choice(indices_1, min(n1, len(indices_1)), replace=False)
            
            sampled_indices = np.concatenate([sampled_0, sampled_1])
            np.random.shuffle(sampled_indices)
            
            student_feat = student_feat[sampled_indices]
            teacher_feat = teacher_feat[sampled_indices]
            pixel_labels = pixel_labels[sampled_indices]
        
        # 进行 PCA 预降维 + t-SNE 降维并计算可分性（显著加速）
        try:
            # PCA 预降维
            if student_feat.shape[1] > PCA_DIM:
                pca = PCA(n_components=PCA_DIM, random_state=TSNE_RANDOM_STATE)
                student_low = pca.fit_transform(student_feat)
                teacher_low = pca.fit_transform(teacher_feat)
            else:
                student_low = student_feat
                teacher_low = teacher_feat

            eff_perplexity = max(5, min(perplexity, max(5, len(pixel_labels)//3)))

            if JOINT_TSNE_FOR_PLOT:
                # 为提高视觉对比一致性：将师生特征拼接后做一次t-SNE，再拆分
                joint = np.concatenate([student_low, teacher_low], axis=0)
                tsne = TSNE(n_components=2, perplexity=eff_perplexity, 
                           random_state=TSNE_RANDOM_STATE, max_iter=TSNE_MAX_ITER, n_jobs=-1)
                joint_2d = tsne.fit_transform(joint)
                student_2d = joint_2d[:len(student_low)]
                teacher_2d = joint_2d[len(student_low):]
            else:
                tsne = TSNE(n_components=2, perplexity=eff_perplexity, 
                           random_state=TSNE_RANDOM_STATE, max_iter=TSNE_MAX_ITER, n_jobs=-1)
                student_2d = tsne.fit_transform(student_low)
                tsne = TSNE(n_components=2, perplexity=eff_perplexity, 
                           random_state=TSNE_RANDOM_STATE, max_iter=TSNE_MAX_ITER, n_jobs=-1)
                teacher_2d = tsne.fit_transform(teacher_low)

            # 可分性依然各自独立评估（不受拼接影响）
            student_score = calculate_separability(student_2d, pixel_labels)
            teacher_score = calculate_separability(teacher_2d, pixel_labels)
            
            # 计算平均可分性（师生平均）
            avg_score = (student_score + teacher_score) / 2
            
            # 保存结果
            img_base = os.path.splitext(img_name)[0]
            save_path = os.path.join(output_dir, f'{img_base}_tsne.png')
            
            # 保存2D降维结果用于可视化（先快速绘图）
            try:
                visualize_single_image_tsne(
                    student_2d, teacher_2d,
                    pixel_labels,
                    img_name, student_score, teacher_score, save_path,
                    perplexity=eff_perplexity, random_state=TSNE_RANDOM_STATE
                )
            except Exception:
                pass
            
            results.append({
                'img_name': img_name,
                'student_score': student_score,
                'teacher_score': teacher_score,
                'avg_score': avg_score,
                'save_path': save_path,
                'n_pixels': len(pixel_labels),
                'n_change': np.sum(pixel_labels == 1),
                'n_unchange': np.sum(pixel_labels == 0)
            })
            
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            continue
    
    if len(results) == 0:
        print("\nNo images processed successfully!")
        return
    
    # 按平均可分性分数排序
    results.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # 仅对Top-N图像重绘高质量图（用全部像素 + 完整流程）
    print(f"\nGenerating high-quality plots for top-{top_n} images...")
    for result in results[:top_n]:
        img_name = result['img_name']
        # 重新加载该图完整像素
        img_A_path = os.path.join(dataset.img_A_dir, img_name)
        img_B_path = os.path.join(dataset.img_B_dir, img_name)
        img_C_path = os.path.join(dataset.img_C_dir, img_name)
        label_path = os.path.join(dataset.test_dir, 'label', img_name)

        img_A = torch.from_numpy(io.imread(img_A_path)).permute(2, 0, 1).float() / 255.0
        img_B = torch.from_numpy(io.imread(img_B_path)).permute(2, 0, 1).float() / 255.0
        img_C = torch.from_numpy(io.imread(img_C_path)).permute(2, 0, 1).float() / 255.0
        label = (io.imread(label_path) > 127).astype(np.uint8)

        stu_full, tea_full, pix_labels = extract_pixel_features(model, img_A, img_B, img_C, label, device)

        # 为避免极端慢，进行合理上限采样
        if len(pix_labels) > MAX_PIXELS:
            idx0 = np.where(pix_labels == 0)[0]
            idx1 = np.where(pix_labels == 1)[0]
            n0 = min(len(idx0), MAX_PIXELS // 2)
            n1 = min(len(idx1), MAX_PIXELS - n0)
            sel = np.concatenate([
                np.random.choice(idx0, n0, replace=False) if n0 > 0 else np.array([], dtype=int),
                np.random.choice(idx1, n1, replace=False) if n1 > 0 else np.array([], dtype=int)
            ])
            stu_full = stu_full[sel]
            tea_full = tea_full[sel]
            pix_labels = pix_labels[sel]

        # PCA 预降维
        if stu_full.shape[1] > PCA_DIM:
            pca = PCA(n_components=PCA_DIM, random_state=TSNE_RANDOM_STATE)
            stu_low = pca.fit_transform(stu_full)
            tea_low = pca.fit_transform(tea_full)
        else:
            stu_low = stu_full
            tea_low = tea_full

        eff_perp = max(5, min(perplexity, max(5, len(pix_labels)//3)))

        if JOINT_TSNE_FOR_PLOT:
            joint = np.concatenate([stu_low, tea_low], axis=0)
            tsne = TSNE(n_components=2, perplexity=eff_perp, random_state=TSNE_RANDOM_STATE, max_iter=TSNE_MAX_ITER, n_jobs=-1)
            joint_2d = tsne.fit_transform(joint)
            s2d = joint_2d[:len(stu_low)]
            t2d = joint_2d[len(stu_low):]
        else:
            tsne = TSNE(n_components=2, perplexity=eff_perp, random_state=TSNE_RANDOM_STATE, max_iter=TSNE_MAX_ITER, n_jobs=-1)
            s2d = tsne.fit_transform(stu_low)
            tsne = TSNE(n_components=2, perplexity=eff_perp, random_state=TSNE_RANDOM_STATE, max_iter=TSNE_MAX_ITER, n_jobs=-1)
            t2d = tsne.fit_transform(tea_low)

        s_score = calculate_separability(s2d, pix_labels)
        t_score = calculate_separability(t2d, pix_labels)

        # 覆盖保存高质量图
        visualize_single_image_tsne(
            s2d, t2d, pix_labels,
            img_name, s_score, t_score, result['save_path'],
            perplexity=eff_perp, random_state=TSNE_RANDOM_STATE
        )

    # 保存所有结果到文件
    results_file = os.path.join(output_dir, 'separability_scores.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Image Separability Analysis Results (sorted by average score)\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Rank':<6} {'Image Name':<30} {'Student':<12} {'Teacher':<12} {'Average':<12} {'Changed':<12} {'Unchanged':<12}\n")
        f.write("-"*100 + "\n")
        
        for rank, result in enumerate(results, 1):
            f.write(f"{rank:<6} {result['img_name']:<30} {result['student_score']:>11.4f} "
                   f"{result['teacher_score']:>11.4f} {result['avg_score']:>11.4f} "
                   f"{result['n_change']:>11} {result['n_unchange']:>11}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write(f"\nTop-{top_n} Best Separability Images (for paper):\n")
        f.write("-"*100 + "\n")
        
        for rank, result in enumerate(results[:top_n], 1):
            f.write(f"{rank}. {result['img_name']}\n")
            f.write(f"   Student Model Separability: {result['student_score']:.4f}\n")
            f.write(f"   Teacher Model Separability: {result['teacher_score']:.4f}\n")
            f.write(f"   Average Separability: {result['avg_score']:.4f}\n")
            f.write(f"   File Path: {result['save_path']}\n\n")
    
    print(f"\n{'='*100}")
    print(f"Processing complete! Successfully processed {len(results)} images")
    print(f"{'='*100}")
    print(f"\nTop-{top_n} Best Separability Images:")
    print(f"{'-'*100}")
    
    for rank, result in enumerate(results[:top_n], 1):
        print(f"{rank}. {result['img_name']}")
        print(f"   Student: {result['student_score']:.4f}, Teacher: {result['teacher_score']:.4f}, "
              f"Average: {result['avg_score']:.4f}")
    
    print(f"\nDetailed results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Per-image t-SNE separability analysis')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT, help='Checkpoint path or directory')
    parser.add_argument('--test_dir', type=str, default=TEST_DIR, help='Test data directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device')
    parser.add_argument('--top_n', type=int, default=TOP_N, help='Number of top images to save')
    parser.add_argument('--min_samples', type=int, default=MIN_SAMPLES_PER_CLASS, 
                       help='Minimum samples per class')
    parser.add_argument('--perplexity', type=int, default=TSNE_PERPLEXITY, help='t-SNE perplexity')
    
    # 图表显示选项
    parser.add_argument('--no-title', action='store_true', help='Hide title')
    parser.add_argument('--no-score', action='store_true', help='Hide separability score in title')
    parser.add_argument('--no-axis-labels', action='store_true', help='Hide axis labels')
    parser.add_argument('--no-legend', action='store_true', help='Hide legend')
    parser.add_argument('--no-grid', action='store_true', help='Hide grid')
    
    args = parser.parse_args()
    
    # 更新全局显示配置
    global SHOW_TITLE, SHOW_SCORE_IN_TITLE, SHOW_AXIS_LABELS, SHOW_LEGEND, SHOW_GRID
    if args.no_title:
        SHOW_TITLE = False
    if args.no_score:
        SHOW_SCORE_IN_TITLE = False
    if args.no_axis_labels:
        SHOW_AXIS_LABELS = False
    if args.no_legend:
        SHOW_LEGEND = False
    if args.no_grid:
        SHOW_GRID = False
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("\n" + "="*70)
    print("Loading Model")
    print("="*70)
    model = load_model(args.checkpoint, device)
    
    # 创建数据集
    print("\n" + "="*70)
    print("Loading Dataset")
    print("="*70)
    dataset = SingleImageDataset(args.test_dir)
    
    # 处理所有图像
    print("\n" + "="*70)
    print("Processing Images and Analyzing Separability")
    print("="*70)
    process_all_images(
        model, dataset, args.output_dir, device,
        top_n=args.top_n,
        min_samples=args.min_samples,
        perplexity=args.perplexity
    )
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
