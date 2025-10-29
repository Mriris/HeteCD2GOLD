import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm
import warnings
from models.hetecd import hetecd as Net
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 抑制警告
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ===== 顶部参数配置 =====
CHECKPOINT = r'checkpoints\HeteCD\trios43\HeteCD_375IoU59.62'
TEST_DIR = r'D:\0Program\Datasets\241120\Compare\Datas\tsne'
OUTPUT_DIR = r'results/tsne_baseline'
DEVICE = 'cuda'
NUM_WORKERS = 0

# t-SNE 与加速参数
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_MAX_ITER = 500
PCA_DIM = 50
MAX_PIXELS = 2000
RANK_SAMPLE = 3000
SILHOUETTE_SAMPLE = 1000

# 可视化与筛选参数
TOP_N = 10
MIN_SAMPLES_PER_CLASS = 10

# 绘图配置
BALANCE_PLOT = True
NEG_POS_RATIO = 3
PLOT_POS_MAX = 800

# 图表元素显示配置
SHOW_TITLE = True
SHOW_SCORE_IN_TITLE = False
SHOW_AXIS_LABELS = True
SHOW_LEGEND = True
SHOW_GRID = True


class FeatureExtractor:
    """使用 forward hook 提取中间特征"""
    def __init__(self):
        self.features = None
    
    def hook(self, module, input, output):
        """Hook 函数，保存模块输出"""
        self.features = output.detach()


class BaselineDataset(Dataset):
    """Baseline数据集，只加载A和B"""
    def __init__(self, test_dir):
        self.test_dir = test_dir
        
        # 读取图像路径
        img_A_dir = os.path.join(test_dir, 'A')
        img_B_dir = os.path.join(test_dir, 'B')
        
        if not os.path.exists(img_A_dir):
            raise ValueError(f"Dataset directory A does not exist: {img_A_dir}")
        if not os.path.exists(img_B_dir):
            raise ValueError(f"Dataset directory B does not exist: {img_B_dir}")
        
        self.img_A_dir = img_A_dir
        self.img_B_dir = img_B_dir
        
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
        
        # 读取 A、B 两个图像
        img_A_path = os.path.join(self.img_A_dir, img_name)
        img_B_path = os.path.join(self.img_B_dir, img_name)
        
        img_A = io.imread(img_A_path)
        img_B = io.imread(img_B_path)
        
        # 转换为张量
        img_A = torch.from_numpy(img_A).permute(2, 0, 1).float() / 255.0
        img_B = torch.from_numpy(img_B).permute(2, 0, 1).float() / 255.0
        
        # 读取标签
        label_path = os.path.join(self.test_dir, 'label', img_name)
        label = io.imread(label_path)
        label = (label > 127).astype(np.uint8)
        
        return img_A, img_B, label, img_name


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
    """Load baseline model weights"""
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
    
    model.eval()
    
    return model


def extract_pixel_features(model, img_A, img_B, label, device='cuda'):
    """
    提取单张图像所有像素点的特征
    
    Returns:
        features: [H*W, C] 模型特征
        pixel_labels: [H*W] 每个像素的标签
    """
    model.eval()
    
    # 创建特征提取器
    extractor = FeatureExtractor()
    
    # 注册 hook (baseline只有学生解码器)
    hook = model.CD_Decoder.dense_1x.register_forward_hook(extractor.hook)
    
    # 添加batch维度
    img_A = img_A.unsqueeze(0).to(device)
    img_B = img_B.unsqueeze(0).to(device)
    
    with torch.no_grad():
        _ = model(img_A, img_B)
    
    # 提取特征 [1, C, H, W]
    features = extractor.features  # [1, C, H, W]
    
    # 移除 batch 维度并转置为 [H*W, C]
    features = features.squeeze(0).permute(1, 2, 0).contiguous()  # [H, W, C]
    
    H, W, C = features.shape
    
    features = features.view(-1, C).cpu().numpy()  # [H*W, C]
    
    # 调整label大小以匹配特征图
    label_resized = F.interpolate(
        torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float(),
        size=(H, W),
        mode='nearest'
    ).squeeze().numpy().astype(np.uint8)
    
    pixel_labels = label_resized.reshape(-1)  # [H*W]
    
    # 移除 hook
    hook.remove()
    
    return features, pixel_labels


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


def visualize_tsne(features_2d, labels, img_name, score, save_path):
    """
    生成baseline模型的t-SNE图
    
    Args:
        features_2d: [N, 2] 2D特征
        labels: [N] 像素标签
        img_name: 图像名称
        score: 可分性分数
        save_path: 保存路径
    """
    # 可视化前类别均衡抽样
    vis_idx = np.arange(len(labels))
    if BALANCE_PLOT:
        idx_pos = np.where(labels == 1)[0]
        idx_neg = np.where(labels == 0)[0]
        if len(idx_pos) > 0 and len(idx_neg) > 0:
            n_pos = min(len(idx_pos), PLOT_POS_MAX)
            sel_pos = np.random.choice(idx_pos, n_pos, replace=False)
            n_neg = min(len(idx_neg), n_pos * NEG_POS_RATIO)
            sel_neg = np.random.choice(idx_neg, n_neg, replace=False)
            vis_idx = np.concatenate([sel_pos, sel_neg])
            np.random.shuffle(vis_idx)

    features_2d = features_2d[vis_idx]
    labels = labels[vis_idx]

    # 颜色映射
    colors = np.array(['blue' if label == 0 else 'red' for label in labels])
    
    # 生成图
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.5, s=5, edgecolors='none')
    
    # 标题（可选）
    if SHOW_TITLE:
        if SHOW_SCORE_IN_TITLE:
            title = f'Baseline Model (Optical+SAR)\nSeparability Score: {score:.4f}'
        else:
            title = 'Baseline Model (Optical+SAR)'
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        img_A, img_B, label, img_name = dataset[idx]
        
        # 提取像素级特征
        features, pixel_labels = extract_pixel_features(
            model, img_A, img_B, label, device
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
            
            features = features[sampled_indices]
            pixel_labels = pixel_labels[sampled_indices]
        
        # 进行 PCA 预降维 + t-SNE 降维并计算可分性
        try:
            # PCA 预降维
            if features.shape[1] > PCA_DIM:
                pca = PCA(n_components=PCA_DIM, random_state=TSNE_RANDOM_STATE)
                features_low = pca.fit_transform(features)
            else:
                features_low = features

            eff_perplexity = max(5, min(perplexity, max(5, len(pixel_labels)//3)))

            tsne = TSNE(n_components=2, perplexity=eff_perplexity, 
                       random_state=TSNE_RANDOM_STATE, max_iter=TSNE_MAX_ITER, n_jobs=-1)
            features_2d = tsne.fit_transform(features_low)
            score = calculate_separability(features_2d, pixel_labels)
            
            # 保存结果
            img_base = os.path.splitext(img_name)[0]
            save_path = os.path.join(output_dir, f'{img_base}_baseline.png')
            
            # 保存2D降维结果用于可视化
            try:
                visualize_tsne(features_2d, pixel_labels, img_name, score, save_path)
            except Exception:
                pass
            
            results.append({
                'img_name': img_name,
                'score': score,
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
    
    # 按可分性分数排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 仅对Top-N图像重绘高质量图
    print(f"\nGenerating high-quality plots for top-{top_n} images...")
    for result in results[:top_n]:
        img_name = result['img_name']
        # 重新加载该图完整像素
        img_A_path = os.path.join(dataset.img_A_dir, img_name)
        img_B_path = os.path.join(dataset.img_B_dir, img_name)
        label_path = os.path.join(dataset.test_dir, 'label', img_name)

        img_A = torch.from_numpy(io.imread(img_A_path)).permute(2, 0, 1).float() / 255.0
        img_B = torch.from_numpy(io.imread(img_B_path)).permute(2, 0, 1).float() / 255.0
        label = (io.imread(label_path) > 127).astype(np.uint8)

        feat_full, pix_labels = extract_pixel_features(model, img_A, img_B, label, device)

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
            feat_full = feat_full[sel]
            pix_labels = pix_labels[sel]

        # PCA 预降维
        if feat_full.shape[1] > PCA_DIM:
            pca = PCA(n_components=PCA_DIM, random_state=TSNE_RANDOM_STATE)
            feat_low = pca.fit_transform(feat_full)
        else:
            feat_low = feat_full

        eff_perp = max(5, min(perplexity, max(5, len(pix_labels)//3)))

        tsne = TSNE(n_components=2, perplexity=eff_perp, random_state=TSNE_RANDOM_STATE, 
                   max_iter=TSNE_MAX_ITER, n_jobs=-1)
        feat_2d = tsne.fit_transform(feat_low)
        new_score = calculate_separability(feat_2d, pix_labels)

        # 覆盖保存高质量图
        visualize_tsne(feat_2d, pix_labels, img_name, new_score, result['save_path'])

    # 保存所有结果到文件
    results_file = os.path.join(output_dir, 'separability_scores.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Baseline Model - Image Separability Analysis Results (sorted by score)\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Rank':<6} {'Image Name':<30} {'Score':<12} {'Changed':<12} {'Unchanged':<12}\n")
        f.write("-"*100 + "\n")
        
        for rank, result in enumerate(results, 1):
            f.write(f"{rank:<6} {result['img_name']:<30} {result['score']:>11.4f} "
                   f"{result['n_change']:>11} {result['n_unchange']:>11}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write(f"\nTop-{top_n} Best Separability Images (for paper):\n")
        f.write("-"*100 + "\n")
        
        for rank, result in enumerate(results[:top_n], 1):
            f.write(f"{rank}. {result['img_name']}\n")
            f.write(f"   Separability Score: {result['score']:.4f}\n")
            f.write(f"   File Path: {result['save_path']}\n\n")
    
    print(f"\n{'='*100}")
    print(f"Processing complete! Successfully processed {len(results)} images")
    print(f"{'='*100}")
    print(f"\nTop-{top_n} Best Separability Images:")
    print(f"{'-'*100}")
    
    for rank, result in enumerate(results[:top_n], 1):
        print(f"{rank}. {result['img_name']}")
        print(f"   Score: {result['score']:.4f}")
    
    print(f"\nDetailed results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Baseline model per-image t-SNE separability analysis')
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
    print("Loading Baseline Model")
    print("="*70)
    model = load_model(args.checkpoint, device)
    
    # 创建数据集
    print("\n" + "="*70)
    print("Loading Dataset")
    print("="*70)
    dataset = BaselineDataset(args.test_dir)
    
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

