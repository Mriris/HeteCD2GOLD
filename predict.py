import os
import time
import argparse
import numpy as np
import torch
from skimage import io, exposure
from torch.nn import functional as F
from torchvision.transforms import functional as FF
from torch.utils.data import DataLoader
from datasets import RS_ST as RS
from models.hetecd import hetecd as Net
import cv2
from scipy.spatial.distance import euclidean
from scipy.stats import entropy, wasserstein_distance
from scipy.special import rel_entr
import seaborn as sns
from tqdm import tqdm
from utils.utils import accuracy, SCDD_eval_all, AverageMeter, get_confuse_matrix, cm2score
DATA_NAME = 'ST'
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 输入：两个BCHW张量
def visualize_features(tensor1, tensor2, output_filename, sample_size=10000):
    """
    可视化两组特征的直方图和KDE，并保存图像到指定文件。

    参数:
        tensor1 (torch.Tensor): 第一组特征，形状为 (B, C, H, W)。
        tensor2 (torch.Tensor): 第二组特征，形状为 (B, C, H, W)。
        output_filename (str): 图像保存的文件名（包含路径）。
        sample_size (int): 随机采样的大小，默认10000。
    """
    # 将特征展平成 1D 数组
    flattened_features1 = tensor1.flatten().cpu().numpy()
    flattened_features2 = tensor2.flatten().cpu().numpy()

    # 随机采样来加快计算
    sampled_features1 = np.random.choice(flattened_features1, size=sample_size, replace=False)
    sampled_features2 = np.random.choice(flattened_features2, size=sample_size, replace=False)

    # 创建一个2x2的绘图窗口
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 绘制 tensor1 的直方图
    sns.histplot(sampled_features1, bins=50, color='blue', ax=axs[0, 0])
    axs[0, 0].set_title('Histogram of Features1')
    axs[0, 0].set_xlabel('Feature Values')
    axs[0, 0].set_ylabel('Count')

    # 绘制 tensor2 的直方图
    sns.histplot(sampled_features2, bins=50, color='red', ax=axs[0, 1])
    axs[0, 1].set_title('Histogram of Features2')
    axs[0, 1].set_xlabel('Feature Values')
    axs[0, 1].set_ylabel('Count')

    # 绘制 tensor1 的 KDE
    sns.kdeplot(sampled_features1, color='blue', ax=axs[1, 0])
    axs[1, 0].set_title('KDE of Features1')
    axs[1, 0].set_xlabel('Feature Values')
    axs[1, 0].set_ylabel('Density')

    # 绘制 tensor2 的 KDE
    sns.kdeplot(sampled_features2, color='red', ax=axs[1, 1])
    axs[1, 1].set_title('KDE of Features2')
    axs[1, 1].set_xlabel('Feature Values')
    axs[1, 1].set_ylabel('Density')

    # 调整图像布局
    plt.tight_layout()

    # 保存图像到文件
    plt.savefig(output_filename)
    plt.close()

# # 示例：创建两个随机的BCHW张量
# tensor1 = torch.rand(4, 64, 128, 128)
# tensor2 = torch.rand(4, 64, 128, 128)

# # 调用函数对比特征分布


def compute_feature_distances(tensor1, tensor2):
    # 检查输入的形状是否相同
    if tensor1.shape != tensor2.shape:
        raise ValueError("输入的两个tensor形状必须相同")

    # 将tensor展平为 [B, C * H * W] 的形状
    B = tensor1.size(0)  # batch size
    tensor1_flat = tensor1.view(B, -1).cpu().numpy()
    tensor2_flat = tensor2.view(B, -1).cpu().numpy()

    # 1. 欧氏距离 (Euclidean Distance)
    euclidean_distances = np.array([euclidean(t1, t2) for t1, t2 in zip(tensor1_flat, tensor2_flat)])
    euclidean_mean = np.mean(euclidean_distances)

    # 2. KL散度 (Kullback-Leibler Divergence)
    tensor1_prob = torch.nn.functional.softmax(torch.tensor(tensor1_flat), dim=-1).numpy()
    tensor2_prob = torch.nn.functional.softmax(torch.tensor(tensor2_flat), dim=-1).numpy()

    kl_divergence = np.array([np.sum(rel_entr(t1, t2)) for t1, t2 in zip(tensor1_prob, tensor2_prob)])
    kl_mean = np.mean(kl_divergence)

    # 3. JS散度 (Jensen-Shannon Divergence)
    def jensen_shannon_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m) + entropy(q, m))

    js_divergence = np.array([jensen_shannon_divergence(t1, t2) for t1, t2 in zip(tensor1_prob, tensor2_prob)])
    js_mean = np.mean(js_divergence)

    # 4. Wasserstein 距离 (Wasserstein Distance)
    wasserstein_distances = np.array([wasserstein_distance(t1, t2) for t1, t2 in zip(tensor1_flat, tensor2_flat)])
    wasserstein_mean = np.mean(wasserstein_distances)

    # 5. Hellinger 距离 (Hellinger Distance)
    def hellinger_distance(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    hellinger_distances = np.array([hellinger_distance(t1, t2) for t1, t2 in zip(tensor1_prob, tensor2_prob)])
    hellinger_mean = np.mean(hellinger_distances)

    # 返回结果
    results = {
        "Euclidean Distance": euclidean_mean,
        "KL Divergence": kl_mean,
        "JS Divergence": js_mean,
        "Wasserstein Distance": wasserstein_mean,
        "Hellinger Distance": hellinger_mean
    }

    return results


def create_error_visualization(pred, label):
    """
    创建误差可视化图像
    
    参数:
        pred: 预测结果 (H, W)，值为 0 或 1
        label: 真值标签 (H, W)，值为 0 或 1
        
    返回:
        可视化图像 (H, W, 3)，BGR格式
        - 白色: 正确预测的变化区域
        - 红色: 假阳性（预测为1，实际为0）
        - 绿色: 假阴性（预测为0，实际为1）
        - 黑色: 正确的背景
    """
    h, w = pred.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 正确的变化区域（白色）
    correct_change = (pred == 1) & (label == 1)
    vis[correct_change] = [255, 255, 255]
    
    # 假阳性（红色）- 预测为变化但实际未变化
    false_positive = (pred == 1) & (label == 0)
    vis[false_positive] = [0, 0, 255]  # BGR: 红色
    
    # 假阴性（绿色）- 预测为未变化但实际变化
    false_negative = (pred == 0) & (label == 1)
    vis[false_negative] = [0, 255, 0]  # BGR: 绿色
    
    # 正确的背景保持黑色
    
    return vis


class PredOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', type=int, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', type=str, default='/data/jingwei/yantingxuan/Datasets/CityCN/Split45/test', help='directory to test images')
        parser.add_argument('--pred_dir', type=str, default='/data/jingwei/yantingxuan/0Program/HeteCD2GOLD/results0', help='directory to output masks')
        parser.add_argument('--chkpt_path', type=str, default='/data/jingwei/yantingxuan/0Program/HeteCD2GOLD/checkpoints/HeteCD/gold43/EXP20250920171943/HeteCD_375IoU59.62', help='path to checkpoint')
        parser.add_argument('--save_error', action='store_true', default=True, help='save error visualization')
        parser.add_argument('--input_size', type=int, default=512, help='input size for FLOPs calculation')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def count_parameters(model):
    """
    计算模型参数量
    
    参数:
        model: PyTorch 模型
        
    返回:
        参数量（单位：百万）
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6


def calculate_flops(model, input_shape=(1, 3, 512, 512), device='cuda'):
    """
    计算模型 FLOPs
    
    参数:
        model: PyTorch 模型
        input_shape: 输入张量形状 (B, C, H, W)
        device: 运行设备
        
    返回:
        FLOPs（单位：十亿）
    """
    try:
        from thop import profile
        # 创建两个输入（img_A, img_B）
        inputs = (
            torch.randn(input_shape).to(device),
            torch.randn(input_shape).to(device)
        )
        flops, params = profile(model, inputs=inputs, verbose=False)
        return flops / 1e9  # 转换为 G
    except ImportError:
        print("警告: 未安装 thop 库，无法计算 FLOPs")
        print("请运行: pip install thop")
        return -1
    except Exception as e:
        print(f"警告: 计算 FLOPs 时出错: {e}")
        return -1


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net(3).cuda()
    checkpoint = torch.load(opt.chkpt_path, map_location='cuda:0')
    #打印checkpoint中的层名
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
    
    net.load_state_dict(new_state_dict,strict=False)

    net.eval()

    # 计算模型复杂度
    print("\n" + "="*70)
    print("模型复杂度分析")
    print("="*70)
    
    total_params, trainable_params = count_parameters(net)
    print(f"总参数量: {total_params:.2f}M")
    print(f"可训练参数量: {trainable_params:.2f}M")
    
    print("\n计算 FLOPs...")
    flops = calculate_flops(net, input_shape=(1, 3, opt.input_size, opt.input_size), device='cuda')
    if flops > 0:
        print(f"FLOPs: {flops:.2f}G (输入尺寸: {opt.input_size}x{opt.input_size})")
    
    # 执行预测
    preds_all, labels_all = predict(net, opt.test_dir, opt.pred_dir, save_error=opt.save_error)
    
    # 计算评估指标
    print("\n" + "="*70)
    print("评估指标")
    print("="*70)
    
    hist = get_confuse_matrix(2, labels_all, preds_all)
    metrics = cm2score(hist)
    
    # 提取变化类指标
    precision = metrics['precision_1'] * 100  # 变化类精确率
    recall = metrics['recall_1'] * 100  # 变化类召回率
    f1 = metrics['F1_1'] * 100  # 变化类F1
    iou = metrics['iou_1'] * 100  # 变化类IoU
    
    # 打印表格
    print("\n综合性能指标表:")
    print("-" * 100)
    print(f"{'Method':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'IoU':<10} | {'Params(M)':<12} | {'FLOPs(G)':<12}")
    print("-" * 100)
    
    flops_str = f"{flops:.2f}" if flops > 0 else "N/A"
    
    print(f"{'HeteCD':<15} | {precision:>9.2f}% | {recall:>9.2f}% | {f1:>9.2f}% | {iou:>9.2f}% | {total_params:>11.2f} | {flops_str:>11}")
    print("-" * 100)
    
    # 打印详细指标
    print(f"\n详细指标:")
    print(f"  整体准确率 (Accuracy): {metrics['acc']*100:.2f}%")
    print(f"  平均IoU (mIoU): {metrics['miou']*100:.2f}%")
    print(f"  平均F1分数 (mF1): {metrics['mf1']*100:.2f}%")
    
    print(f"\n  类别IoU:")
    print(f"    背景 (IoU_0): {metrics['iou_0']*100:.2f}%")
    print(f"    变化 (IoU_1): {metrics['iou_1']*100:.2f}%")
    
    print(f"\n  类别F1:")
    print(f"    背景 (F1_0): {metrics['F1_0']*100:.2f}%")
    print(f"    变化 (F1_1): {metrics['F1_1']*100:.2f}%")
    
    print(f"\n  精确率和召回率:")
    print(f"    变化类精确率: {metrics['precision_1']*100:.2f}%")
    print(f"    变化类召回率: {metrics['recall_1']*100:.2f}%")
    
    time_use = time.time() - begin_time
    print(f'\n总耗时: {time_use:.2f}s')
    print(f"结果已保存至: {opt.pred_dir}")


def load_images_from_folder(folder):
    images_A = []
    images_B = []
    labels = []
    filenames = sorted(os.listdir(os.path.join(folder, 'A')))
    for filename in filenames:
        img_A = io.imread(os.path.join(folder, 'A', filename))
        img_B = io.imread(os.path.join(folder, 'B', filename))
        label = io.imread(os.path.join(folder, 'label', filename))
        if img_A is not None and img_B is not None:
            images_A.append(img_A)
            images_B.append(img_B)
            labels.append(label)
    return images_A, images_B, labels, filenames


def predict(net, test_dir, pred_dir, save_error=True):
    images_A, images_B, labels, filenames = load_images_from_folder(test_dir)
    preds_all = []
    labels_all = []
    
    for img_A, img_B, label, filename in tqdm(zip(images_A, images_B, labels, filenames), 
                                                total=len(filenames), 
                                                desc='预测进度'):
        img_A = FF.to_tensor(img_A).cuda().float().unsqueeze(0)
        img_B = FF.to_tensor(img_B).cuda().float().unsqueeze(0)
        
        with torch.no_grad():
            # for i in range(100):
            #     t1 = time.time()
            #     out_change, features = net(img_A, img_B)
            out_change, features = net(img_A, img_B)
            
            preds = torch.argmax(out_change, dim=1)
            pred_numpy = preds[0].cpu().numpy().astype(np.uint8)
            # pred_numpy = preds.cpu().numpy()
            labels_numpy = label[np.newaxis, ...]
            #print(np.unique(labels_numpy))
            preds_all.append(pred_numpy[np.newaxis, ...])
            labels_all.append(labels_numpy)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        
        # 保存预测结果
        save_path = os.path.join(pred_dir, filename)
        cv2.imwrite(save_path, pred_numpy * 255)
        
        # 保存误差可视化图
        if save_error:
            # 标准化标签为0/1二值
            label_binary = (label > 127).astype(np.uint8)
            # 生成误差图
            error_vis = create_error_visualization(pred_numpy, label_binary)
            # 保存误差图
            error_filename = os.path.splitext(filename)[0] + '_error.png'
            error_path = os.path.join(pred_dir, error_filename)
            cv2.imwrite(error_path, error_vis)
    
    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)//255
    
    return preds_all, labels_all


if __name__ == '__main__':
    main()