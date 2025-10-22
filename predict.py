import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage import io
from PIL import Image
from tqdm import tqdm
import cv2
import warnings
import time
from models.hetecd import hetecd as Net
from utils.utils import get_confuse_matrix, cm2score

# 抑制 skimage 的低对比度图像警告
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# 设备配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ===== 顶部参数配置（可直接在此处修改） =====
# 说明：
# - 若不传命令行参数，将使用这里的默认值
# - 命令行参数依然可覆盖这些默认值
CHECKPOINT = r'/data/jingwei/yantingxuan/0Program/HeteCD2GOLD/checkpoints/gold/trios45/EXP20251021225208MSE+DA|MultiImgPhotoMetric|PolyLR/gold_teacher_398IoU66.24.pth'
TEST_DIR = r'/data/jingwei/yantingxuan/Datasets/CityCN/Split45/val'
OUTPUT_DIR = r'/data/jingwei/yantingxuan/0Program/HeteCD2GOLD/results'
BATCH_SIZE = 4
DEVICE = 'cuda'
INPUT_SIZE = 512  # 用于 FLOPs 计算的输入尺寸

# 可视化输出控制（仅在有 label 时生效）
SAVE_PRED = False   # 是否保存预测结果 (_pred.png)
SAVE_LABEL = False  # 是否保存真值标签 (_label.png)
SAVE_ERROR = True  # 是否保存误差可视化 (_error.png)

# 推理速度测量为可选步骤
MEASURE_SPEED = False  # 是否测量推理速度（默认关闭）
SPEED_WARMUP = 10      # 推理速度预热次数
SPEED_ITERS = 100      # 推理速度测量迭代次数

# DataLoader 相关
NUM_WORKERS = 4
PIN_MEMORY = True


def normalize_state_dict(state_dict):
    """
    规范化权重键名，自动剥离 DataParallel 和 torch.compile 添加的前缀
    
    Args:
        state_dict: 原始权重字典
        
    Returns:
        规范化后的权重字典
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # 移除可能的前缀组合
        new_key = key
        
        # 处理各种可能的前缀
        prefixes_to_remove = [
            '_orig_mod.module.',  # torch.compile + DataParallel
            'module._orig_mod.',  # 另一种可能的顺序
            '_orig_mod.',         # 仅 torch.compile
            'module.',            # 仅 DataParallel
        ]
        
        for prefix in prefixes_to_remove:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        
        new_state_dict[new_key] = value
    
    return new_state_dict


class TestDataset(Dataset):
    """
    测试数据集类，支持学生网络（A+B）和教师网络（A+C）
    自动检测 test/label 是否存在用于评估
    
    Args:
        test_dir: 测试数据目录
        use_teacher_mode: 是否为教师网络模式（True: 使用 A+C，False: 使用 A+B）
    """
    def __init__(self, test_dir, use_teacher_mode=False):
        self.test_dir = test_dir
        self.use_teacher_mode = use_teacher_mode
        
        # 读取图像路径
        img_A_dir = os.path.join(test_dir, 'A')
        
        # 根据模式选择第二个输入目录
        if use_teacher_mode:
            img_second_dir = os.path.join(test_dir, 'C')
            mode_name = "教师模式（A+C 同源）"
        else:
            img_second_dir = os.path.join(test_dir, 'B')
            mode_name = "学生模式（A+B 异源）"
        
        if not os.path.exists(img_A_dir):
            raise ValueError(f"数据集目录 A 不存在: {img_A_dir}")
        if not os.path.exists(img_second_dir):
            raise ValueError(f"数据集目录 {'C' if use_teacher_mode else 'B'} 不存在: {img_second_dir}")
        
        self.img_A_dir = img_A_dir
        self.img_second_dir = img_second_dir
        
        self.img_names = sorted([f for f in os.listdir(img_A_dir) if f.endswith(('.png', '.tif', '.jpg'))])
        
        # 自动检测标签目录是否存在
        label_dir = os.path.join(test_dir, 'label')
        self.has_label = os.path.exists(label_dir)
        
        print(f"数据集模式: {mode_name}")
        if self.has_label:
            print(f"检测到标签目录，将计算评估指标")
        else:
            print(f"未检测到标签目录，将仅输出预测结果")
        
        print(f"加载了 {len(self.img_names)} 个测试样本")
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # 读取图像 A（时间点1，光学）
        img_A_path = os.path.join(self.img_A_dir, img_name)
        img_A = io.imread(img_A_path)
        
        # 教师模式只需要 A 和 C
        if self.use_teacher_mode:
            # 读取 C（光学）
            img_C_path = os.path.join(self.img_second_dir, img_name)
            img_C = io.imread(img_C_path)
            
            # 转换为张量
            img_A = torch.from_numpy(img_A).permute(2, 0, 1).float() / 255.0
            img_C = torch.from_numpy(img_C).permute(2, 0, 1).float() / 255.0
            img_B = img_A  # 教师模式不需要 B，用 A 占位（避免 sar_encoder 处理错误类型）
        else:
            # 学生模式只需要 A 和 B
            img_B_path = os.path.join(self.img_second_dir, img_name)
            img_B = io.imread(img_B_path)
            
            img_A = torch.from_numpy(img_A).permute(2, 0, 1).float() / 255.0
            img_B = torch.from_numpy(img_B).permute(2, 0, 1).float() / 255.0
            img_C = img_B  # 学生模式不需要 C，用 B 占位
        
        # 读取标签（如果存在）
        label = None
        if self.has_label:
            label_path = os.path.join(self.test_dir, 'label', img_name)
            if os.path.exists(label_path):
                label = io.imread(label_path)
                # 二值化标签
                label = (label > 127).astype(np.uint8)
            else:
                # 单个样本缺失标签则返回 None，不影响其他样本
                label = None
        
        # 统一返回格式：A, B, C
        # 训练时模型期望 forward(A, B, C)
        return img_A, img_B, img_C, label, img_name


def extract_iou_from_filename(filename):
    """
    从文件名中提取 IoU 值
    
    Args:
        filename: 文件名，如 'gold_376IoU66.63.pth'
        
    Returns:
        IoU 值（float），如果无法解析则返回 -1
    """
    import re
    # 匹配 IoU 后面的数字，支持小数
    match = re.search(r'IoU(\d+\.?\d*)', filename, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return -1
    return -1


def load_model(checkpoint_path, device='cuda', force_teacher=None):
    """
    加载模型权重，自动处理键名不匹配问题
    
    Args:
        checkpoint_path: 权重文件路径或目录
        device: 运行设备
        force_teacher: 强制指定是否为教师模式（None=自动检测，True=教师，False=学生）
        
    Returns:
        加载好权重的模型, 是否为教师模式
    """
    is_teacher_weight = False
    
    # 如果是目录，自动查找权重文件
    if os.path.isdir(checkpoint_path):
        # 优先级: best_checkpoint.pth > IoU最高的.pth > last_checkpoint.pth
        
        # 1. 首先查找 best_checkpoint.pth
        best_path = os.path.join(checkpoint_path, 'best_checkpoint.pth')
        if os.path.exists(best_path):
            checkpoint_path = best_path
            print(f"找到 best_checkpoint.pth")
        else:
            # 2. 查找所有 .pth 文件并按 IoU 排序
            all_pth_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
            
            if not all_pth_files:
                raise FileNotFoundError(f"目录 {checkpoint_path} 中未找到任何 .pth 权重文件")
            
            # 提取包含 IoU 的文件（区分教师和学生权重）
            student_iou_files = []
            teacher_iou_files = []
            other_files = []
            
            for f in all_pth_files:
                if f == 'last_checkpoint.pth':
                    continue  # last_checkpoint 留到最后
                iou_value = extract_iou_from_filename(f)
                if iou_value > 0:
                    if 'teacher' in f.lower():
                        teacher_iou_files.append((f, iou_value))
                    else:
                        student_iou_files.append((f, iou_value))
                else:
                    other_files.append(f)
            
            # 根据 force_teacher 参数选择权重
            selected_files = None
            if force_teacher is True:
                selected_files = teacher_iou_files
                is_teacher_weight = True
            elif force_teacher is False:
                selected_files = student_iou_files
                is_teacher_weight = False
            else:
                # 自动选择：优先学生权重
                if student_iou_files:
                    selected_files = student_iou_files
                    is_teacher_weight = False
                elif teacher_iou_files:
                    selected_files = teacher_iou_files
                    is_teacher_weight = True
            
            # 按 IoU 降序排序
            if selected_files:
                selected_files.sort(key=lambda x: x[1], reverse=True)
                checkpoint_path = os.path.join(checkpoint_path, selected_files[0][0])
                weight_type = "教师" if is_teacher_weight else "学生"
                print(f"找到 IoU 最高的{weight_type}权重: {selected_files[0][0]} (IoU={selected_files[0][1]})")
            elif other_files:
                # 3. 如果没有 IoU 文件，使用其他文件
                checkpoint_path = os.path.join(checkpoint_path, other_files[0])
                print(f"使用权重文件: {other_files[0]}")
            else:
                # 4. 最后才使用 last_checkpoint.pth
                last_path = os.path.join(checkpoint_path, 'last_checkpoint.pth')
                if os.path.exists(last_path):
                    checkpoint_path = last_path
                    print(f"使用 last_checkpoint.pth")
                else:
                    raise FileNotFoundError(f"目录 {checkpoint_path} 中未找到可用的权重文件")
    else:
        # 直接指定文件时，从文件名判断
        filename = os.path.basename(checkpoint_path)
        is_teacher_weight = 'teacher' in filename.lower()
    
    # 创建模型
    model = Net(input_nc=3, output_nc=2).to(device)
    
    # 加载权重
    print(f"加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取状态字典
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        epoch_info = checkpoint.get('epoch', 'unknown')
        print(f"加载的权重来自 epoch: {epoch_info}")
    else:
        state_dict = checkpoint
    
    # 规范化键名
    state_dict = normalize_state_dict(state_dict)
    
    # 检查是否为教师专用权重（仅包含 optical_encoder 和 teacher_decoder）
    has_only_teacher = all(
        k.startswith('optical_encoder.') or k.startswith('teacher_decoder.')
        for k in state_dict.keys()
    )
    
    if has_only_teacher and not is_teacher_weight:
        print("⚠ 检测到教师专用权重，自动切换到教师模式")
        is_teacher_weight = True
    
    # 如果是教师专用权重，需要从完整checkpoint或best_checkpoint加载学生部分进行补全
    if has_only_teacher:
        print("检测到教师专用权重（仅 optical_encoder + teacher_decoder），正在加载完整模型进行补全...")
        
        # 查找对应的完整权重文件
        base_dir = os.path.dirname(checkpoint_path)
        complete_checkpoint = None
        
        # 优先使用 best_checkpoint.pth
        best_path = os.path.join(base_dir, 'best_checkpoint.pth')
        if os.path.exists(best_path):
            complete_checkpoint = best_path
        else:
            # 查找同epoch的完整权重
            filename = os.path.basename(checkpoint_path)
            # 从 gold_teacher_398IoU66.24.pth 提取 epoch
            import re
            match = re.search(r'teacher_(\d+)IoU', filename)
            if match:
                epoch_num = match.group(1)
                # 查找对应的学生权重
                student_pattern = f"gold_{epoch_num}IoU*.pth"
                import glob
                candidates = glob.glob(os.path.join(base_dir, student_pattern.replace('*', '[0-9]*')))
                if candidates:
                    complete_checkpoint = candidates[0]
        
        # 如果找不到，使用 last_checkpoint
        if complete_checkpoint is None:
            last_path = os.path.join(base_dir, 'last_checkpoint.pth')
            if os.path.exists(last_path):
                complete_checkpoint = last_path
        
        if complete_checkpoint:
            print(f"  使用完整权重初始化: {os.path.basename(complete_checkpoint)}")
            complete_state = torch.load(complete_checkpoint, map_location=device)
            if isinstance(complete_state, dict) and 'model_state' in complete_state:
                complete_state = complete_state['model_state']
            complete_state = normalize_state_dict(complete_state)
            
            # 先加载完整权重，再用教师权重覆盖
            model.load_state_dict(complete_state, strict=False)
            model.load_state_dict(state_dict, strict=False)
            print("✓ 教师权重加载成功（已用完整模型补全缺失部分）")
            missing_keys, unexpected_keys = [], []
        else:
            print("⚠ 未找到完整权重，教师模型部分参数未初始化，预测结果可能不正确")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    else:
        # 加载完整权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # 报告加载情况
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("✓ 权重完整加载成功")
        else:
            if len(missing_keys) > 0:
                print(f"⚠ 缺失的键 ({len(missing_keys)}): {missing_keys[:5]}...")
            if len(unexpected_keys) > 0:
                print(f"⚠ 多余的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    # 应用 force_teacher 覆盖自动检测结果
    if force_teacher is not None:
        is_teacher_weight = force_teacher
    
    weight_type = "教师网络（AC 同源）" if is_teacher_weight else "学生网络（AB 异源）"
    print(f"模型类型: {weight_type}")
    
    # 教师模式下，确保 use_teacher 标志开启
    if is_teacher_weight and hasattr(model, 'use_teacher'):
        model.use_teacher = True
    elif is_teacher_weight and hasattr(model, 'module') and hasattr(model.module, 'use_teacher'):
        model.module.use_teacher = True
    
    model.eval()
    return model, is_teacher_weight


def create_error_visualization(pred, label):
    """
    创建误差可视化图像
    
    Args:
        pred: 预测结果 (H, W)，值为 0 或 1
        label: 真值标签 (H, W)，值为 0 或 1
        
    Returns:
        可视化图像 (H, W, 3)，BGR格式
        - 白色: 正确预测
        - 红色: 假阳性 (预测为1，实际为0)
        - 绿色: 假阴性 (预测为0，实际为1)
        - 黑色: 正确的背景
    """
    h, w = pred.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 正确的变化区域 (白色)
    correct_change = (pred == 1) & (label == 1)
    vis[correct_change] = [255, 255, 255]
    
    # 假阳性 (红色) - 预测为变化但实际未变化
    false_positive = (pred == 1) & (label == 0)
    vis[false_positive] = [0, 0, 255]  # BGR: 红色
    
    # 假阴性 (绿色) - 预测为未变化但实际变化
    false_negative = (pred == 0) & (label == 1)
    vis[false_negative] = [0, 255, 0]  # BGR: 绿色
    
    # 正确的背景保持黑色
    
    return vis


def predict(model, test_loader, save_dir, device='cuda', save_pred=True, save_label=True, save_error=True, use_teacher_output=False):
    """
    执行预测并保存结果
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        save_dir: 保存目录
        device: 运行设备
        save_pred: 是否保存预测结果 (_pred.png)
        save_label: 是否保存真值标签 (_label.png)
        save_error: 是否保存误差可视化 (_error.png)
        use_teacher_output: 是否使用教师网络输出（True 时取 output[2]，False 时取 output[0]）
        
    Returns:
        预测结果和标签（用于计算指标）
    """
    model.eval()
    
    # 创建保存目录 - 直接使用输出目录，不创建子文件夹
    os.makedirs(save_dir, exist_ok=True)
    
    all_preds = []
    all_labels = []
    has_label = False
    
    with torch.no_grad():
        for img_A, img_B, img_C, label, names in tqdm(test_loader, desc="预测中"):
            # 移到设备
            img_A = img_A.to(device)
            img_B = img_B.to(device)
            img_C = img_C.to(device)
            
            # 前向传播
            output = model(img_A, img_B, img_C)
            
            # 获取预测结果
            if isinstance(output, (list, tuple)):
                if use_teacher_output and len(output) >= 3:
                    output = output[2]  # 教师模式取第3个输出（teacher_pred）
                else:
                    output = output[0]  # 学生模式取第1个输出（student_pred）
            
            # 转换为概率
            pred_prob = F.softmax(output, dim=1)
            # 取变化类别的概率
            pred = pred_prob[:, 1, :, :].cpu().numpy()
            # 二值化
            pred_binary = (pred > 0.5).astype(np.uint8)
            
            # 保存每个样本
            for i in range(pred_binary.shape[0]):
                name = names[i] if isinstance(names, (list, tuple)) else names
                pred_img = pred_binary[i] * 255
                
                # 如果有标签，创建可视化
                if label is not None and label[i] is not None:
                    has_label = True
                    label_np = label[i].cpu().numpy() if torch.is_tensor(label[i]) else label[i]
                    all_preds.append(pred_binary[i])
                    all_labels.append(label_np)
                    
                    # 保存真值
                    if save_label:
                        label_img = label_np * 255
                        label_path = os.path.join(save_dir, f"{os.path.splitext(name)[0]}_label.png")
                        io.imsave(label_path, label_img.astype(np.uint8))
                    
                    # 保存预测值
                    if save_pred:
                        pred_vis_path = os.path.join(save_dir, f"{os.path.splitext(name)[0]}_pred.png")
                        io.imsave(pred_vis_path, pred_img.astype(np.uint8))
                    
                    # 保存误差可视化
                    if save_error:
                        error_vis = create_error_visualization(pred_binary[i], label_np)
                        error_path = os.path.join(save_dir, f"{os.path.splitext(name)[0]}_error.png")
                        cv2.imwrite(error_path, error_vis)
                else:
                    # 没有标签时，只保存预测二值图
                    all_preds.append(pred_binary[i])
                    if save_pred:
                        pred_path = os.path.join(save_dir, f"{os.path.splitext(name)[0]}_pred.png")
                        io.imsave(pred_path, pred_img.astype(np.uint8))
    
    print(f"\n预测完成！结果保存至: {save_dir}")
    if has_label:
        saved_types = []
        if save_label:
            saved_types.append("label")
        if save_pred:
            saved_types.append("pred")
        if save_error:
            saved_types.append("error")
        if saved_types:
            print(f"生成了 {len(all_preds)} 组可视化结果（{', '.join(saved_types)}）")
    else:
        if save_pred:
            print(f"生成了 {len(all_preds)} 个预测结果")
    
    return all_preds, all_labels if has_label else None


def calculate_metrics(preds, labels, num_classes=2):
    """
    计算评估指标
    
    Args:
        preds: 预测结果列表
        labels: 真值标签列表
        num_classes: 类别数
        
    Returns:
        指标字典
    """
    if labels is None or len(labels) == 0:
        print("没有标签，跳过指标计算")
        return None
    
    # 计算混淆矩阵
    confusion_matrix = get_confuse_matrix(num_classes, labels, preds)
    
    # 计算各项指标
    scores = cm2score(confusion_matrix)
    
    return scores


def count_parameters(model):
    """
    计算模型参数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        参数量（单位：百万）
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6


def calculate_flops(model, input_shape=(1, 3, 256, 256), device='cuda'):
    """
    计算模型 FLOPs
    
    Args:
        model: PyTorch 模型
        input_shape: 输入张量形状 (B, C, H, W)
        device: 运行设备
        
    Returns:
        FLOPs（单位：十亿）
    """
    try:
        from thop import profile
        # 创建三个输入（img_A, img_B, img_C）
        inputs = (
            torch.randn(input_shape).to(device),
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


def measure_inference_speed(model, test_loader, device='cuda', warmup=10, num_iterations=100):
    """
    测量模型推理速度
    
    Args:
        model: PyTorch 模型
        test_loader: 测试数据加载器
        device: 运行设备
        warmup: 预热次数
        num_iterations: 测试迭代次数
        
    Returns:
        平均推理时间（单位：毫秒）
    """
    model.eval()
    
    # 获取一个批次的数据用于测试
    test_iter = iter(test_loader)
    try:
        img_A, img_B, img_C, _, _ = next(test_iter)
    except StopIteration:
        print("警告: 测试集为空，无法测量推理速度")
        return -1
    
    img_A = img_A.to(device)
    img_B = img_B.to(device)
    img_C = img_C.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(img_A, img_B, img_C)
    
    # 测量时间
    if device == 'cuda':
        torch.cuda.synchronize()
    
    timings = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(img_A, img_B, img_C)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            timings.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    
    return avg_time, std_time


def main():
    parser = argparse.ArgumentParser(description='变化检测预测脚本')
    parser.add_argument('--checkpoint', type=str, 
                        default=CHECKPOINT,
                        help='权重文件路径')
    parser.add_argument('--test_dir', type=str, 
                        default=TEST_DIR,
                        help='测试数据目录（学生模式需要 A、B，教师模式需要 A、C）')
    parser.add_argument('--output_dir', type=str, 
                        default=OUTPUT_DIR,
                        help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批次大小')
    parser.add_argument('--device', type=str, default=DEVICE, help='运行设备')
    parser.add_argument('--input_size', type=int, default=INPUT_SIZE, help='输入图像尺寸（用于计算FLOPs）')
    # 教师/学生模式控制
    parser.add_argument('--teacher', action='store_true', default=None, help='强制使用教师模式（A+C）')
    parser.add_argument('--student', action='store_true', default=None, help='强制使用学生模式（A+B）')
    # 可视化输出控制
    parser.add_argument('--save_pred', action='store_true', default=None, help='保存预测结果')
    parser.add_argument('--no_save_pred', action='store_true', default=None, help='不保存预测结果')
    parser.add_argument('--save_label', action='store_true', default=None, help='保存真值标签')
    parser.add_argument('--no_save_label', action='store_true', default=None, help='不保存真值标签')
    parser.add_argument('--save_error', action='store_true', default=None, help='保存误差可视化')
    parser.add_argument('--no_save_error', action='store_true', default=None, help='不保存误差可视化')
    # 可选推理速度测量
    parser.add_argument('--measure_speed', action='store_true', default=None, help='测量推理速度（覆盖顶部默认）')
    parser.add_argument('--no_measure_speed', action='store_true', default=None, help='禁止测量推理速度（覆盖顶部默认）')
    parser.add_argument('--speed_warmup', type=int, default=SPEED_WARMUP, help='推理速度预热次数')
    parser.add_argument('--speed_iters', type=int, default=SPEED_ITERS, help='推理速度测量迭代次数')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 解析可视化输出开关
    save_pred = SAVE_PRED
    if args.save_pred is True:
        save_pred = True
    elif args.no_save_pred is True:
        save_pred = False
    
    save_label = SAVE_LABEL
    if args.save_label is True:
        save_label = True
    elif args.no_save_label is True:
        save_label = False
    
    save_error = SAVE_ERROR
    if args.save_error is True:
        save_error = True
    elif args.no_save_error is True:
        save_error = False

    # 解析速度测量开关
    measure_speed = MEASURE_SPEED
    if args.measure_speed is True:
        measure_speed = True
    elif args.no_measure_speed is True:
        measure_speed = False

    # 解析教师/学生模式
    force_teacher = None
    if args.teacher is True:
        force_teacher = True
    elif args.student is True:
        force_teacher = False

    # 加载模型（自动检测或强制指定模式）
    model, is_teacher_mode = load_model(args.checkpoint, device, force_teacher=force_teacher)
    
    # 创建数据集（根据模型类型选择数据）
    test_dataset = TestDataset(args.test_dir, use_teacher_mode=is_teacher_mode)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # 计算模型复杂度指标
    print("\n" + "="*70)
    print("模型复杂度分析")
    print("="*70)
    
    # 参数量
    total_params, trainable_params = count_parameters(model)
    print(f"总参数量: {total_params:.2f}M")
    print(f"可训练参数量: {trainable_params:.2f}M")
    
    # FLOPs
    print("\n计算 FLOPs...")
    flops = calculate_flops(model, input_shape=(1, 3, args.input_size, args.input_size), device=device)
    if flops > 0:
        print(f"FLOPs: {flops:.2f}G (输入尺寸: {args.input_size}x{args.input_size})")
    
    # 推理速度（可选）
    avg_time, std_time = -1, -1
    if measure_speed:
        print("\n测量推理速度...")
        speed_result = measure_inference_speed(
            model, test_loader, device=device, warmup=args.speed_warmup, num_iterations=args.speed_iters
        )
        if isinstance(speed_result, tuple):
            avg_time, std_time = speed_result
            print(f"推理速度: {avg_time:.2f} ± {std_time:.2f} ms")
        else:
            avg_time = speed_result
            std_time = 0
            if avg_time > 0:
                print(f"推理速度: {avg_time:.2f} ms")
    
    # 执行预测
    preds, labels = predict(
        model, 
        test_loader, 
        args.output_dir, 
        device, 
        save_pred=save_pred,
        save_label=save_label,
        save_error=save_error,
        use_teacher_output=is_teacher_mode  # 教师模式使用教师输出
    )
    
    # 计算指标
    if labels is not None:
        print("\n" + "="*70)
        print("评估指标:")
        print("="*70)
        
        metrics = calculate_metrics(preds, labels)
        
        if metrics:
            # 获取模型名称（区分教师和学生）
            method_name = "GOLD-Teacher" if is_teacher_mode else "GOLD-Student"
            
            # 准备表格数据
            precision = metrics['precision_1'] * 100  # 变化类精确率
            recall = metrics['recall_1'] * 100  # 变化类召回率
            f1 = metrics['F1_1'] * 100  # 变化类F1
            iou = metrics['iou_1'] * 100  # 变化类IoU
            
            # 打印表格
            print("\n综合性能指标表:")
            print("-" * 120)
            print(f"{'Method':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'IoU':<10} | {'Params(M)':<12} | {'FLOPs(G)':<12} | {'Speed(ms)':<15}")
            print("-" * 120)
            
            flops_str = f"{flops:.2f}" if flops > 0 else "N/A"
            speed_str = f"{avg_time:.2f}" if (isinstance(avg_time, (int, float)) and avg_time > 0) else "N/A"
            
            print(f"{method_name:<15} | {precision:>9.2f}% | {recall:>9.2f}% | {f1:>9.2f}% | {iou:>9.2f}% | {total_params:>11.2f} | {flops_str:>11} | {speed_str:>14}")
            print("-" * 120)
            
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
            
            # 保存指标到文件
            metrics_path = os.path.join(args.output_dir, 'metrics.txt')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("综合性能指标\n")
                f.write("="*70 + "\n\n")
                
                # 写入表格
                f.write("-" * 120 + "\n")
                f.write(f"{'Method':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'IoU':<10} | {'Params(M)':<12} | {'FLOPs(G)':<12} | {'Speed(ms)':<15}\n")
                f.write("-" * 120 + "\n")
                f.write(f"{method_name:<15} | {precision:>9.2f}% | {recall:>9.2f}% | {f1:>9.2f}% | {iou:>9.2f}% | {total_params:>11.2f} | {flops_str:>11} | {speed_str:>14}\n")
                f.write("-" * 120 + "\n\n")
                
                # 写入详细指标
                f.write("="*50 + "\n")
                f.write("详细评估指标\n")
                f.write("="*50 + "\n\n")
                f.write(f"整体准确率 (Accuracy): {metrics['acc']*100:.2f}%\n")
                f.write(f"平均IoU (mIoU): {metrics['miou']*100:.2f}%\n")
                f.write(f"平均F1分数 (mF1): {metrics['mf1']*100:.2f}%\n\n")
                f.write(f"类别IoU:\n")
                f.write(f"  背景 (IoU_0): {metrics['iou_0']*100:.2f}%\n")
                f.write(f"  变化 (IoU_1): {metrics['iou_1']*100:.2f}%\n\n")
                f.write(f"类别F1:\n")
                f.write(f"  背景 (F1_0): {metrics['F1_0']*100:.2f}%\n")
                f.write(f"  变化 (F1_1): {metrics['F1_1']*100:.2f}%\n\n")
                f.write(f"精确率和召回率:\n")
                f.write(f"  变化类精确率: {metrics['precision_1']*100:.2f}%\n")
                f.write(f"  变化类召回率: {metrics['recall_1']*100:.2f}%\n\n")
                
                # 写入模型复杂度
                f.write("="*50 + "\n")
                f.write("模型复杂度\n")
                f.write("="*50 + "\n\n")
                f.write(f"总参数量: {total_params:.2f}M\n")
                f.write(f"可训练参数量: {trainable_params:.2f}M\n")
                if flops > 0:
                    f.write(f"FLOPs: {flops:.2f}G\n")
                if isinstance(avg_time, (int, float)) and avg_time > 0:
                    if isinstance(std_time, (int, float)) and std_time > 0:
                        f.write(f"推理速度: {avg_time:.2f} ± {std_time:.2f} ms\n")
                    else:
                        f.write(f"推理速度: {avg_time:.2f} ms\n")
            
            print(f"\n指标已保存至: {metrics_path}")
    
    print("\n预测完成！")


if __name__ == '__main__':
    main()
