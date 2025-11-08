"""
超参占比探索批量训练脚本
自动运行3个不同权重配置的训练实验
"""

import os
import sys
import time
import random
import socket
import shlex
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# 添加项目路径
working_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, working_path)

from configs_weight_ablation import EXPERIMENT_CONFIGS, get_config
from utils.utils_fit import train
from utils.loss import CrossEntropyLoss2d
from datasets import RS_ST as RS
from models.hetecd import hetecd as Net

# 环境配置
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.set_num_threads(4)

# TF32与禁用Inductor autotune减少编译与运行开销
os.environ.setdefault('TORCHINDUCTOR_MAX_AUTOTUNE', '0')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# 固定随机种子
seed = 222
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(seed)

# 基础配置
NET_NAME = 'gold'
DATA_NAME = 'trios43'


def find_free_port(start_port=6006, max_attempts=100):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"在端口 {start_port}-{start_port+max_attempts} 范围内未找到可用端口")


def run_experiment(config_name):
    """
    运行单个配置的训练实验
    
    Args:
        config_name: 配置名称（如 'config_A'）
    """
    print("\n" + "=" * 100)
    print(f"开始实验: {config_name}")
    print("=" * 100 + "\n")
    
    # 获取配置
    config = get_config(config_name)
    
    # 创建实验名称
    EXP_NAME = f"EXP{time.strftime('%Y%m%d%H%M%S')}_{config['name']}"
    
    # 训练参数
    args = {
        'net_name': NET_NAME,
        'data_name': DATA_NAME,
        'exp_name': EXP_NAME,
        'train_batch_size': 4,
        'val_batch_size': 4,
        'dice': True,
        'lr': 0.0005,
        'epochs': 400,
        'gpu': True,
        'lr_decay_power': 1.0,
        'weight_decay': 0.01,
        'momentum': 0.9,
        'print_freq': 5,
        'predict_step': 5,
        'pred_dir': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, EXP_NAME, "vis"),
        'chkpt_dir': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, EXP_NAME),
        'log_dir': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, EXP_NAME),
        'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth'),
        'use_multi_img_photometric': True,
        'resume_path': None,
        'log_name': "/log.txt",
        'loss_weights': config['loss_weights'],  # 使用配置中的损失权重
    }
    
    # 创建目录
    os.makedirs(args['log_dir'], exist_ok=True)
    os.makedirs(args['pred_dir'], exist_ok=True)
    os.makedirs(args['chkpt_dir'], exist_ok=True)
    
    # 写入配置信息到日志
    with open(os.path.join(args['log_dir'], 'log.txt'), 'w') as f:
        f.write('=' * 80 + '\n')
        f.write(f'实验配置: {config["name"]}\n')
        f.write(f'描述: {config["description"]}\n')
        f.write(f'预期损失占比: {config["expected_loss_ratio"]}\n')
        f.write('=' * 80 + '\n\n')
        
        if 'key_changes' in config:
            f.write('关键变化:\n')
            for change in config['key_changes']:
                f.write(f'  - {change}\n')
            f.write('\n')
        
        f.write('=' * 80 + '\n')
        f.write('训练参数:\n')
        for key, value in args.items():
            if key != 'loss_weights':
                f.write(f'  {key}: {value}\n')
        f.write('=' * 80 + '\n\n')
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # 检查GPU状态
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU设备: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
        print(f"已分配内存: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
    
    # 创建模型
    net = Net(input_nc=3, output_nc=2).cuda()
    net = nn.DataParallel(net)
    
    # 加载backbone权重
    model_dict = net.state_dict()
    pretrained_dict = torch.load("backbone_weights.pth")
    load_key, no_load_key, temp_dict = [], [], {}
    
    for k, v in pretrained_dict.items():
        if "backbone" in k:
            k1 = k.replace("backbone.", "optical_encoder.")
            k2 = k.replace("backbone.", "sar_encoder.")
            if k1 in model_dict.keys() and np.shape(model_dict[k1]) == np.shape(v):
                temp_dict[k1] = v
                load_key.append(k1)
            if k2 in model_dict.keys() and np.shape(model_dict[k2]) == np.shape(v):
                temp_dict[k2] = v
                load_key.append(k2)
        else:
            no_load_key.append(k)
    
    model_dict.update(temp_dict)
    net.load_state_dict(model_dict)
    
    print(f"\n匹配成功的Key数量: {len(load_key)}")
    print(f"匹配失败的Key数量: {len(no_load_key)}")
    
    # 模型编译（单GPU时）
    if torch.cuda.device_count() <= 1:
        try:
            net = torch.compile(net, mode='reduce-overhead')
            print("已启用 torch.compile(mode='reduce-overhead')")
        except Exception as e:
            print(f"torch.compile 跳过: {e}")
    
    # 数据加载器
    train_set_change = RS.Data('train', random_flip=True, use_multi_img_photometric=args['use_multi_img_photometric'])
    train_loader_change = DataLoader(
        train_set_change,
        batch_size=args['train_batch_size'],
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        pin_memory_device='cuda',
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    
    train_set_unchange = RS.Data('train', random_flip=True, use_multi_img_photometric=args['use_multi_img_photometric'])
    train_loader_unchange = DataLoader(
        train_set_unchange,
        batch_size=args['train_batch_size'],
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        pin_memory_device='cuda',
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    
    val_set = RS.Data('val', use_multi_img_photometric=False)
    val_loader = DataLoader(
        val_set,
        batch_size=args['val_batch_size'],
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        pin_memory_device='cuda',
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )
    
    # 损失函数和优化器
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    
    try:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args['lr'],
            betas=(0.9, 0.999),
            weight_decay=args['weight_decay'],
            fused=True
        )
    except TypeError:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args['lr'],
            betas=(0.9, 0.999),
            weight_decay=args['weight_decay']
        )
    
    # 学习率调度器
    steps_per_epoch = max(1, len(train_loader_change))
    total_iters = args['epochs'] * steps_per_epoch
    warmup_iters = max(steps_per_epoch, total_iters // 50)
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / args['lr'],
        end_factor=1.0,
        total_iters=warmup_iters
    )
    
    main_scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=total_iters - warmup_iters,
        power=args['lr_decay_power'],
    )
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_iters]
    )
    
    args['use_onecycle'] = True
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args['log_dir'])
    args['tb_writer'] = writer
    
    # 查找可用端口并记录
    try:
        tb_port = find_free_port()
        tb_logdir_safe = shlex.quote(args['log_dir'])
        tb_command = f"tensorboard --logdir {tb_logdir_safe} --port {tb_port}"
        tb_url = f"http://localhost:{tb_port}"
        
        with open(os.path.join(args['log_dir'], 'log.txt'), 'a') as f:
            f.write('=' * 80 + '\n')
            f.write('TensorBoard 可视化信息:\n')
            f.write(f'  可用端口: {tb_port}\n')
            f.write(f'  启动命令: {tb_command}\n')
            f.write(f'  访问地址: {tb_url}\n')
            f.write('=' * 80 + '\n\n')
        
        print('=' * 80)
        print('TensorBoard 可视化信息:')
        print(f'  可用端口: {tb_port}')
        print(f'  启动命令: {tb_command}')
        print(f'  访问地址: {tb_url}')
        print('=' * 80 + '\n')
    except Exception as e:
        print(f"TensorBoard 端口查找失败: {e}")
    
    # 开始训练
    print(f"\n开始训练配置: {config_name}")
    print(f"实验目录: {args['chkpt_dir']}\n")
    
    try:
        train(train_loader_change, train_loader_unchange, net, criterion, optimizer, scheduler, val_loader, args)
        print(f"\n配置 {config_name} 训练完成！")
    except Exception as e:
        print(f"\n配置 {config_name} 训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭TensorBoard
        try:
            writer.close()
        except Exception:
            pass
        
        # 清理GPU内存
        torch.cuda.empty_cache()


def main():
    """主函数：依次运行所有配置"""
    print("\n" + "=" * 100)
    print("超参占比探索批量训练")
    print("=" * 100)
    print(f"\n将依次运行以下配置: {EXPERIMENT_CONFIGS}")
    print(f"每个配置训练 400 epochs")
    print(f"固定随机种子: {seed}")
    print("\n" + "=" * 100 + "\n")
    
    # 打印所有配置摘要
    from configs_weight_ablation import print_config_summary
    print_config_summary()
    
    # 依次运行实验
    for i, config_name in enumerate(EXPERIMENT_CONFIGS, 1):
        print(f"\n进度: [{i}/{len(EXPERIMENT_CONFIGS)}]")
        run_experiment(config_name)
        
        # 短暂休息，让GPU冷却
        if i < len(EXPERIMENT_CONFIGS):
            print("\n等待5秒后开始下一个实验...")
            time.sleep(5)
    
    print("\n" + "=" * 100)
    print("所有实验完成！")
    print("=" * 100)


if __name__ == '__main__':
    main()

