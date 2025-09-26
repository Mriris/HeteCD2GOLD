import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
working_path = os.path.dirname(os.path.abspath(__file__))
from utils.utils_fit import train
from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity,SCA_Loss,FeatureConsistencyLoss
from utils.utils import accuracy, SCDD_eval_all, AverageMeter, get_confuse_matrix, cm2score
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#Data and model choose
torch.set_num_threads(4)

# TF32与禁用Inductor autotune减少编译与运行开销
os.environ.setdefault('TORCHINDUCTOR_MAX_AUTOTUNE', '0')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

###############################################
from datasets import RS_ST as RS
# from models.BiSRNet import BiSRNet as Net
from models.hetecd import hetecd as Net
# 生成一个随机5位整数
import math
nums = math.floor(1e5 * random.random())
seed = 222
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(seed)
# from models.SSCDl import SSCDl as Net
NET_NAME = 'Tgold'
DATA_NAME = 'trios43'
EXP_NAME = "EXP"+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+"MSE"
###############################################    
#Training options
###############################################
args = {
    'net_name': NET_NAME,
    'data_name': DATA_NAME,
    'exp_name':EXP_NAME,
    'train_batch_size': 4,
    'val_batch_size': 4,
    'dice': True,
    'lr': 0.0005,
    'epochs': 400,
    'gpu': True,
    'lr_decay_power': 1.5,
    'weight_decay': 1e-2,
    'momentum': 0.9,
    'print_freq': 5,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'checkpoints',  NET_NAME,DATA_NAME,EXP_NAME,"vis"),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME,EXP_NAME),
    'log_dir': os.path.join(working_path, 'checkpoints', NET_NAME,DATA_NAME, EXP_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth'),
    'use_multi_img_photometric': True,
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
# writer = SummaryWriter(args['log_dir'])
#日期时间作为日志文件名
args['log_name']="/log.txt"
def main():        
    # 清空GPU缓存
    torch.cuda.empty_cache()
    # 设置内存分配策略
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # 检查GPU内存状态
    if torch.cuda.is_available():
        device = torch.cuda.current_device()  # 自动获取当前可用设备
        print(f"GPU设备: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
        print(f"已分配内存: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
        print(f"缓存内存: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")
    
    net = Net(3).cuda()
    # 保持输入为 channels_last，模型不整体转换，避免 rank 不一致错误
    net = nn.DataParallel(net)
    model_dict      = net.state_dict()
    pretrained_dict = torch.load("backbone_weights.pth")
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if "backbone" in k:
            # 映射到光学编码器和SAR编码器
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
#------------------------------------------------------#
#   显示没有匹配上的Key
#------------------------------------------------------#
    if len(load_key) == 0:
        print("没有匹配上的Key")
    else:
        print("\n匹配成功的Key:", str(load_key)[:5000], "……\n匹配成功的Key Num:", len(load_key))
    if len(no_load_key) == 0:
        print("所有Key都已匹配")
    else:
        print("\n匹配失败的Key:", str(no_load_key)[:5000], "……\n匹配失败的Key num:", len(no_load_key))
    model_dict.update(temp_dict)
    net.load_state_dict(model_dict)
    
    # 显示模型加载后的内存使用情况
    device = torch.cuda.current_device()
    print(f"\n模型加载后的内存状态:")
    print(f"已分配内存: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
    print(f"缓存内存: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")
    print(f"可用内存: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_reserved(device)) / (1024**3):.2f} GB")
    # 在加载权重后启用编译以降低 Python/框架开销
    try:
        net.module = torch.compile(net.module, mode='reduce-overhead')
        print("已启用 torch.compile(mode='reduce-overhead')")
    except Exception as e:
        print(f"torch.compile 跳过：{e}")
        
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
    
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    try:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], betas=(0.9, 0.999), weight_decay=0.01, fused=True)
    except TypeError:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    # 使用 OneCycleLR 提速早期收敛
    steps_per_epoch = max(1, len(train_loader_change))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args['lr'],
        epochs=args['epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
    )
    args['use_onecycle'] = True

    train(train_loader_change, train_loader_unchange,net, criterion, optimizer, scheduler, val_loader, args)
    print('Training finished.')

        
if __name__ == '__main__':
    main()
