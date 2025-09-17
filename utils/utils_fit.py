import os
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim

# 抑制 scikit-image 的低对比度警告
warnings.filterwarnings("ignore", message=".*low contrast image.*")
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
working_path = os.path.dirname(os.path.abspath(__file__))
import copy
from utils.loss import CE_Loss, AlignmentLoss, ChangeSimilarity,Dice_loss, KLDivergenceLoss,FeatureConsistencyLoss
from utils.utils import accuracy, SCDD_eval_all, AverageMeter, get_confuse_matrix, cm2score
import torch.nn.functional as FF

# 损失权重配置
DEFAULT_LOSS_WEIGHTS = {
    'kd_weight': 5e-4,
    'kd_warmup_epochs': 12,
    'kd_loss_clip': 50.0,
    'teacher_weight': 0.04,
    'teacher_warmup_epochs': 8,
    'align_scale_student': 0.3,
    'align_scale_teacher': 0.2,
    'align_base_weight': 0.5,
    'temperature': 8.0,
    'ce_class_weights': (0.2, 0.8),
}


def _compute_kd_weight(epoch: int, loss_weights: dict) -> float:
    """Return epoch-specific KD multiplier with optional warm-up."""
    warmup = loss_weights.get('kd_warmup_epochs', 0)
    if warmup and warmup > 0:
        scale = min(1.0, (epoch + 1) / float(warmup))
    else:
        scale = 1.0
    return loss_weights['kd_weight'] * scale


def _compute_teacher_weight(epoch: int, loss_weights: dict) -> float:
    """根据epoch返回教师分支的动态权重。"""
    warmup = loss_weights.get('teacher_warmup_epochs', 0)
    base = loss_weights.get('teacher_weight', 0.0)
    if warmup and warmup > 0:
        scale = min(1.0, (epoch + 1) / float(warmup))
    else:
        scale = 1.0
    return base * scale


def _make_class_weights(loss_weights: dict, device: torch.device) -> torch.Tensor:
    """Create class weights tensor for CE loss on the desired device."""
    weights = loss_weights.get('ce_class_weights', (1.0, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(train_loader,train_loader_unchange, net, criterion, optimizer, scheduler, val_loader, args):
    NET_NAME = args['net_name']
    bestiouT=0
    bestiou=0.0
    bestloss=1.0
    bestaccV = 0.0
    begin_time = time.time()
    all_iters = float(len(train_loader)*args['epochs'])
    al_loss = AlignmentLoss()
    # kl_loss = KLDivergenceLoss()
    fc_loss = FeatureConsistencyLoss()
    # criterion_sc = SCA_Loss().cuda()
    # 训练权重配置
    LOSS_WEIGHTS = copy.deepcopy(DEFAULT_LOSS_WEIGHTS)
    LOSS_WEIGHTS.update(args.get('loss_weights', {}))
    
    # 记录权重配置到日志开头
    args['loss_weights'] = LOSS_WEIGHTS

    curr_epoch=0
    if curr_epoch == 0:
        with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            f.write('='*80 + '\n')
            f.write('损失权重配置:\n')
            for key, value in LOSS_WEIGHTS.items():
                f.write(f'  {key}: {value}\n')
            f.write('='*80 + '\n\n')
    
    while True:
        net.train()
        #freeze_model(net.FCN)
        start = time.time()
        train_loss = AverageMeter()
        # 各个损失分量的追踪器
        train_loss_ce = AverageMeter()
        train_loss_teacher = AverageMeter()
        train_loss_kd = AverageMeter()
        train_loss_align_student = AverageMeter()
        train_loss_align_teacher = AverageMeter()
        train_loss_dice = AverageMeter()
        curr_iter = curr_epoch*len(train_loader)
        kd_lambda = _compute_kd_weight(curr_epoch, LOSS_WEIGHTS)
        teacher_lambda = _compute_teacher_weight(curr_epoch, LOSS_WEIGHTS)
        epoch_align_weight = (1.0 / (curr_epoch + 1)) * LOSS_WEIGHTS['align_base_weight']
        preds_all = []
        labels_all = []
        names_all = []
        
        # # print(i)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter+i+1
            adjust_lr(optimizer, running_iter, all_iters,args)
            imgs_A, imgs_B, imgs_C, labels, names = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                imgs_C = imgs_C.cuda().float()
                labels = labels.cuda().long()
            
            optimizer.zero_grad()
            
            # 前向传播
            forward_result = net(imgs_A, imgs_B, imgs_C)
            
            if len(forward_result) == 4:
                # 师生模式：学生预测 + 学生特征 + 教师预测 + 教师特征
                out_change, student_features, teacher_pred, teacher_features = forward_result
                out_change = FF.interpolate(out_change, size=(512,512), mode='bilinear', align_corners=True)
                teacher_pred = FF.interpolate(teacher_pred, size=(512,512), mode='bilinear', align_corners=True)
                
                cls_weights = _make_class_weights(LOSS_WEIGHTS, labels.device)
                
                # 学生损失（异源ab）
                loss_ce = CE_Loss(out_change, labels, cls_weights=cls_weights)
                
                # 教师损失（同源ac）  
                loss_teacher = CE_Loss(teacher_pred, labels, cls_weights=cls_weights)
                
                # 学生和教师的特征对齐损失
                align_loss_student = al_loss(student_features)
                align_loss_teacher = al_loss(teacher_features)
                
                # 知识蒸馏损失（学生学习教师）
                temperature = LOSS_WEIGHTS['temperature']
                student_log_prob = F.log_softmax(out_change / temperature, dim=1)
                teacher_prob = F.softmax(teacher_pred.detach() / temperature, dim=1)
                kd_loss_map = F.kl_div(
                    student_log_prob,
                    teacher_prob,
                    reduction='none'
                )
                kd_loss = kd_loss_map.sum(dim=1).mean() * (temperature ** 2)
                kd_clip = LOSS_WEIGHTS.get('kd_loss_clip')
                if kd_clip is not None:
                    kd_loss = torch.clamp(kd_loss, max=kd_clip)

                # Dice损失
                dice_loss_val = 0.0
                if args['dice']:
                    loss_dice = Dice_loss(out_change, labels)
                    loss_ce = loss_ce + loss_dice
                    dice_loss_val = loss_dice.item()
                    
                # 动态调整对齐损失权重：训练初期权重大，后期逐渐减小
                # 总损失：学生损失 + 教师损失 + 知识蒸馏 + 特征对齐
                loss = (loss_ce + 
                       teacher_lambda * loss_teacher + 
                       kd_lambda * kd_loss + 
                       epoch_align_weight * LOSS_WEIGHTS['align_scale_student'] * align_loss_student +
                       epoch_align_weight * LOSS_WEIGHTS['align_scale_teacher'] * align_loss_teacher)
                
                # 记录各个损失分量
                train_loss_ce.update(loss_ce.item())
                train_loss_teacher.update(loss_teacher.item())
                train_loss_kd.update(kd_loss.item())
                train_loss_align_student.update(align_loss_student.item())
                train_loss_align_teacher.update(align_loss_teacher.item())
                if args['dice']:
                    train_loss_dice.update(dice_loss_val)
                
            else:
                # 推理模式：只有学生预测 + 特征
                out_change, features = forward_result
                out_change = FF.interpolate(out_change, size=(512,512), mode='bilinear', align_corners=True)
                
                cls_weights = _make_class_weights(LOSS_WEIGHTS, labels.device)
                loss_ce = CE_Loss(out_change, labels, cls_weights=cls_weights)
                align_loss = al_loss(features)
                
                dice_loss_val = 0.0
                if args['dice']:
                    loss_dice = Dice_loss(out_change, labels)
                    loss = loss_ce + loss_dice
                    dice_loss_val = loss_dice.item()
                else:
                    loss = loss_ce
                # 动态调整对齐损失权重：训练初期权重大，后期逐渐减小
                loss = loss + epoch_align_weight * align_loss
                
                # 记录各个损失分量（推理模式）
                train_loss_ce.update(loss_ce.item())
                train_loss_align_student.update(align_loss.item())
                if args['dice']:
                    train_loss_dice.update(dice_loss_val)
                # 推理模式下没有教师损失和KD损失，设为0
                train_loss_teacher.update(0.0)
                train_loss_kd.update(0.0)
                train_loss_align_teacher.update(0.0)
            
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(out_change, dim=1)
            pred_numpy = preds.cpu().numpy()
            labels_numpy = labels.cpu().numpy()

            preds_all.append(pred_numpy)
            labels_all.append(labels_numpy)
            names_all.extend(names)
            # hist = get_confuse_matrix(2,labels,preds.cpu().numpy())
            # score = cm2score(hist)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start
            # if (i + 1) % args['print_freq'] == 0:
            #     print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [loss %.4f] [score %s]' % (
            #         curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
            #         train_loss.val, {key: score[key] for key in score})) 
            #     with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            #         f.write('[epoch %d] [iter %d / %d %.1fs] [lr %f] [bn_loss %.4f] [score %s]\n' % (
            #                         curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
            #                         train_loss.val, {key: score[key] for key in score}))
            
        # if curr_epoch>200 and curr_epoch<260:
        #     for i, data in enumerate(train_loader_unchange):
        #         running_iter = curr_iter+i+1
        #         adjust_lr(optimizer, running_iter, all_iters,args)
        #         imgs_A, imgs_B, labels, names = data
        #         # print(names)
        #         # print(imgs_A.shape, imgs_B.shape, labels.shape)
        #         if args['gpu']:
        #             imgs_A = imgs_A.cuda().float()
        #             imgs_B = imgs_B.cuda().float()
        #             labels = labels.cuda().long()
        #             # labels = labels>10
        #         optimizer.zero_grad()
        #         out_change,features = net(imgs_A, imgs_B)
        #         # out_change_neg,features_neg = net(imgs_A, torch.roll(imgs_B, shifts=-1, dims=0),)
        #         # ones_tensor = torch.ones(out_change.size(0), out_change.size(2), out_change.size(3)).cuda().long()
        #         # zeros_tensor = torch.zeros(out_change.size(0), out_change.size(2), out_change.size(3)).cuda().long()
        #         align_loss = kl_loss(features)
        #         loss_ce = CE_Loss(out_change, labels)
                
        #         if args['dice']:
        #             loss_dice = Dice_loss(out_change, labels)
        #             loss_seg =  loss_ce + loss_dice
        #         else:
        #             loss_seg =  loss_ce
        #         loss = align_loss+loss_seg*0.1
        #         loss.backward()
        #         optimizer.step()
        #         curr_time = time.time() - start
        #         # 打印align_loss
        #         if (i + 1) % args['print_freq'] == 0:
        #             print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [loss %.4f] [align_loss %.4f] [seg_loss %.4f]' % (
        #                 curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
        #                 loss, align_loss, loss_seg))
        #             with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
        #                 f.write('[epoch %d] [iter %d / %d %.1fs] [lr %f] [bn_loss %.4f] [align_loss %.4f] [seg_loss %.4f]\n' % (
        #                                 curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
        #                                 loss, align_loss, loss_seg))
        
        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)   
        score_train = cm2score(get_confuse_matrix(2,labels_all, preds_all))     
        
        # 详细损失信息
        loss_details = {
            'total_loss': train_loss.average(),
            'ce_loss': train_loss_ce.average(),
            'teacher_loss': train_loss_teacher.average(),
            'kd_loss': train_loss_kd.average(),
            'align_student_loss': train_loss_align_student.average(),
            'align_teacher_loss': train_loss_align_teacher.average()
        }
        if args['dice']:
            loss_details['dice_loss'] = train_loss_dice.average()

        epoch_align_weight = (1.0 / (curr_epoch + 1)) * LOSS_WEIGHTS['align_base_weight']

        with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            f.write('Epoch: %d  Total time: %.1fs  Train loss %.4f  score %s\n' %(curr_epoch, time.time()-begin_time,train_loss.average(),{key: score_train[key] for key in score_train}))
            f.write('  Detailed losses: CE=%.4f, Teacher=%.4f, KD=%.4f, AlignS=%.4f, AlignT=%.4f' % (
                train_loss_ce.average(), train_loss_teacher.average(), train_loss_kd.average(),
                train_loss_align_student.average(), train_loss_align_teacher.average()))
            if args['dice']:
                f.write(', Dice=%.4f' % train_loss_dice.average())
            f.write(', Epoch_align_weight=%.4f, Epoch_teacher_weight=%.6f, Epoch_kd_weight=%.6f\n' % (epoch_align_weight, teacher_lambda, kd_lambda))

        print('Epoch: %d  Total time: %.1fs  Train loss %.4f  score %s' %(curr_epoch, time.time()-begin_time, train_loss.average(),{key: score_train[key] for key in score_train}))
        print('  Detailed losses: CE=%.4f, Teacher=%.4f, KD=%.4f, AlignS=%.4f, AlignT=%.4f' % (
            train_loss_ce.average(), train_loss_teacher.average(), train_loss_kd.average(),
            train_loss_align_student.average(), train_loss_align_teacher.average()), end='')
        if args['dice']:
            print(', Dice=%.4f' % train_loss_dice.average(), end='')
        print(', Epoch_align_weight=%.4f, Epoch_teacher_weight=%.6f, Epoch_kd_weight=%.6f' % (epoch_align_weight, teacher_lambda, kd_lambda))

        if score_train['iou_1']>bestiouT:
            bestiouT = score_train['iou_1']
            for pred, label, name in zip(preds_all, labels_all, names_all):
                pred = pred.astype(np.uint8)*255
                label = label.astype(np.uint8)*255
                vis_img = np.concatenate([pred, label], axis=1)
                io.imsave(os.path.join(args['pred_dir'], "train_"+name), vis_img)
        score_val,loss_val, val_preds, val_labels, val_names = validate(val_loader, net, criterion, curr_epoch,args)
        # if acc_meter.avg>bestaccT: bestaccT=acc_meter.avg
        if score_val['iou_1']>bestiou:
            bestiou = score_val['iou_1']
            bestloss = loss_val
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME+'_%dIoU%.2f'\
                %(curr_epoch, score_val['iou_1']*100)))
            for pred, label, name in zip(val_preds, val_labels, val_names):
                pred = pred.astype(np.uint8)*255
                label = label.astype(np.uint8)*255
                vis_img = np.concatenate([pred, label], axis=1)
                io.imsave(os.path.join(args['pred_dir'], "val_"+name), vis_img)
        with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            f.write('Epoch: %d  Total time: %.1fs  Val iou %.2f  loss %.4f\n' %(curr_epoch, time.time()-begin_time, bestiou*100, bestloss))
        print('Epoch: %d  Total time: %.1fs  Val iou %.2f  loss %.4f' %(curr_epoch, time.time()-begin_time, bestiou*100, bestloss))
        curr_epoch += 1
        #scheduler.step()
        if curr_epoch >= args['epochs']:
            return

def validate(val_loader, net, criterion, curr_epoch, args):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()


    loss_weights = args.get('loss_weights', DEFAULT_LOSS_WEIGHTS)

    preds_all = []
    labels_all = []
    names_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, imgs_C, labels, names = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            imgs_C = imgs_C.cuda().float()
            labels = labels.cuda().long()
        cls_weights = _make_class_weights(loss_weights, labels.device)
        with torch.no_grad():
            # 验证时可能返回2个或4个值
            forward_result = net(imgs_A, imgs_B, imgs_C)
            
            if len(forward_result) == 4:
                # 师生模式：使用学生预测进行验证
                out_change, student_features, teacher_pred, teacher_features = forward_result
            else:
                # 仅学生模式
                out_change, features = forward_result
                
            out_change = FF.interpolate(out_change, size=(512,512), mode='bilinear', align_corners=True)
            loss_bn = CE_Loss(out_change, labels, cls_weights)
            loss = loss_bn
            
        val_loss.update(loss.cpu().detach().numpy())
        
        preds = torch.argmax(out_change, dim=1)
        pred_numpy = preds.cpu().numpy()
        labels_numpy = labels.cpu().numpy()
        preds_all.append(pred_numpy)
        labels_all.append(labels_numpy)
        names_all.extend(names)
        # if curr_epoch%args['predict_step']==0:
        #     # print(preds.shape)
        #     for pred, label, name in zip(preds, labels, names):
        #         pred = pred.cpu().numpy().astype(np.uint8)*255
        #         label = label.cpu().numpy().astype(np.uint8)*255
        #         vis_img = np.concatenate([pred, label], axis=1)
        #         io.imsave(os.path.join(args['pred_dir'], "val_"+name), vis_img)
        #         # io.imsave(os.path.join(args['pred_dir'], NET_NAME+'.png'), pred.cpu().numpy()*255)
        #     print('Prediction saved!')
    
    # Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 2)
    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    hist = get_confuse_matrix(2,labels_all,preds_all)
    score = cm2score(hist)
    curr_time = time.time() - start
    with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
        f.write('Epoch: %d  %.1fs Val loss: %.2f  score: %s\n'\
            %(curr_epoch, curr_time, val_loss.average(),{key: score[key] for key in score}))
    print('Epoch: %d  %.1fs Val loss: %.2f  score: %s'\
    %(curr_epoch, curr_time, val_loss.average(),{key: score[key] for key in score}))

    return score, val_loss.average(), preds_all, labels_all, names_all

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()            

def adjust_lr(optimizer, curr_iter, all_iter, args):
    init_lr=args['lr']
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr
