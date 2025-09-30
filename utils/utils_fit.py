"""
训练循环与验证流程（含 AMP、BF16 优先与 non_blocking/channels_last 优化）。
"""

import os
import time
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io

from utils.loss import (
    CE_Loss,
    AlignmentLoss,
    Dice_loss,
    FeatureConsistencyLoss,
    MaskedFeatureMSELoss,
    DifferenceAttentionDistillationLoss,
)
from utils.utils import AverageMeter, get_confuse_matrix, cm2score


warnings.filterwarnings("ignore", message=".*low contrast image.*")


# 断点保存工具函数
def save_checkpoint(
    chkpt_dir: str,
    net: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_iou: float,
    best_loss: float,
    net_name: str,
    is_best: bool = False,
) -> None:
    """
    保存训练断点与最佳模型。

    - 始终写入 last_checkpoint.pth（包含模型、优化器、调度器、AMP 状态等）。
    - 当 is_best=True 时，同时写入 best_checkpoint.pth 与带 IoU 的模型权重（仅 state_dict，后缀 .pth）。
    """
    os.makedirs(chkpt_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'best_iou': float(best_iou),
        'best_loss': float(best_loss),
        'model_state': net.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state': scaler.state_dict() if scaler is not None else None,
    }

    last_path = os.path.join(chkpt_dir, 'last_checkpoint.pth')
    torch.save(checkpoint, last_path)

    if is_best:
        best_ckpt_path = os.path.join(chkpt_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_ckpt_path)
        # 仅保存权重（兼容推理脚本）
        best_sd_path = os.path.join(
            chkpt_dir,
            f"{net_name}_{epoch}IoU{best_iou * 100:.2f}.pth",
        )
        torch.save(net.state_dict(), best_sd_path)

# 损失权重配置
DEFAULT_LOSS_WEIGHTS = {
    'kd_weight': 5e-3,
    'kd_warmup_epochs': 12,
    'kd_loss_clip': 50.0,
    'teacher_weight': 0.12,
    'teacher_warmup_epochs': 8,
    'align_scale_student': 0.3,
    'align_scale_teacher': 0.2,
    'align_base_weight': 0.5,
    'align_reg_scale': 0.3,
    'temperature': 8.0,
    'ce_class_weights': (1.0, 1.0),
    'feat_kd_weight': 0.5,
    'feat_kd_pos': 3.0,
    'feat_kd_neg': 1.0,
    'attD_enable': True,
    'attD_weight': 0.5,
    'attD_map_w': 0.6,
    'attD_ch_w': 0.25,
    'attD_sp_w': 0.15,
    'attD_warmup_epochs': 12,
}


def _compute_kd_weight(epoch: int, loss_weights: dict) -> float:
    warmup = loss_weights.get('kd_warmup_epochs', 0)
    if warmup and warmup > 0:
        scale = min(1.0, (epoch + 1) / float(warmup))
    else:
        scale = 1.0
    return loss_weights['kd_weight'] * scale


def _compute_teacher_weight(epoch: int, loss_weights: dict) -> float:
    warmup = loss_weights.get('teacher_warmup_epochs', 0)
    base = loss_weights.get('teacher_weight', 0.0)
    if warmup and warmup > 0:
        scale = min(1.0, (epoch + 1) / float(warmup))
    else:
        scale = 1.0
    return base * scale


def _make_class_weights(loss_weights: dict, device: torch.device) -> torch.Tensor:
    weights = loss_weights.get('ce_class_weights', (1.0, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(train_loader, train_loader_unchange, net, criterion, optimizer, scheduler, val_loader, args):
    NET_NAME = args['net_name']
    bestiouT = 0.0
    bestiou = 0.0
    bestloss = 1.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])

    al_loss = AlignmentLoss()
    feat_kd_loss = MaskedFeatureMSELoss()
    fc_loss = FeatureConsistencyLoss()
    att_distill_loss = DifferenceAttentionDistillationLoss()

    loss_weights = copy.deepcopy(DEFAULT_LOSS_WEIGHTS)
    loss_weights.update(args.get('loss_weights', {}))
    args['loss_weights'] = loss_weights

    # 支持从外部指定的起始 epoch（用于断点续训）
    curr_epoch = int(args.get('start_epoch', 0))
    if curr_epoch == 0:
        with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            f.write('=' * 80 + '\n')
            f.write('损失权重配置:\n')
            for key, value in loss_weights.items():
                f.write(f'  {key}: {value}\n')
            f.write('=' * 80 + '\n\n')

    # AMP：优先 BF16，不支持则使用 FP16 + GradScaler
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda') if not use_bf16 else None
    # 恢复 AMP Scaler 状态（如存在）
    scaler_state = args.get('scaler_state', None)
    if scaler is not None and scaler_state is not None:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception:
            pass

    while True:
        net.train()
        start = time.time()
        train_loss = AverageMeter()
        train_loss_ce = AverageMeter()
        train_loss_teacher = AverageMeter()
        train_loss_kd = AverageMeter()
        train_loss_align_student = AverageMeter()
        train_loss_align_teacher = AverageMeter()
        train_loss_dice = AverageMeter()
        train_loss_feat_kd = AverageMeter()
        train_loss_attD_map = AverageMeter()
        train_loss_attD_sp = AverageMeter()
        train_loss_attD_ch = AverageMeter()
        train_loss_teacher_dice = AverageMeter()
        train_loss_teacher_ce = AverageMeter()
        
        # 计算当前epoch的attention distillation权重（用于日志输出）
        epoch_attD_weight = 0.0
        if loss_weights.get('attD_enable', True):
            warmup_att = loss_weights.get('attD_warmup_epochs', 0)
            if warmup_att and warmup_att > 0:
                epoch_attD_weight = loss_weights['attD_weight'] * min(1.0, (curr_epoch + 1) / float(warmup_att))
            else:
                epoch_attD_weight = loss_weights['attD_weight']

        curr_iter = curr_epoch * len(train_loader)
        kd_lambda = _compute_kd_weight(curr_epoch, loss_weights)
        teacher_lambda = _compute_teacher_weight(curr_epoch, loss_weights)
        epoch_align_weight = (1.0 / (curr_epoch + 1)) * loss_weights['align_base_weight']

        preds_all = []
        labels_all = []
        names_all = []

        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            if not args.get('use_onecycle', False):
                adjust_lr(optimizer, running_iter, all_iters, args)
            imgs_A, imgs_B, imgs_C, labels, names = data

            if args['gpu']:
                imgs_A = imgs_A.cuda(non_blocking=True)
                imgs_B = imgs_B.cuda(non_blocking=True)
                imgs_C = imgs_C.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True).long()
                imgs_A = imgs_A.contiguous(memory_format=torch.channels_last)
                imgs_B = imgs_B.contiguous(memory_format=torch.channels_last)
                imgs_C = imgs_C.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if use_bf16 else torch.float16):
                forward_result = net(imgs_A, imgs_B, imgs_C)

                if len(forward_result) == 4:
                    out_change, student_features, teacher_pred, teacher_features = forward_result
                    out_change = F.interpolate(out_change, size=(512, 512), mode='bilinear', align_corners=True)
                    teacher_pred = F.interpolate(teacher_pred, size=(512, 512), mode='bilinear', align_corners=True)

                    cls_weights = _make_class_weights(loss_weights, labels.device)
                    loss_ce = CE_Loss(out_change, labels, cls_weights=cls_weights)
                    loss_teacher = CE_Loss(teacher_pred, labels, cls_weights=cls_weights)
                    teacher_ce_val = loss_teacher.item()

                    align_loss_student = AlignmentLoss()(student_features)
                    align_loss_teacher = AlignmentLoss()(teacher_features)

                    change_mask = (labels > 0).long()
                    feat_kd = feat_kd_loss(
                        student_features,
                        teacher_features,
                        change_mask,
                        pos_weight=loss_weights.get('feat_kd_pos', 3.0),
                        neg_weight=loss_weights.get('feat_kd_neg', 1.0),
                    )

                    temperature = loss_weights['temperature']
                    student_log_prob = F.log_softmax(out_change / temperature, dim=1)
                    teacher_prob = F.softmax(teacher_pred.detach() / temperature, dim=1)
                    kd_loss_map = F.kl_div(student_log_prob, teacher_prob, reduction='none')
                    kd_loss = kd_loss_map.sum(dim=1).mean() * (temperature ** 2)
                    kd_clip = loss_weights.get('kd_loss_clip')
                    if kd_clip is not None:
                        kd_loss = torch.clamp(kd_loss, max=kd_clip)

                    dice_loss_val = 0.0
                    if args['dice']:
                        loss_dice_stu = Dice_loss(out_change, labels)
                        loss_dice_tea = Dice_loss(teacher_pred, labels)
                        loss_ce = loss_ce + loss_dice_stu
                        loss_teacher = loss_teacher + loss_dice_tea
                        dice_loss_val = loss_dice_stu.item()
                        train_loss_teacher_dice.update(loss_dice_tea.item())
                    else:
                        train_loss_teacher_dice.update(0.0)

                    # 差异图注意力蒸馏
                    attD_map = 0.0
                    attD_sp = 0.0
                    attD_ch = 0.0
                    if loss_weights.get('attD_enable', True) and epoch_attD_weight > 0.0:
                        attD_map, attD_sp, attD_ch = att_distill_loss(student_features, teacher_features, teacher_pred)
                        attD_total = (
                            loss_weights.get('attD_map_w', 0.5) * attD_map
                            + loss_weights.get('attD_ch_w', 0.3) * attD_ch
                            + loss_weights.get('attD_sp_w', 0.2) * attD_sp
                        )
                    else:
                        attD_total = 0.0

                    loss = (
                        loss_ce
                        + teacher_lambda * loss_teacher
                        + kd_lambda * kd_loss
                        + loss_weights.get('feat_kd_weight', 0.5) * feat_kd
                        + epoch_align_weight * loss_weights['align_reg_scale'] * (
                            loss_weights['align_scale_student'] * align_loss_student
                            + loss_weights['align_scale_teacher'] * align_loss_teacher
                        )
                        + epoch_attD_weight * attD_total
                    )

                    train_loss_ce.update(loss_ce.item())
                    train_loss_teacher.update(loss_teacher.item())
                    train_loss_teacher_ce.update(teacher_ce_val)
                    train_loss_kd.update(kd_loss.item())
                    train_loss_align_student.update(align_loss_student.item())
                    train_loss_align_teacher.update(align_loss_teacher.item())
                    train_loss_feat_kd.update(feat_kd.item())
                    if loss_weights.get('attD_enable', True):
                        train_loss_attD_map.update(float(attD_map.detach()))
                        train_loss_attD_sp.update(float(attD_sp.detach()))
                        train_loss_attD_ch.update(float(attD_ch.detach()))
                    if args['dice']:
                        train_loss_dice.update(dice_loss_val)

                else:
                    out_change, features = forward_result
                    out_change = F.interpolate(out_change, size=(512, 512), mode='bilinear', align_corners=True)

                    cls_weights = _make_class_weights(loss_weights, labels.device)
                    loss_ce = CE_Loss(out_change, labels, cls_weights=cls_weights)
                    align_loss = AlignmentLoss()(features)

                    dice_loss_val = 0.0
                    if args['dice']:
                        loss_dice = Dice_loss(out_change, labels)
                        loss = loss_ce + loss_dice
                        dice_loss_val = loss_dice.item()
                    else:
                        loss = loss_ce
                    loss = loss + epoch_align_weight * align_loss

                    train_loss_ce.update(loss_ce.item())
                    train_loss_align_student.update(align_loss.item())
                    if args['dice']:
                        train_loss_dice.update(dice_loss_val)
                    train_loss_teacher.update(0.0)
                    train_loss_kd.update(0.0)
                    train_loss_align_teacher.update(0.0)
                    train_loss_feat_kd.update(0.0)
                    train_loss_attD_map.update(0.0)
                    train_loss_attD_sp.update(0.0)
                    train_loss_attD_ch.update(0.0)
                    train_loss_teacher_dice.update(0.0)
                    train_loss_teacher_ce.update(0.0)

            if scaler is not None:
                scaler.scale(loss).backward()
                # 梯度裁剪（与 AMP 兼容）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
            # OneCycleLR 每步更新
            if args.get('use_onecycle', False):
                try:
                    scheduler.step()
                except Exception:
                    pass

            preds = torch.argmax(out_change, dim=1)
            pred_numpy = preds.cpu().numpy()
            labels_numpy = labels.cpu().numpy()
            preds_all.append(pred_numpy)
            labels_all.append(labels_numpy)
            names_all.extend(names)
            # 使用 float32 标量，避免 BF16/FP16 -> numpy 转换问题
            train_loss.update(loss.detach().float().cpu().item())

        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        score_train = cm2score(get_confuse_matrix(2, labels_all, preds_all))

        epoch_align_weight = (1.0 / (curr_epoch + 1)) * loss_weights['align_base_weight']
        with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            f.write('Epoch: %d  Total time: %.1fs  Train loss %.4f  score %s\n' % (
                curr_epoch, time.time() - begin_time, train_loss.average(), {k: score_train[k] for k in score_train}))
            f.write('  Detailed losses: CE=%.4f, Teacher=%.4f (CE=%.4f, Dice=%.4f), KD=%.4f, FeatKD=%.4f, AlignS=%.4f, AlignT=%.4f' % (
                train_loss_ce.average(), train_loss_teacher.average(), train_loss_teacher_ce.average(), train_loss_teacher_dice.average(), train_loss_kd.average(),
                train_loss_feat_kd.average(), train_loss_align_student.average(), train_loss_align_teacher.average()))
            if args['dice']:
                f.write(', Dice=%.4f' % train_loss_dice.average())
            if loss_weights.get('attD_enable', True):
                f.write(', AttD_map=%.4f, AttD_sp=%.4f, AttD_ch=%.4f' % (
                    train_loss_attD_map.average(), train_loss_attD_sp.average(), train_loss_attD_ch.average()))
            f.write(', Epoch_align_weight=%.4f, Epoch_teacher_weight=%.6f, Epoch_kd_weight=%.6f' % (
                epoch_align_weight, teacher_lambda, kd_lambda))
            if loss_weights.get('attD_enable', True):
                f.write(', Epoch_attD_weight=%.6f' % epoch_attD_weight)
            f.write('\n')

        print('Epoch: %d  Total time: %.1fs  Train loss %.4f  score %s' % (
            curr_epoch, time.time() - begin_time, train_loss.average(), {k: score_train[k] for k in score_train}))
        print('  Detailed losses: CE=%.4f, Teacher=%.4f (CE=%.4f, Dice=%.4f), KD=%.4f, FeatKD=%.4f, AlignS=%.4f, AlignT=%.4f' % (
            train_loss_ce.average(), train_loss_teacher.average(), train_loss_teacher_ce.average(), train_loss_teacher_dice.average(), train_loss_kd.average(),
            train_loss_feat_kd.average(), train_loss_align_student.average(), train_loss_align_teacher.average()), end='')
        if args['dice']:
            print(', Dice=%.4f' % train_loss_dice.average(), end='')
        if loss_weights.get('attD_enable', True):
            print(', AttD_map=%0.4f, AttD_sp=%0.4f, AttD_ch=%0.4f' % (
                train_loss_attD_map.average(), train_loss_attD_sp.average(), train_loss_attD_ch.average()), end='')
        print(', Epoch_align_weight=%.4f, Epoch_teacher_weight=%.6f, Epoch_kd_weight=%.6f' % (
            epoch_align_weight, teacher_lambda, kd_lambda))

        if score_train['iou_1'] > bestiouT:
            bestiouT = score_train['iou_1']
            for pred, label, name in zip(preds_all, labels_all, names_all):
                pred = pred.astype(np.uint8) * 255
                label = label.astype(np.uint8) * 255
                vis_img = np.concatenate([pred, label], axis=1)
                io.imsave(os.path.join(args['pred_dir'], "train_" + name), vis_img)

        score_val, loss_val, val_preds, val_labels, val_names = validate(val_loader, net, criterion, curr_epoch, args)
        # 判断是否刷新最佳指标
        improved = score_val['iou_1'] > bestiou
        if improved:
            bestiou = score_val['iou_1']
            bestloss = loss_val
            for pred, label, name in zip(val_preds, val_labels, val_names):
                pred = pred.astype(np.uint8) * 255
                label = label.astype(np.uint8) * 255
                vis_img = np.concatenate([pred, label], axis=1)
                io.imsave(os.path.join(args['pred_dir'], "val_" + name), vis_img)

        # 保存 last_checkpoint（以及可能的最佳权重）
        try:
            save_checkpoint(
                chkpt_dir=args['chkpt_dir'],
                net=net,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=curr_epoch,
                best_iou=bestiou,
                best_loss=bestloss,
                net_name=NET_NAME,
                is_best=improved,
            )
        except Exception:
            # 保存失败不影响训练继续
            pass

        with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            f.write('Epoch: %d  Total time: %.1fs  Val iou %.2f  loss %.4f\n' % (
                curr_epoch, time.time() - begin_time, bestiou * 100, bestloss))
        print('Epoch: %d  Total time: %.1fs  Val iou %.2f  loss %.4f' % (
            curr_epoch, time.time() - begin_time, bestiou * 100, bestloss))

        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch, args):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()

    preds_all = []
    labels_all = []
    names_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, imgs_C, labels, names = data
        if args['gpu']:
            imgs_A = imgs_A.cuda(non_blocking=True)
            imgs_B = imgs_B.cuda(non_blocking=True)
            imgs_C = imgs_C.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True).long()
            imgs_A = imgs_A.contiguous(memory_format=torch.channels_last)
            imgs_B = imgs_B.contiguous(memory_format=torch.channels_last)
            imgs_C = imgs_C.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode():
            # 验证也使用 AMP（与训练一致）
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if use_bf16 else torch.float16):
                forward_result = net(imgs_A, imgs_B, imgs_C)
            if len(forward_result) == 4:
                out_change, student_features, teacher_pred, teacher_features = forward_result
            else:
                out_change, features = forward_result

            out_change = F.interpolate(out_change, size=(512, 512), mode='bilinear', align_corners=True)
            loss_bn = CE_Loss(out_change, labels, _make_class_weights(DEFAULT_LOSS_WEIGHTS, labels.device))
            loss = loss_bn

        # 使用 float32 标量，避免 BF16/FP16 -> numpy 转换问题
        val_loss.update(loss.detach().float().cpu().item())

        preds = torch.argmax(out_change, dim=1)
        pred_numpy = preds.cpu().numpy()
        labels_numpy = labels.cpu().numpy()
        preds_all.append(pred_numpy)
        labels_all.append(labels_numpy)
        names_all.extend(names)

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    hist = get_confuse_matrix(2, labels_all, preds_all)
    score = cm2score(hist)

    curr_time = time.time() - start
    with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
        f.write('Epoch: %d  %.1fs Val loss: %.2f  score: %s\n' % (
            curr_epoch, curr_time, val_loss.average(), {k: score[k] for k in score}))
    print('Epoch: %d  %.1fs Val loss: %.2f  score: %s' % (
        curr_epoch, curr_time, val_loss.average(), {k: score[k] for k in score}))

    return score, val_loss.average(), preds_all, labels_all, names_all


def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, args):
    init_lr = args['lr']
    scale_running_lr = ((1.0 - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr



