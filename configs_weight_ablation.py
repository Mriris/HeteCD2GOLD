"""
超参占比探索实验配置
用于对比不同权重组合对师生在线蒸馏和差异图注意力迁移机制的影响
"""

# ================================================================================
# Baseline配置（当前最佳 - 仅作参考，不训练）
# ================================================================================
BASELINE = {
    'name': 'baseline',
    'description': '当前最佳配置（参考）',
    'loss_weights': {
        'kd_weight': 0.005,
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
    },
    'expected_loss_ratio': 'CE:89% | Teacher:11% | KD:2%',
}

# ================================================================================
# 配置A：强化师生蒸馏占比
# ================================================================================
CONFIG_A = {
    'name': 'config_A_enhanced_kd',
    'description': '强化KD占比（2%→5%），提升知识蒸馏贡献',
    'loss_weights': {
        'kd_weight': 0.012,              # +140% (使KD占比达到5%)
        'kd_warmup_epochs': 12,
        'kd_loss_clip': 50.0,
        'teacher_weight': 0.12,          # 保持不变
        'teacher_warmup_epochs': 8,
        'align_scale_student': 0.3,
        'align_scale_teacher': 0.2,
        'align_base_weight': 0.5,
        'align_reg_scale': 0.3,
        'temperature': 8.0,
        'ce_class_weights': (1.0, 1.0),
        'feat_kd_weight': 0.5,           # 保持不变
        'feat_kd_pos': 3.0,
        'feat_kd_neg': 1.0,
        'attD_enable': True,
        'attD_weight': 0.5,              # 保持不变
        'attD_map_w': 0.6,
        'attD_ch_w': 0.25,
        'attD_sp_w': 0.15,
        'attD_warmup_epochs': 12,
    },
    'expected_loss_ratio': 'CE:85% | Teacher:10% | KD:5%',
    'key_changes': [
        'kd_weight: 0.005 → 0.012 (+140%)',
        '目标：测试提升知识蒸馏权重对收敛速度和最终性能的影响',
    ],
}

# ================================================================================
# 配置B：强化差异图注意力机制
# ================================================================================
CONFIG_B = {
    'name': 'config_B_enhanced_attD',
    'description': '强化AttD权重并延长预热期，突出差异图注意力贡献',
    'loss_weights': {
        'kd_weight': 0.005,              # 保持不变
        'kd_warmup_epochs': 12,
        'kd_loss_clip': 50.0,
        'teacher_weight': 0.12,          # 保持不变
        'teacher_warmup_epochs': 8,
        'align_scale_student': 0.3,
        'align_scale_teacher': 0.2,
        'align_base_weight': 0.5,
        'align_reg_scale': 0.3,
        'temperature': 8.0,
        'ce_class_weights': (1.0, 1.0),
        'feat_kd_weight': 0.6,           # +20% 强化特征蒸馏
        'feat_kd_pos': 3.0,
        'feat_kd_neg': 1.0,
        'attD_enable': True,
        'attD_weight': 0.8,              # +60% 大幅提升AttD权重
        'attD_map_w': 0.65,              # +8% 进一步强化差异图
        'attD_ch_w': 0.25,               # 保持不变
        'attD_sp_w': 0.10,               # -33% 降低空间注意力
        'attD_warmup_epochs': 20,        # 延长预热期（12→20）
    },
    'expected_loss_ratio': 'CE:85% | Teacher:10% | KD:2% | AttD中期占比更高',
    'key_changes': [
        'attD_weight: 0.5 → 0.8 (+60%)',
        'attD_map_w: 0.6 → 0.65 (+8%)',
        'attD_sp_w: 0.15 → 0.10 (-33%)',
        'attD_warmup_epochs: 12 → 20',
        'feat_kd_weight: 0.5 → 0.6 (+20%)',
        '目标：测试差异图注意力迁移机制在更长训练周期中的持续贡献',
    ],
}

# ================================================================================
# 配置C：均衡师生蒸馏与教师监督
# ================================================================================
CONFIG_C = {
    'name': 'config_C_balanced',
    'description': '均衡Teacher和KD权重（各占8%），三分权重策略',
    'loss_weights': {
        'kd_weight': 0.010,              # +100% (使KD占比达到8%)
        'kd_warmup_epochs': 12,
        'kd_loss_clip': 50.0,
        'teacher_weight': 0.10,          # -17% (降低Teacher占比到8%)
        'teacher_warmup_epochs': 8,
        'align_scale_student': 0.3,
        'align_scale_teacher': 0.2,
        'align_base_weight': 0.5,
        'align_reg_scale': 0.3,
        'temperature': 8.0,
        'ce_class_weights': (1.0, 1.0),
        'feat_kd_weight': 0.5,           # 保持不变
        'feat_kd_pos': 3.0,
        'feat_kd_neg': 1.0,
        'attD_enable': True,
        'attD_weight': 0.5,              # 保持不变
        'attD_map_w': 0.6,
        'attD_ch_w': 0.25,
        'attD_sp_w': 0.15,
        'attD_warmup_epochs': 12,
    },
    'expected_loss_ratio': 'CE:84% | Teacher:8% | KD:8%',
    'key_changes': [
        'kd_weight: 0.005 → 0.010 (+100%)',
        'teacher_weight: 0.12 → 0.10 (-17%)',
        '目标：测试师生蒸馏与教师监督均衡时的协同效果',
    ],
}

# ================================================================================
# 所有配置汇总
# ================================================================================
ALL_CONFIGS = {
    'baseline': BASELINE,
    'config_A': CONFIG_A,
    'config_B': CONFIG_B,
    'config_C': CONFIG_C,
}

# 实验运行顺序（不包含baseline）
EXPERIMENT_CONFIGS = ['config_A', 'config_B', 'config_C']


def get_config(config_name):
    """获取指定配置"""
    if config_name not in ALL_CONFIGS:
        raise ValueError(f"未知配置: {config_name}。可用配置: {list(ALL_CONFIGS.keys())}")
    return ALL_CONFIGS[config_name]


def print_config_summary():
    """打印所有配置的对比摘要"""
    print("=" * 100)
    print("超参占比探索实验配置总览")
    print("=" * 100)
    print()
    
    for config_name in ['baseline', 'config_A', 'config_B', 'config_C']:
        config = ALL_CONFIGS[config_name]
        print(f"【{config['name']}】")
        print(f"  描述: {config['description']}")
        print(f"  预期占比: {config['expected_loss_ratio']}")
        
        if 'key_changes' in config:
            print(f"  关键变化:")
            for change in config['key_changes']:
                print(f"    - {change}")
        
        print()
    
    print("=" * 100)
    print(f"待运行实验: {EXPERIMENT_CONFIGS}")
    print("=" * 100)


if __name__ == '__main__':
    print_config_summary()

