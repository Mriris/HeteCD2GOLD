"""
配置对比工具
生成不同权重配置之间的对比表格
"""

from configs_weight_ablation import ALL_CONFIGS


def print_comparison_table():
    """打印权重配置对比表"""
    
    # 定义要对比的权重参数
    weight_keys = [
        'kd_weight',
        'kd_warmup_epochs',
        'teacher_weight',
        'teacher_warmup_epochs',
        'feat_kd_weight',
        'feat_kd_pos',
        'feat_kd_neg',
        'attD_weight',
        'attD_map_w',
        'attD_ch_w',
        'attD_sp_w',
        'attD_warmup_epochs',
        'temperature',
        'align_base_weight',
    ]
    
    configs = ['baseline', 'config_A', 'config_B', 'config_C']
    
    print("\n" + "=" * 140)
    print("权重配置对比表")
    print("=" * 140)
    print()
    
    # 打印表头
    header = f"{'参数名':<30}"
    for config_name in configs:
        config = ALL_CONFIGS[config_name]
        display_name = config['name'][:20]
        header += f" | {display_name:>20}"
    print(header)
    print("-" * 140)
    
    # 打印每个参数的对比
    for key in weight_keys:
        row = f"{key:<30}"
        baseline_value = ALL_CONFIGS['baseline']['loss_weights'].get(key, 'N/A')
        
        for config_name in configs:
            config = ALL_CONFIGS[config_name]
            value = config['loss_weights'].get(key, 'N/A')
            
            # 计算变化百分比
            if config_name != 'baseline' and isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
                if baseline_value != 0:
                    change_pct = ((value - baseline_value) / baseline_value) * 100
                    if abs(change_pct) > 0.1:  # 有明显变化
                        if change_pct > 0:
                            display_value = f"{value} (+{change_pct:.0f}%)"
                        else:
                            display_value = f"{value} ({change_pct:.0f}%)"
                    else:
                        display_value = str(value)
                else:
                    display_value = str(value)
            else:
                display_value = str(value)
            
            row += f" | {display_value:>20}"
        
        print(row)
    
    print("-" * 140)
    print()
    
    # 打印预期损失占比
    print("预期损失占比:")
    print("-" * 140)
    for config_name in configs:
        config = ALL_CONFIGS[config_name]
        print(f"  {config['name']:<30}: {config['expected_loss_ratio']}")
    print()
    
    # 打印关键变化总结
    print("=" * 140)
    print("关键变化总结:")
    print("=" * 140)
    print()
    
    for config_name in ['config_A', 'config_B', 'config_C']:
        config = ALL_CONFIGS[config_name]
        print(f"【{config['name']}】")
        print(f"  描述: {config['description']}")
        if 'key_changes' in config:
            print(f"  变化:")
            for change in config['key_changes']:
                print(f"    • {change}")
        print()
    
    print("=" * 140)


def print_simplified_table():
    """打印简化版对比表（仅显示有变化的参数）"""
    
    configs = ['baseline', 'config_A', 'config_B', 'config_C']
    baseline_weights = ALL_CONFIGS['baseline']['loss_weights']
    
    # 找出有变化的参数
    changed_keys = set()
    for config_name in ['config_A', 'config_B', 'config_C']:
        config_weights = ALL_CONFIGS[config_name]['loss_weights']
        for key, value in config_weights.items():
            if baseline_weights.get(key) != value:
                changed_keys.add(key)
    
    changed_keys = sorted(changed_keys)
    
    print("\n" + "=" * 100)
    print("简化对比表（仅显示变化的参数）")
    print("=" * 100)
    print()
    
    # 打印表头
    header = f"{'参数名':<25}"
    for config_name in configs:
        config = ALL_CONFIGS[config_name]
        display_name = config['name'][:15]
        header += f" | {display_name:>15}"
    print(header)
    print("-" * 100)
    
    # 打印有变化的参数
    for key in changed_keys:
        row = f"{key:<25}"
        baseline_value = baseline_weights.get(key, 'N/A')
        
        for config_name in configs:
            config = ALL_CONFIGS[config_name]
            value = config['loss_weights'].get(key, 'N/A')
            
            # 标记变化
            if config_name != 'baseline' and value != baseline_value:
                if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
                    if baseline_value != 0:
                        change_pct = ((value - baseline_value) / baseline_value) * 100
                        display_value = f"{value} ({change_pct:+.0f}%)"
                    else:
                        display_value = str(value)
                else:
                    display_value = f"{value} ✓"
            else:
                display_value = str(value)
            
            row += f" | {display_value:>15}"
        
        print(row)
    
    print("-" * 100)
    print()


def export_to_csv():
    """导出对比表到CSV文件"""
    import csv
    
    configs = ['baseline', 'config_A', 'config_B', 'config_C']
    
    # 收集所有权重参数
    all_keys = set()
    for config_name in configs:
        all_keys.update(ALL_CONFIGS[config_name]['loss_weights'].keys())
    all_keys = sorted(all_keys)
    
    # 写入CSV
    with open('weight_configs_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ['参数名'] + [ALL_CONFIGS[c]['name'] for c in configs]
        writer.writerow(header)
        
        # 写入数据
        for key in all_keys:
            row = [key]
            for config_name in configs:
                value = ALL_CONFIGS[config_name]['loss_weights'].get(key, 'N/A')
                row.append(value)
            writer.writerow(row)
        
        # 写入预期损失占比
        writer.writerow([])
        writer.writerow(['预期损失占比'])
        for config_name in configs:
            writer.writerow([ALL_CONFIGS[config_name]['name'], 
                           ALL_CONFIGS[config_name]['expected_loss_ratio']])
    
    print("✓ 对比表已导出到: weight_configs_comparison.csv")


def main():
    """主函数"""
    print("\n超参占比配置对比工具")
    print("=" * 100)
    
    while True:
        print("\n请选择操作:")
        print("  1. 查看完整对比表")
        print("  2. 查看简化对比表（仅显示变化的参数）")
        print("  3. 导出到CSV文件")
        print("  4. 查看所有配置详情")
        print("  0. 退出")
        print()
        
        choice = input("请输入选项 (0-4): ").strip()
        
        if choice == '1':
            print_comparison_table()
        elif choice == '2':
            print_simplified_table()
        elif choice == '3':
            export_to_csv()
        elif choice == '4':
            from configs_weight_ablation import print_config_summary
            print_config_summary()
        elif choice == '0':
            print("\n再见！")
            break
        else:
            print("\n无效选项，请重试")


if __name__ == '__main__':
    # 如果直接运行，显示简化表
    print_simplified_table()
    print("\n提示: 运行 python compare_configs.py 进入交互模式查看更多选项")

