"""
预测结果拼接脚本（拼接回源图像）

功能：
根据 tiles_meta.json 将预测的切片拼接回各个源图像对应的大图
支持拼接 pred、label、error 三种结果
"""

import os
import json
from typing import Dict, List
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# ==================== 配置参数 ====================
# 输入输出路径
OUTPUT_ROOT = r"D:\0Program\Datasets\241120\Compare\Datas\AIO12"
PREDICT_OUTPUT_DIR = r"D:\0Program\HeteCD2GOLD\results\AIO12"

# 拼接参数
STITCH_TYPES = ['pred', 'label', 'error']  # 要拼接的类型
# ==================================================


def load_metadata(metadata_path: str) -> Dict:
    """加载元数据"""
    print(f"加载元数据: {metadata_path}")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"  版本: {metadata.get('version', 'unknown')}")
    print(f"  切片大小: {metadata.get('tile_size', 'unknown')}")
    print(f"  源图像数: {len(metadata.get('images', []))}")
    print(f"  切片数量: {len(metadata.get('tiles', []))}")
    
    return metadata


def stitch_image(image_info: Dict, tiles_info: List[Dict], 
                 predict_dir: str, output_dir: str, stitch_types: List[str]):
    """拼接单个源图像的预测结果"""
    base_name = image_info['base_name']
    img_width, img_height = image_info['size']
    
    print(f"\n拼接源图像: {base_name}")
    print(f"  图像尺寸: {img_width}x{img_height}")
    print(f"  切片数量: {len(tiles_info)}")
    
    # 为每种类型创建画布
    canvases = {}
    
    if 'pred' in stitch_types:
        canvases['pred'] = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if 'label' in stitch_types:
        canvases['label'] = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if 'error' in stitch_types:
        canvases['error'] = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # 统计
    stats = {t: {'success': 0, 'missing': 0, 'error': 0} for t in stitch_types}
    
    # 处理每个切片
    for tile_info in tiles_info:
        tile_name = tile_info['name']
        x1, y1, x2, y2 = tile_info['pixel_bounds']
        actual_w, actual_h = tile_info['actual_size']
        
        # 拼接 pred
        if 'pred' in stitch_types:
            pred_path = os.path.join(predict_dir, f"{tile_name}_pred.png")
            if os.path.exists(pred_path):
                try:
                    pred_img = Image.open(pred_path).convert('L')
                    pred_array = np.array(pred_img)
                    
                    # 提取实际数据区域并放置
                    canvases['pred'][y1:y2, x1:x2] = pred_array[:actual_h, :actual_w]
                    stats['pred']['success'] += 1
                except Exception as e:
                    stats['pred']['error'] += 1
            else:
                stats['pred']['missing'] += 1
        
        # 拼接 label
        if 'label' in stitch_types:
            label_path = os.path.join(predict_dir, f"{tile_name}_label.png")
            if os.path.exists(label_path):
                try:
                    label_img = Image.open(label_path).convert('L')
                    label_array = np.array(label_img)
                    
                    # 提取实际数据区域并放置
                    canvases['label'][y1:y2, x1:x2] = label_array[:actual_h, :actual_w]
                    stats['label']['success'] += 1
                except Exception as e:
                    stats['label']['error'] += 1
            else:
                stats['label']['missing'] += 1
        
        # 拼接 error
        if 'error' in stitch_types:
            error_path = os.path.join(predict_dir, f"{tile_name}_error.png")
            if os.path.exists(error_path):
                try:
                    error_img = cv2.imread(error_path)
                    
                    if error_img is not None:
                        # 提取实际数据区域并放置
                        canvases['error'][y1:y2, x1:x2] = error_img[:actual_h, :actual_w]
                        stats['error']['success'] += 1
                    else:
                        stats['error']['error'] += 1
                except Exception as e:
                    stats['error']['error'] += 1
            else:
                stats['error']['missing'] += 1
    
    # 保存拼接结果
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    if 'pred' in stitch_types and stats['pred']['success'] > 0:
        pred_img = Image.fromarray(canvases['pred'])
        pred_path = os.path.join(output_dir, f"{base_name}_pred.png")
        pred_img.save(pred_path)
        saved_files.append(pred_path)
    
    if 'label' in stitch_types and stats['label']['success'] > 0:
        label_img = Image.fromarray(canvases['label'])
        label_path = os.path.join(output_dir, f"{base_name}_label.png")
        label_img.save(label_path)
        saved_files.append(label_path)
    
    if 'error' in stitch_types and stats['error']['success'] > 0:
        error_path = os.path.join(output_dir, f"{base_name}_error.png")
        cv2.imwrite(error_path, canvases['error'])
        saved_files.append(error_path)
    
    # 打印统计
    print(f"  拼接统计: ", end="")
    stat_strs = []
    for stitch_type in stitch_types:
        s = stats[stitch_type]
        stat_strs.append(f"{stitch_type}={s['success']}/{s['success']+s['missing']+s['error']}")
    print(", ".join(stat_strs))
    
    return saved_files, stats


def main():
    """主函数"""
    print("="*70)
    print("预测结果拼接（拼接回源图像）")
    print("="*70)
    print(f"输出根目录: {OUTPUT_ROOT}")
    print(f"预测输出目录: {PREDICT_OUTPUT_DIR}")
    print(f"拼接类型: {', '.join(STITCH_TYPES)}")
    print("="*70)
    
    # 加载元数据
    metadata_path = os.path.join(OUTPUT_ROOT, 'tiles_meta.json')
    metadata = load_metadata(metadata_path)
    
    images = metadata.get('images', [])
    all_tiles = metadata.get('tiles', [])
    
    if not images:
        raise ValueError("元数据中没有图像信息")
    
    if not all_tiles:
        raise ValueError("元数据中没有切片信息")
    
    # 按源图像分组切片
    image_tiles = defaultdict(list)
    for tile in all_tiles:
        source_image = tile['source_image']
        image_tiles[source_image].append(tile)
    
    # 拼接每个源图像
    stitch_dir = os.path.join(OUTPUT_ROOT, 'stitch')
    all_saved_files = []
    all_stats = []
    
    for image_info in tqdm(images, desc="拼接源图像"):
        base_name = image_info['base_name']
        tiles_info = image_tiles[base_name]
        
        if not tiles_info:
            print(f"\n警告: 源图像 {base_name} 没有切片，跳过")
            continue
        
        saved_files, stats = stitch_image(
            image_info,
            tiles_info,
            PREDICT_OUTPUT_DIR,
            stitch_dir,
            STITCH_TYPES
        )
        
        all_saved_files.extend(saved_files)
        all_stats.append((base_name, stats))
    
    # 打印总结
    print("\n" + "="*70)
    print("拼接完成！")
    print("="*70)
    print(f"处理源图像数: {len(images)}")
    print(f"生成文件数: {len(all_saved_files)}")
    print(f"\n输出目录: {stitch_dir}")
    
    # 统计成功率
    total_stats = {t: {'success': 0, 'total': 0} for t in STITCH_TYPES}
    for _, stats in all_stats:
        for stitch_type in STITCH_TYPES:
            if stitch_type in stats:
                s = stats[stitch_type]
                total_stats[stitch_type]['success'] += s['success']
                total_stats[stitch_type]['total'] += s['success'] + s['missing'] + s['error']
    
    print(f"\n总体统计:")
    for stitch_type in STITCH_TYPES:
        s = total_stats[stitch_type]
        if s['total'] > 0:
            success_rate = s['success'] / s['total'] * 100
            print(f"  {stitch_type}: {s['success']}/{s['total']} ({success_rate:.1f}%)")
    
    print("="*70)


if __name__ == "__main__":
    main()
