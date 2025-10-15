"""
为每个源图像独立生成切片（用于预测）

功能：
1. 为每个源图像独立生成切片（与 preprocess.py 保持一致）
2. 可选生成缩放预览图
3. 输出元数据用于后续拼接回源图像
"""

import os
import glob
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
from tqdm import tqdm
import rasterio
from rasterio.transform import Affine

# ==================== 配置参数 ====================
# 输入输出路径
INPUT_DIR = r"D:\0Program\Datasets\241120\Compare\Datas\Final"
OUTPUT_ROOT = r"D:\0Program\Datasets\241120\Compare\Datas\AIO12"

# 切片参数
TILE_SIZE = 512  # 切片大小
TILE_OVERLAP = 0  # 无重叠

# 预览图参数
OVERVIEW_DOWNSCALE = 0.1  # 预览图缩放比例（0.1表示缩小到原来的10%）
GENERATE_OVERVIEW = True  # 是否生成预览图

# 其他参数
METADATA_VERSION = "2.0"  # 版本2：每个源图像独立切片
# ==================================================


@dataclass
class ImageInfo:
    """源图像信息"""
    base_name: str
    size: Tuple[int, int]  # 像素尺寸 (width, height)
    num_tiles: int  # 切片数量


@dataclass
class TileInfo:
    """切片信息"""
    name: str  # 切片文件名（不含扩展名）
    source_image: str  # 源图像 base_name
    pixel_bounds: Tuple[int, int, int, int]  # 在源图像中的像素边界 (x1, y1, x2, y2)
    actual_size: Tuple[int, int]  # 实际数据尺寸 (width, height)


def process_single_image(base_name: str, files: Dict[str, str], output_dir: str) -> Tuple[ImageInfo, List[TileInfo]]:
    """处理单个源图像，生成切片"""
    
    # 读取图像尺寸
    with rasterio.open(files['A']) as src:
        width = src.width
        height = src.height
    
    # 计算切片位置
    stride = TILE_SIZE - TILE_OVERLAP
    
    col_positions = list(range(0, width, stride))
    row_positions = list(range(0, height, stride))
    
    if not col_positions:
        col_positions = [0]
    if not row_positions:
        row_positions = [0]
    
    num_tiles = len(col_positions) * len(row_positions)
    
    # 创建输出目录
    for img_type in ['A', 'B', 'C', 'label']:
        os.makedirs(os.path.join(output_dir, img_type), exist_ok=True)
    
    has_label = files['label'] is not None
    
    tiles_info = []
    
    # 生成每个切片
    for row_idx, y1 in enumerate(row_positions):
        for col_idx, x1 in enumerate(col_positions):
            # 计算实际数据边界
            x2 = min(x1 + TILE_SIZE, width)
            y2 = min(y1 + TILE_SIZE, height)
            
            actual_width = x2 - x1
            actual_height = y2 - y1
            
            # 切片文件名
            tile_name = f"{base_name}_r{row_idx}_c{col_idx}"
            
            # 创建固定大小画布（512x512）
            tile_canvases = {
                'A': np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8),
                'B': np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8),
                'D': np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8),
                'label': np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
            }
            
            # 读取并裁剪 A、B、D
            for img_type in ['A', 'B', 'D']:
                try:
                    with rasterio.open(files[img_type]) as src:
                        # 读取窗口数据
                        data = src.read(window=((y1, y2), (x1, x2)))  # (C, H, W)
                        
                        if data.size == 0:
                            continue
                        
                        # 转换为 (H, W, C)
                        channels = min(3, data.shape[0])
                        data_hw3 = np.transpose(data[:channels], (1, 2, 0))
                        
                        if data_hw3.shape[-1] == 1:
                            data_hw3 = np.repeat(data_hw3, 3, axis=-1)
                        
                        # 放到画布左上角
                        h, w = data_hw3.shape[:2]
                        tile_canvases[img_type][:h, :w] = data_hw3
                
                except Exception as e:
                    print(f"      警告: 读取 {base_name} 的 {img_type} 时出错: {e}")
            
            # 读取并裁剪 label
            if has_label:
                try:
                    with Image.open(files['label']).convert('L') as img:
                        label_array = np.array(img)
                    
                    # 直接像素裁剪（与 preprocess.py 一致）
                    src_region = label_array[y1:y2, x1:x2]
                    
                    if src_region.size > 0:
                        # 放到画布左上角
                        h, w = src_region.shape
                        tile_canvases['label'][:h, :w] = src_region
                
                except Exception as e:
                    print(f"      警告: 读取 {base_name} 的 label 时出错: {e}")
            
            # 保存切片
            for img_type in ['A', 'B']:
                img = Image.fromarray(tile_canvases[img_type])
                save_path = os.path.join(output_dir, img_type, f"{tile_name}.png")
                img.save(save_path)
            
            # 保存 D 为 C
            img = Image.fromarray(tile_canvases['D'])
            save_path = os.path.join(output_dir, 'C', f"{tile_name}.png")
            img.save(save_path)
            
            if has_label:
                # 保存 label
                img = Image.fromarray(tile_canvases['label'])
                save_path = os.path.join(output_dir, 'label', f"{tile_name}.png")
                img.save(save_path)
            
            # 记录切片信息
            tile_info = TileInfo(
                name=tile_name,
                source_image=base_name,
                pixel_bounds=(x1, y1, x2, y2),
                actual_size=(actual_width, actual_height)
            )
            tiles_info.append(tile_info)
    
    # 创建图像信息
    image_info = ImageInfo(
        base_name=base_name,
        size=(width, height),
        num_tiles=num_tiles
    )
    
    return image_info, tiles_info


def generate_overview(base_name: str, files: Dict[str, str], output_dir: str):
    """生成单个源图像的缩放预览"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像尺寸
    with rasterio.open(files['A']) as src:
        width = src.width
        height = src.height
    
    # 计算缩放后尺寸
    overview_width = max(1, int(width * OVERVIEW_DOWNSCALE))
    overview_height = max(1, int(height * OVERVIEW_DOWNSCALE))
    
    has_label = files['label'] is not None
    
    # 处理 A、B、D
    for img_type in ['A', 'B', 'D']:
        try:
            with rasterio.open(files[img_type]) as src:
                # 读取所有数据
                data = src.read()  # (C, H, W)
                
                # 转换为 (H, W, C)
                channels = min(3, data.shape[0])
                data_hw3 = np.transpose(data[:channels], (1, 2, 0))
                
                if data_hw3.shape[-1] == 1:
                    data_hw3 = np.repeat(data_hw3, 3, axis=-1)
                
                # 缩放
                img = Image.fromarray(data_hw3.astype(np.uint8))
                img_resized = img.resize((overview_width, overview_height), Image.BILINEAR)
                
                # 保存
                save_path = os.path.join(output_dir, f"{base_name}_{img_type}.png")
                img_resized.save(save_path)
        
        except Exception as e:
            print(f"    警告: 处理 {base_name} 的 {img_type} 预览时出错: {e}")
    
    # 处理 label
    if has_label:
        try:
            with Image.open(files['label']).convert('L') as img:
                label_array = np.array(img)
            
            # 缩放
            img = Image.fromarray(label_array)
            img_resized = img.resize((overview_width, overview_height), Image.NEAREST)
            
            # 保存
            save_path = os.path.join(output_dir, f"{base_name}_label.png")
            img_resized.save(save_path)
        
        except Exception as e:
            print(f"    警告: 处理 {base_name} 的 label 预览时出错: {e}")


def main():
    """主函数"""
    print("="*70)
    print("为每个源图像独立生成切片")
    print("="*70)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"切片大小: {TILE_SIZE}x{TILE_SIZE}")
    print(f"切片重叠: {TILE_OVERLAP} 像素")
    print(f"预览图缩放: {OVERVIEW_DOWNSCALE}")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 查找所有图像组
    print("\n收集图像...")
    base_names = set()
    a_files = glob.glob(os.path.join(INPUT_DIR, "*_A.tif"))
    for a_file in a_files:
        filename = os.path.basename(a_file)
        base_name = filename[:-6] if filename.endswith("_A.tif") else filename
        base_names.add(base_name)
    
    if not base_names:
        raise ValueError("未找到任何符合格式的图像文件")
    
    print(f"找到 {len(base_names)} 个源图像")
    
    # 处理每个源图像
    all_images_info = []
    all_tiles_info = []
    
    test_dir = os.path.join(OUTPUT_ROOT, 'test')
    overview_dir = os.path.join(OUTPUT_ROOT, 'overview')
    
    for base_name in tqdm(sorted(base_names), desc="处理源图像"):
        # 构建文件路径
        files = {
            'A': os.path.join(INPUT_DIR, f"{base_name}_A.tif"),
            'B': os.path.join(INPUT_DIR, f"{base_name}_B.tif"),
            'D': os.path.join(INPUT_DIR, f"{base_name}_D.tif"),
            'label': os.path.join(INPUT_DIR, f"{base_name}_E.png")
        }
        
        # 检查必需文件
        required_files = ['A', 'B', 'D']
        if not all(os.path.exists(files[k]) for k in required_files):
            print(f"警告: {base_name} 缺少必需文件，跳过")
            continue
        
        # 检查label
        if not os.path.exists(files['label']):
            files['label'] = None
        
        # 生成预览图
        if GENERATE_OVERVIEW:
            generate_overview(base_name, files, overview_dir)
        
        # 生成切片
        image_info, tiles_info = process_single_image(base_name, files, test_dir)
        
        all_images_info.append(image_info)
        all_tiles_info.extend(tiles_info)
    
    # 保存元数据
    metadata = {
        'version': METADATA_VERSION,
        'tile_size': TILE_SIZE,
        'tile_overlap': TILE_OVERLAP,
        'images': [asdict(img) for img in all_images_info],
        'tiles': [asdict(t) for t in all_tiles_info]
    }
    
    metadata_path = os.path.join(OUTPUT_ROOT, 'tiles_meta.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 打印统计
    print("\n" + "="*70)
    print("处理完成！")
    print("="*70)
    print(f"源图像数: {len(all_images_info)}")
    print(f"总切片数: {len(all_tiles_info)}")
    
    if GENERATE_OVERVIEW:
        print(f"\n预览图目录: {overview_dir}")
    
    print(f"切片目录: {test_dir}")
    print(f"元数据文件: {metadata_path}")
    
    print("\n下一步:")
    print(f"1. 运行 predict.py --test_dir {test_dir} --output_dir <输出目录>")
    print(f"2. 运行 stitch_predict_tiles.py 拼接预测结果")
    print("="*70)


if __name__ == "__main__":
    main()
