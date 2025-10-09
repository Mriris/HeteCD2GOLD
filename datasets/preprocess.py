"""
基于元数据的地理坐标图像合并与数据集划分脚本

功能：
1. 收集图像元数据，计算全局合并边界
2. 基于元数据计算所有切片位置和源图像映射
3. 进行前景平衡的数据集划分
4. 仅在最终输出时处理实际图像数据
"""

import os
import glob
import random
import shutil
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import numpy as np
from PIL import Image
from tqdm import tqdm
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds, transform as warp_transform
from rasterio.transform import Affine
from dataclasses import dataclass
import math

# 默认参数
DEFAULT_INPUT_DIR = r"/data/jingwei/yantingxuan/Datasets/CityCN/Final"
DEFAULT_OUTPUT_DIR = r"/data/jingwei/yantingxuan/Datasets/CityCN/Split45"
DEFAULT_TILE_SIZE = 512
DEFAULT_VAL_RATIO = 0.2 # 验证集比例
DEFAULT_BLACK_THRESHOLD = 0.95
DEFAULT_TILE_OVERLAP_RATIO = 0.25  # 0 表示无重叠
DEFAULT_LABEL_SIMILARITY_THRESHOLD = 0.98 # 标签相似度阈值
DEFAULT_MIN_LABEL_VARIATION_RATIO = 0.05 # 标签最小变化阈值
DEFAULT_RANDOM_SEED = 666


@dataclass
class ImageMetadata:
    """图像元数据"""
    base_name: str
    files: Dict[str, str]  # 类型 -> 文件路径
    bounds: Tuple[float, float, float, float]  # 地理边界
    crs: object  # 坐标系 (rasterio.crs.CRS)
    size: Tuple[int, int]  # 像素尺寸
    transform: object  # 地理变换矩阵
    resolution: float  # 分辨率
    bounds_in_target: Optional[Tuple[float, float, float, float]] = None  # 转到目标CRS后的边界


@dataclass
class TileMetadata:
    """切片元数据"""
    row: int
    col: int
    global_bounds: Tuple[float, float, float, float]  # 全局坐标
    pixel_bounds: Tuple[int, int, int, int]  # 在合并图像中的像素坐标
    source_mappings: List[str]  # 来源图像标识
    foreground_ratio: Optional[float] = None
    visual_features: Optional[np.ndarray] = None
    group_id: Optional[int] = None
    label_signature: Optional[np.ndarray] = None


@dataclass
class TileGroup:
    """重叠约束下的切片分组"""
    group_id: int
    tiles: List[TileMetadata]
    foreground_ratio: float
    visual_features: Optional[np.ndarray]

    @property
    def tile_count(self) -> int:
        return len(self.tiles)


class MetadataProcessor:
    """元数据处理器"""
    
    def __init__(self, input_dir: str, tile_size: int = 512,
                 black_threshold: float = 0.95,
                 overlap_ratio: float = DEFAULT_TILE_OVERLAP_RATIO):
        self.input_dir = input_dir
        self.tile_size = tile_size
        self.black_threshold = black_threshold
        self.overlap_ratio = max(0.0, min(overlap_ratio, 0.95))
        overlap_pixels = int(round(self.tile_size * self.overlap_ratio))
        self.overlap_pixels = max(0, min(self.tile_size - 1, overlap_pixels))
        self.tile_stride = max(1, self.tile_size - self.overlap_pixels)
        self.images_metadata: List[ImageMetadata] = []
        self.global_bounds: Tuple[float, float, float, float] = None
        self.global_size: Tuple[int, int] = None
        self.global_transform: object = None
        self.target_crs: str = None
        self.target_resolution: float = None
    
    def collect_image_metadata(self):
        """收集所有图像的元数据"""
        print("收集图像元数据...")
        
        # 查找所有图像组
        base_names = set()
        a_files = glob.glob(os.path.join(self.input_dir, "*_A.tif"))
        for a_file in a_files:
            filename = os.path.basename(a_file)
            base_name = filename[:-6] if filename.endswith("_A.tif") else filename
            base_names.add(base_name)
        
        if not base_names:
            raise ValueError("未找到任何符合格式的图像文件")
        
        print(f"找到 {len(base_names)} 个图像组")
        
        # 收集每个图像组的元数据
        for base_name in tqdm(base_names, desc="收集元数据"):
            try:
                # 构建文件路径
                files = {
                    'A': os.path.join(self.input_dir, f"{base_name}_A.tif"),
                    'B': os.path.join(self.input_dir, f"{base_name}_B.tif"),
                    'D': os.path.join(self.input_dir, f"{base_name}_D.tif"),
                    'label': os.path.join(self.input_dir, f"{base_name}_E.png")
                }
                
                # 检查文件存在性
                if not all(os.path.exists(f) for f in files.values()):
                    print(f"警告: 图像组 {base_name} 文件不完整，跳过")
                    continue
                
                # 读取主要TIF文件的元数据
                with rasterio.open(files['A']) as src:
                    bounds = src.bounds
                    crs = src.crs
                    size = (src.width, src.height)
                    transform = src.transform
                    resolution = abs(transform.a)  # x方向分辨率
                
                metadata = ImageMetadata(
                    base_name=base_name,
                    files=files,
                    bounds=bounds,
                    crs=crs,
                    size=size,
                    transform=transform,
                    resolution=resolution
                )
                
                self.images_metadata.append(metadata)
                
            except Exception as e:
                print(f"警告: 处理图像 {base_name} 时出错: {e}")
                continue
        
        if not self.images_metadata:
            raise ValueError("未收集到任何有效的图像元数据")
        
        print(f"成功收集 {len(self.images_metadata)} 个图像的元数据")
    
    def calculate_global_bounds(self):
        """计算全局边界和目标参数"""
        print("计算全局边界...")
        
        # 统计坐标系和分辨率
        crs_counts = {}
        resolutions = []
        
        for metadata in self.images_metadata:
            crs_key = str(metadata.crs)
            crs_counts[crs_key] = crs_counts.get(crs_key, 0) + 1
            resolutions.append(metadata.resolution)
        
        # 选择最常见的坐标系和中位数分辨率
        from rasterio.crs import CRS
        target_crs_str = max(crs_counts.items(), key=lambda x: x[1])[0]
        self.target_crs = CRS.from_string(target_crs_str)
        self.target_resolution = np.median(resolutions)
        
        print(f"目标坐标系: {self.target_crs}")
        print(f"目标分辨率: {self.target_resolution}")
        
        # 将所有图像边界转换到目标坐标系并计算全局边界
        bounds_in_target_list = []
        for m in self.images_metadata:
            try:
                bx1, by1, bx2, by2 = m.bounds
                tb = transform_bounds(m.crs, self.target_crs, bx1, by1, bx2, by2, densify_pts=21)
                m.bounds_in_target = tb
                bounds_in_target_list.append(tb)
            except Exception as e:
                print(f"警告: 转换边界到目标CRS失败: {e}")
                # 回退：使用原边界（可能导致误差）
                m.bounds_in_target = m.bounds
                bounds_in_target_list.append(m.bounds)
        
        min_x = min(b[0] for b in bounds_in_target_list)
        min_y = min(b[1] for b in bounds_in_target_list)
        max_x = max(b[2] for b in bounds_in_target_list)
        max_y = max(b[3] for b in bounds_in_target_list)
        
        self.global_bounds = (min_x, min_y, max_x, max_y)
        
        # 计算全局图像尺寸
        width = int((max_x - min_x) / self.target_resolution)
        height = int((max_y - min_y) / self.target_resolution)
        self.global_size = (width, height)
        
        # 创建全局变换矩阵
        self.global_transform = rasterio.transform.from_bounds(
            min_x, min_y, max_x, max_y, width, height
        )
        
        print(f"全局边界: {self.global_bounds}")
        print(f"全局尺寸: {width} x {height}")
    
    def compute_src_dst_windows(self, 
                                tile_meta: 'TileMetadata', 
                                img_meta: 'ImageMetadata') -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        """计算给定切片与源图像之间的源窗口与目标窗口。
        返回 (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2)
        若无交集则返回 None。
        """
        # 切片地理边界（目标CRS）
        tile_min_x, tile_min_y, tile_max_x, tile_max_y = tile_meta.global_bounds
        # 源图像在目标CRS下的边界
        if img_meta.bounds_in_target is not None:
            img_min_x, img_min_y, img_max_x, img_max_y = img_meta.bounds_in_target
        else:
            img_min_x, img_min_y, img_max_x, img_max_y = img_meta.bounds
        
        # 计算地理交集（目标CRS）
        ix_min_x = max(tile_min_x, img_min_x)
        ix_max_x = min(tile_max_x, img_max_x)
        ix_min_y = max(tile_min_y, img_min_y)
        ix_max_y = min(tile_max_y, img_max_y)
        
        if ix_min_x >= ix_max_x or ix_min_y >= ix_max_y:
            return None
        
        # 将交集四角从目标CRS转换到源图像CRS
        xs_t = [ix_min_x, ix_max_x, ix_min_x, ix_max_x]
        ys_t = [ix_max_y, ix_max_y, ix_min_y, ix_min_y]
        try:
            xs_s, ys_s = warp_transform(self.target_crs, img_meta.crs, xs_t, ys_t)
        except Exception as _:
            # 回退：若转换失败，直接使用目标CRS坐标（仅当CRS一致时才正确）
            xs_s, ys_s = xs_t, ys_t
        
        # 计算源CRS下的包围盒
        sx_min, sx_max = min(xs_s), max(xs_s)
        sy_min, sy_max = min(ys_s), max(ys_s)
        
        # 使用 from_bounds 计算像素窗口
        src_window = rasterio.windows.from_bounds(
            left=sx_min, bottom=sy_min, right=sx_max, top=sy_max, transform=img_meta.transform
        )
        src_col_off = math.floor(src_window.col_off)
        src_row_off = math.floor(src_window.row_off)
        src_width = math.ceil(src_window.width)
        src_height = math.ceil(src_window.height)
        src_x1, src_y1 = src_col_off, src_row_off
        src_x2, src_y2 = src_col_off + src_width, src_row_off + src_height
        
        # 边界裁剪
        src_x1 = max(0, min(img_meta.size[0], src_x1))
        src_y1 = max(0, min(img_meta.size[1], src_y1))
        src_x2 = max(0, min(img_meta.size[0], src_x2))
        src_y2 = max(0, min(img_meta.size[1], src_y2))
        
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return None
        
        # 计算目标窗口（切片内像素坐标，目标CRS）
        # 切片左上角地理坐标为 (tile_min_x, tile_max_y)
        dst_x1 = int(max(0, math.floor((ix_min_x - tile_min_x) / self.target_resolution)))
        dst_y1 = int(max(0, math.floor((tile_max_y - ix_max_y) / self.target_resolution)))
        dst_x2 = int(min(self.tile_size, math.ceil((ix_max_x - tile_min_x) / self.target_resolution)))
        dst_y2 = int(min(self.tile_size, math.ceil((tile_max_y - ix_min_y) / self.target_resolution)))
        
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return None
        
        return (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2)
    
    def generate_tile_metadata(self) -> List[TileMetadata]:
        """生成所有切片的元数据"""
        print("生成切片元数据...")
        
        width, height = self.global_size
        min_x, min_y, max_x, max_y = self.global_bounds

        def build_positions(length: int, tile_size: int, stride: int) -> List[int]:
            if length < tile_size:
                return []
            positions = list(range(0, length - tile_size + 1, stride))
            last_start = length - tile_size
            if not positions:
                positions = [last_start]
            elif positions[-1] != last_start:
                positions.append(last_start)
            return positions

        stride = self.tile_stride
        row_positions = build_positions(height, self.tile_size, stride)
        col_positions = build_positions(width, self.tile_size, stride)

        rows = len(row_positions)
        cols = len(col_positions)

        print(f"切片步长: {stride} 像素 (重叠 {self.overlap_pixels} 像素, {self.overlap_ratio:.2f} 比例)")
        print(f"将生成 {cols} x {rows} = {cols * rows} 个切片")
        
        tiles_metadata = []
        
        for row_idx, y1 in enumerate(tqdm(row_positions, desc="计算切片映射")):
            for col_idx, x1 in enumerate(col_positions):
                # 计算切片在全局图像中的像素边界
                x2 = x1 + self.tile_size
                y2 = y1 + self.tile_size
                pixel_bounds = (x1, y1, x2, y2)
                
                # 计算切片的地理边界
                tile_min_x = min_x + x1 * self.target_resolution
                tile_max_x = min_x + x2 * self.target_resolution
                tile_max_y = max_y - y1 * self.target_resolution
                tile_min_y = max_y - y2 * self.target_resolution
                global_bounds = (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
                
                # 找到与此切片相交的所有源图像
                matched_images = []
                for img_meta in self.images_metadata:
                    # 检查地理边界是否相交
                    ib = img_meta.bounds_in_target if img_meta.bounds_in_target is not None else img_meta.bounds
                    if (ib[2] > tile_min_x and ib[0] < tile_max_x and
                        ib[3] > tile_min_y and ib[1] < tile_max_y):
                        matched_images.append(img_meta)

                # 为每个匹配的源图像生成独立的切片元数据（不再合并成单一大图）
                for img_meta in matched_images:
                    tile_meta = TileMetadata(
                        row=row_idx,
                        col=col_idx,
                        global_bounds=global_bounds,
                        pixel_bounds=pixel_bounds,
                        source_mappings=[img_meta.base_name]
                    )
                    tiles_metadata.append(tile_meta)
        
        # 基于重叠关系分组，防止数据集划分时泄露
        self._assign_overlap_groups(tiles_metadata)

        print(f"生成了 {len(tiles_metadata)} 个有效切片的元数据")
        return tiles_metadata

    def _assign_overlap_groups(self, tiles_metadata: List[TileMetadata]) -> None:
        """为切片分配重叠组，确保相互重叠的切片归为同一组。"""
        if not tiles_metadata:
            return

        cell_size = max(1, min(self.tile_size, self.tile_stride))

        def rectangles_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            overlap_x = min(ax2, bx2) - max(ax1, bx1)
            overlap_y = min(ay2, by2) - max(ay1, by1)
            return overlap_x > 0 and overlap_y > 0

        group_id = 0
        source_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, tile in enumerate(tiles_metadata):
            if tile.source_mappings:
                source_key = tile.source_mappings[0]
            else:
                source_key = f"unknown_{idx}"
            source_to_indices[source_key].append(idx)

        for indices in source_to_indices.values():
            if not indices:
                continue

            cell_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)

            for idx in indices:
                x1, y1, x2, y2 = tiles_metadata[idx].pixel_bounds
                cell_x1 = x1 // cell_size
                cell_y1 = y1 // cell_size
                cell_x2 = (max(x1, x2 - 1)) // cell_size
                cell_y2 = (max(y1, y2 - 1)) // cell_size
                for cx in range(cell_x1, cell_x2 + 1):
                    for cy in range(cell_y1, cell_y2 + 1):
                        cell_map[(cx, cy)].append(idx)

            visited = set()

            def iter_candidates(tile_idx: int) -> List[int]:
                bounds = tiles_metadata[tile_idx].pixel_bounds
                x1, y1, x2, y2 = bounds
                cell_x1 = x1 // cell_size
                cell_y1 = y1 // cell_size
                cell_x2 = (max(x1, x2 - 1)) // cell_size
                cell_y2 = (max(y1, y2 - 1)) // cell_size
                candidates = set()
                for cx in range(cell_x1, cell_x2 + 1):
                    for cy in range(cell_y1, cell_y2 + 1):
                        for cand in cell_map.get((cx, cy), []):
                            candidates.add(cand)
                candidates.discard(tile_idx)
                return list(candidates)

            for idx in indices:
                if idx in visited:
                    continue
                queue = deque([idx])
                visited.add(idx)
                while queue:
                    current = queue.popleft()
                    tile = tiles_metadata[current]
                    tile.group_id = group_id
                    for cand in iter_candidates(current):
                        if cand in visited:
                            continue
                        if rectangles_overlap(tile.pixel_bounds, tiles_metadata[cand].pixel_bounds):
                            visited.add(cand)
                            queue.append(cand)
                group_id += 1

        print(f"重叠分组完成: {group_id} 组")
    
    def calculate_foreground_ratios(self, tiles_metadata: List[TileMetadata]) -> List[TileMetadata]:
        """计算每个切片的前景比例（仅处理标签）"""
        print("计算切片前景比例...")
        
        for tile_meta in tqdm(tiles_metadata, desc="分析前景比例"):
            try:
                # 创建空的标签tile
                label_tile = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                
                # 合并所有源图像的标签
                for base_name in tile_meta.source_mappings:
                    # 找到对应的图像元数据
                    img_meta = next(m for m in self.images_metadata if m.base_name == base_name)
                    
                    # 计算源/目标窗口
                    windows = self.compute_src_dst_windows(tile_meta, img_meta)
                    if windows is None:
                        continue
                    (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = windows
                    
                    # 读取标签文件（与 A 同像素网格），直接用像素窗口裁剪并对齐到目标窗口
                    label_path = img_meta.files['label']
                    with Image.open(label_path).convert('L') as img:
                        img_array = np.array(img)
                    if img_array.size == 0:
                        continue

                    src_region = img_array[src_y1:src_y2, src_x1:src_x2]

                    # 目标窗口大小
                    dst_h = dst_y2 - dst_y1
                    dst_w = dst_x2 - dst_x1
                    if dst_h <= 0 or dst_w <= 0 or src_region.size == 0:
                        continue

                    # 最近邻缩放到目标窗口
                    if src_region.shape[:2] != (dst_h, dst_w):
                        src_region = np.array(Image.fromarray(src_region).resize((dst_w, dst_h), Image.NEAREST))

                    # 合并到标签tile（最大值并集）
                    label_tile[dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(
                        label_tile[dst_y1:dst_y2, dst_x1:dst_x2], src_region
                    )
                
                # 计算前景比例
                total_pixels = label_tile.size
                fg_pixels = np.sum(label_tile > 0)
                fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0

                tile_meta.foreground_ratio = fg_ratio
                tile_meta.label_signature = self._create_label_signature(label_tile)

            except Exception as e:
                print(f"警告: 计算切片 ({tile_meta.row}, {tile_meta.col}) 前景比例时出错: {e}")
                tile_meta.foreground_ratio = 0
                tile_meta.label_signature = None
        
        return tiles_metadata

    def _create_label_signature(self, label_tile: np.ndarray) -> Optional[np.ndarray]:
        """为标签生成压缩签名，用于相似度筛选"""
        if label_tile is None or label_tile.size == 0:
            return None
        try:
            label_bool = (label_tile > 0).astype(np.uint8)
            target_size = 64 if self.tile_size >= 64 else self.tile_size
            if label_bool.shape[0] != target_size:
                img = Image.fromarray(label_bool * 255)
                img = img.resize((target_size, target_size), Image.NEAREST)
                signature = (np.array(img) > 0).astype(np.uint8)
            else:
                signature = label_bool
            return signature
        except Exception as e:
            print(f"警告: 生成标签签名时出错: {e}")
            return None


class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, val_ratio: float = 0.2, black_threshold: float = 0.95,
                 label_similarity_threshold: float = DEFAULT_LABEL_SIMILARITY_THRESHOLD,
                 min_label_variation: float = DEFAULT_MIN_LABEL_VARIATION_RATIO,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        self.val_ratio = val_ratio
        self.black_threshold = black_threshold
        self.label_similarity_threshold = max(0.0, min(1.0, label_similarity_threshold))
        self.min_label_variation = max(0.0, min(1.0, min_label_variation))
        self._rand = random.Random(random_seed)

    def _build_groups(self, tiles: List[TileMetadata]) -> List[TileGroup]:
        groups: Dict[int, List[TileMetadata]] = defaultdict(list)
        next_group_id = 0
        for tile in tiles:
            gid = tile.group_id
            if gid is None:
                gid = next_group_id
                tile.group_id = gid
                next_group_id += 1
            groups[gid].append(tile)

        tile_groups: List[TileGroup] = []
        for gid, members in groups.items():
            fg_vals = [m.foreground_ratio or 0.0 for m in members]
            fg_ratio = float(np.mean(fg_vals)) if fg_vals else 0.0
            feature_list = [m.visual_features for m in members if m.visual_features is not None]
            if feature_list:
                feature_array = np.vstack(feature_list)
                group_features = feature_array.mean(axis=0).astype(np.float32)
            else:
                group_features = None
            tile_groups.append(TileGroup(
                group_id=gid,
                tiles=members,
                foreground_ratio=fg_ratio,
                visual_features=group_features
            ))
        return tile_groups

    @staticmethod
    def _flatten_groups(groups: List[TileGroup]) -> List[TileMetadata]:
        return [tile for group in groups for tile in group.tiles]

    @staticmethod
    def _count_tiles(groups: List[TileGroup]) -> int:
        return sum(group.tile_count for group in groups)

    @staticmethod
    def _mean_fg_ratio_from_tiles(tiles: List[TileMetadata]) -> float:
        if not tiles:
            return 0.0
        return float(np.mean([t.foreground_ratio or 0.0 for t in tiles]))

    def _summarize_tile_visual(self, tile_array: np.ndarray, fg_ratio: float, coverage_ratio: float) -> np.ndarray:
        """提取用于相似度分析的紧凑视觉特征"""
        tile_float = tile_array.astype(np.float32) / 255.0
        mean_rgb = tile_float.mean(axis=(0, 1))
        std_rgb = tile_float.std(axis=(0, 1))
        grayscale = 0.2989 * tile_float[..., 0] + 0.5870 * tile_float[..., 1] + 0.1140 * tile_float[..., 2]
        if grayscale.shape[0] > 1 and grayscale.shape[1] > 1:
            gx = np.mean(np.abs(np.diff(grayscale, axis=1)))
            gy = np.mean(np.abs(np.diff(grayscale, axis=0)))
            edge_strength = float((gx + gy) * 0.5)
        else:
            edge_strength = 0.0
        features = np.concatenate([
            mean_rgb,
            std_rgb,
            np.array([coverage_ratio, edge_strength, fg_ratio], dtype=np.float32)
        ]).astype(np.float32)
        return features

    def is_black_tile_from_sources(self, processor: MetadataProcessor, 
                                  tile_meta: TileMetadata) -> bool:
        """基于源图像判断是否为黑色切片"""
        try:
            # 创建A通道的合并tile用于黑块检测
            a_tile = np.zeros((processor.tile_size, processor.tile_size, 3), dtype=np.uint8)
            
            for base_name in tile_meta.source_mappings:
                img_meta = next(m for m in processor.images_metadata if m.base_name == base_name)
                windows = processor.compute_src_dst_windows(tile_meta, img_meta)
                if windows is None:
                    continue
                (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = windows
                
                with rasterio.open(img_meta.files['A']) as src:
                    from rasterio.windows import Window, transform as win_transform
                    window = Window(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1)
                    src_transform_win = win_transform(window, img_meta.transform)
                    src_region = src.read(window=window)  # (C, H, W)
                    dst_h = dst_y2 - dst_y1
                    dst_w = dst_x2 - dst_x1
                    if src_region.size == 0 or dst_h <= 0 or dst_w <= 0:
                        continue
                    # 构造目标子窗口仿射变换
                    tile_min_x, tile_min_y, tile_max_x, tile_max_y = tile_meta.global_bounds
                    tile_transform = Affine(processor.target_resolution, 0, tile_min_x, 0, -processor.target_resolution, tile_max_y)
                    dst_transform_win = tile_transform * Affine.translation(dst_x1, dst_y1)
                    
                    # 准备目标数组 (C, H, W)
                    channels = src_region.shape[0]
                    channels = min(3, channels)
                    dst_region = np.zeros((channels, dst_h, dst_w), dtype=np.uint8)
                    
                    # 对每个通道重投影（双线性）
                    for c in range(channels):
                        reproject(
                            source=src_region[c],
                            destination=dst_region[c],
                            src_transform=src_transform_win,
                            src_crs=img_meta.crs,
                            dst_transform=dst_transform_win,
                            dst_crs=processor.target_crs,
                            resampling=Resampling.bilinear
                        )
                    
                    # (C,H,W) -> (H,W,C)
                    data_hw3 = np.transpose(dst_region, (1, 2, 0))
                    if data_hw3.shape[-1] == 1:
                        data_hw3 = np.repeat(data_hw3, 3, axis=-1)
                    
                    valid_mask = np.any(data_hw3 > 0, axis=-1)
                    a_tile_region = a_tile[dst_y1:dst_y2, dst_x1:dst_x2]
                    a_tile_region[valid_mask] = data_hw3[valid_mask]
                    a_tile[dst_y1:dst_y2, dst_x1:dst_x2] = a_tile_region
            
            # 检查是否为黑块
            black_pixels = np.sum(np.all(a_tile <= 5, axis=-1))
            total_pixels = a_tile.shape[0] * a_tile.shape[1]
            ratio = black_pixels / total_pixels if total_pixels > 0 else 0
            coverage = max(0.0, min(1.0, 1.0 - ratio))

            try:
                tile_meta.visual_features = self._summarize_tile_visual(
                    a_tile,
                    tile_meta.foreground_ratio or 0.0,
                    coverage
                )
            except Exception as feature_err:
                print(f"警告: 计算相似度特征时出错: {feature_err}")
                tile_meta.visual_features = None
            
            return ratio >= self.black_threshold
            
        except Exception as e:
            print(f"警告: 检测黑块时出错: {e}")
            return False
    
    def split_tiles(self, processor: MetadataProcessor, 
                   tiles_metadata: List[TileMetadata]) -> Tuple[List[TileMetadata], List[TileMetadata]]:
        """划分训练集和验证集"""
        print("划分数据集...")
        
        # 过滤黑色切片
        print("过滤黑色切片...")
        valid_tiles = []
        for tile_meta in tqdm(tiles_metadata, desc="过滤黑块"):
            if not self.is_black_tile_from_sources(processor, tile_meta):
                valid_tiles.append(tile_meta)
        
        print(f"过滤后保留 {len(valid_tiles)} 个有效切片")
        valid_tiles, removed_similar = self._filter_similar_labels(
            valid_tiles,
            similarity_threshold=self.label_similarity_threshold
        )
        if removed_similar > 0:
            print(f"依据标签相似度过滤后保留 {len(valid_tiles)} 个切片")

        if not valid_tiles:
            raise ValueError("没有有效的切片")

        tile_groups = self._build_groups(valid_tiles)
        total_tiles = self._count_tiles(tile_groups)
        if not tile_groups:
            raise ValueError("没有有效的切片组")
        print(f"重叠约束分组数: {len(tile_groups)} 组，覆盖 {total_tiles} 个切片")

        # 计算全局前景比例
        fg_ratios = [t.foreground_ratio or 0 for t in valid_tiles]
        global_fg_ratio = np.mean(fg_ratios)
        print(f"全局前景比例: {global_fg_ratio:.4f}")

        total_tiles = self._count_tiles(tile_groups)
        target_val_tiles = max(1, min(total_tiles - 1, int(round(total_tiles * self.val_ratio))))

        train_groups, val_groups = self._assign_groups_by_ratio(tile_groups, target_val_tiles)

        train_tiles = self._flatten_groups(train_groups)
        val_tiles = self._flatten_groups(val_groups)

        # 计算初步前景比例
        train_fg_ratio = self._mean_fg_ratio_from_tiles(train_tiles)
        val_fg_ratio = self._mean_fg_ratio_from_tiles(val_tiles)
        current_val_tiles = self._count_tiles(val_groups)
        current_val_ratio = current_val_tiles / total_tiles if total_tiles > 0 else 0.0

        feature_gap = self._calculate_feature_gap(train_tiles, val_tiles)

        print("初始划分结果:")
        print(f"  训练集: {len(train_tiles)} 个切片 ({len(train_groups)} 组)，前景比例: {train_fg_ratio:.4f}")
        print(f"  验证集: {len(val_tiles)} 个切片 ({len(val_groups)} 组)，前景比例: {val_fg_ratio:.4f}")
        print(f"  当前验证占比: {current_val_ratio:.4f}，目标占比: {self.val_ratio:.4f}")
        print(f"  前景比例差异: {abs(train_fg_ratio - val_fg_ratio):.4f}")
        if feature_gap is not None:
            print(f"  视觉特征差异: {feature_gap:.4f}")
        else:
            print("  视觉特征差异: 无法计算")

        adjusted = False
        ratio_diff = abs(train_fg_ratio - val_fg_ratio)
        if ratio_diff > 0.02:  # 差异超过2%时进行调整
            print("前景比例差异较大，进行微调...")
            train_groups, val_groups = self._fine_tune_balance(train_groups, val_groups, target_val_tiles)
            train_tiles = self._flatten_groups(train_groups)
            val_tiles = self._flatten_groups(val_groups)
            adjusted = True

        feature_gap = self._calculate_feature_gap(train_tiles, val_tiles)
        if feature_gap is not None and feature_gap > 0.05:
            print("视觉特征差异较大，尝试对齐...")
            train_groups, val_groups = self._align_feature_distribution(train_groups, val_groups, target_val_tiles, tolerance=0.03)
            train_tiles = self._flatten_groups(train_groups)
            val_tiles = self._flatten_groups(val_groups)
            adjusted = True

        if adjusted:
            train_fg_ratio = self._mean_fg_ratio_from_tiles(train_tiles)
            val_fg_ratio = self._mean_fg_ratio_from_tiles(val_tiles)
            ratio_diff = abs(train_fg_ratio - val_fg_ratio)
            feature_gap = self._calculate_feature_gap(train_tiles, val_tiles)
            current_val_tiles = self._count_tiles(val_groups)
            current_val_ratio = current_val_tiles / total_tiles if total_tiles > 0 else 0.0
            print(f"调整后:")
            print(f"  训练集: {len(train_tiles)} 个切片 ({len(train_groups)} 组)，前景比例: {train_fg_ratio:.4f}")
            print(f"  验证集: {len(val_tiles)} 个切片 ({len(val_groups)} 组)，前景比例: {val_fg_ratio:.4f}")
            print(f"  前景比例差异: {ratio_diff:.4f}")
            print(f"  当前验证占比: {current_val_ratio:.4f}，目标占比: {self.val_ratio:.4f}")
            if feature_gap is not None:
                print(f"  视觉特征差异: {feature_gap:.4f}")
            else:
                print("  视觉特征差异: 无法计算")

        train_tiles, val_tiles, adjusted_group_ids, singleton_group_ids, val_count_delta = self._ensure_group_split(
            tile_groups, train_tiles, val_tiles)

        if adjusted_group_ids:
            print(f"为 {len(adjusted_group_ids)} 个重叠组重新分配切片，使每组大约 {self.val_ratio:.2f} 比例划入验证集。")
        else:
            print("所有重叠组原本已按目标比例覆盖训练集和验证集。")

        if singleton_group_ids:
            print(f"注意: {len(singleton_group_ids)} 个重叠组仅包含 1 个切片，无法同时划分到训练/验证集。")

        if val_count_delta != 0:
            print(f"组覆盖调整导致验证集切片数量变化 {val_count_delta:+d}")

        train_groups = self._build_groups(train_tiles)
        val_groups = self._build_groups(val_tiles)

        train_fg_ratio = self._mean_fg_ratio_from_tiles(train_tiles)
        val_fg_ratio = self._mean_fg_ratio_from_tiles(val_tiles)
        current_val_tiles = len(val_tiles)
        current_val_ratio = current_val_tiles / total_tiles if total_tiles > 0 else 0.0
        feature_gap = self._calculate_feature_gap(train_tiles, val_tiles)

        print("最终划分结果:")
        print(f"  训练集: {len(train_tiles)} 个切片 ({len(train_groups)} 组)，前景比例: {train_fg_ratio:.4f}")
        print(f"  验证集: {len(val_tiles)} 个切片 ({len(val_groups)} 组)，前景比例: {val_fg_ratio:.4f}")
        print(f"  前景比例差异: {abs(train_fg_ratio - val_fg_ratio):.4f}")
        print(f"  当前验证占比: {current_val_ratio:.4f}，目标占比: {self.val_ratio:.4f}")
        if feature_gap is not None:
            print(f"  视觉特征差异: {feature_gap:.4f}")
        else:
            print("  视觉特征差异: 无法计算")

        # 显示详细的平衡度分析
        self._analyze_balance(train_tiles, val_tiles, global_fg_ratio)

        final_val_tiles = len(val_tiles)
        final_val_ratio = final_val_tiles / total_tiles if total_tiles > 0 else 0.0
        print(f"最终验证占比: {final_val_ratio:.4f}，目标占比: {self.val_ratio:.4f}")
        deviation_tiles = abs(final_val_tiles - target_val_tiles)
        tolerance_tiles = max(1, int(total_tiles * 0.02))
        if deviation_tiles > tolerance_tiles:
            print(f"提示: 受重叠组约束影响，验证集切片数相对目标存在 {deviation_tiles} 个切片的偏差。")

        return train_tiles, val_tiles

    def _filter_similar_labels(self, tiles: List[TileMetadata], similarity_threshold: float) -> Tuple[List[TileMetadata], int]:
        """基于标签签名去除高相似度切片"""
        if not tiles:
            return tiles, 0

        similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        if similarity_threshold <= 0.0:
            return tiles, 0

        buckets: Dict[Tuple[int, int], List[TileMetadata]] = defaultdict(list)
        for tile in tiles:
            buckets[(tile.row, tile.col)].append(tile)

        filtered_tiles: List[TileMetadata] = []
        removed = 0

        for bucket_tiles in buckets.values():
            if len(bucket_tiles) <= 1:
                filtered_tiles.extend(bucket_tiles)
                continue

            kept_tiles: List[TileMetadata] = []
            kept_signatures: List[Tuple[Optional[np.ndarray], float]] = []

            for tile in bucket_tiles:
                signature = tile.label_signature
                fg_ratio = tile.foreground_ratio or 0.0
                if signature is None or fg_ratio < self.min_label_variation:
                    kept_tiles.append(tile)
                    kept_signatures.append((None, fg_ratio))
                    continue

                duplicate_found = False
                for kept_sig, kept_fg in kept_signatures:
                    if kept_sig is None or kept_fg < self.min_label_variation:
                        continue
                    if kept_sig.shape != signature.shape:
                        continue
                    similarity = np.mean(kept_sig == signature)
                    if similarity >= similarity_threshold:
                        duplicate_found = True
                        break

                if duplicate_found:
                    removed += 1
                    continue

                kept_tiles.append(tile)
                kept_signatures.append((signature, fg_ratio))

            filtered_tiles.extend(kept_tiles)

        if removed > 0:
            print(f"依据标签相似度移除 {removed} 个切片（阈值 {similarity_threshold:.2f}）")

        return filtered_tiles, removed
    
    def _analyze_balance(self, train_tiles: List[TileMetadata], 
                        val_tiles: List[TileMetadata], global_fg_ratio: float):
        """分析数据集平衡度"""
        print("\n=== 数据集平衡度分析 ===")
        
        train_ratios = [t.foreground_ratio or 0 for t in train_tiles]
        val_ratios = [t.foreground_ratio or 0 for t in val_tiles]
        
        # 基础统计
        train_mean = np.mean(train_ratios)
        val_mean = np.mean(val_ratios)
        train_std = np.std(train_ratios)
        val_std = np.std(val_ratios)
        
        print(f"全局前景比例: {global_fg_ratio:.4f}")
        print(f"训练集统计: 均值={train_mean:.4f}, 标准差={train_std:.4f}")
        print(f"验证集统计: 均值={val_mean:.4f}, 标准差={val_std:.4f}")
        print(f"均值差异: {abs(train_mean - val_mean):.4f}")
        
        # 分布分析
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        train_hist, _ = np.histogram(train_ratios, bins=bins)
        val_hist, _ = np.histogram(val_ratios, bins=bins)
        
        print("\n前景比例分布:")
        print("区间范围       训练集    验证集    比例差异")
        print("-" * 40)
        
        for i in range(len(bins)-1):
            train_pct = train_hist[i] / len(train_tiles) * 100 if len(train_tiles) > 0 else 0
            val_pct = val_hist[i] / len(val_tiles) * 100 if len(val_tiles) > 0 else 0
            diff = abs(train_pct - val_pct)
            
            print(f"[{bins[i]:.1f}-{bins[i+1]:.1f})    {train_pct:6.1f}%  {val_pct:6.1f}%  {diff:6.1f}%")
        
        # 平衡度评分
        ratio_diff = abs(train_mean - val_mean)
        if ratio_diff < 0.01:
            balance_score = "优秀"
        elif ratio_diff < 0.02:
            balance_score = "良好"
        elif ratio_diff < 0.05:
            balance_score = "一般"
        else:
            balance_score = "需要改进"
            
        print(f"\n平衡度评分: {balance_score}")
        print("=" * 30)

    def _collect_feature_matrix(self, items) -> Optional[np.ndarray]:
        if not items:
            return None
        collected = []
        for item in items:
            if isinstance(item, TileGroup):
                features = item.visual_features
            else:
                features = item.visual_features
            if features is None:
                return None
            collected.append(features)
        return np.vstack(collected)

    def _calculate_feature_gap(self, train_tiles: List[TileMetadata], val_tiles: List[TileMetadata]) -> Optional[float]:
        train_features = self._collect_feature_matrix(train_tiles)
        val_features = self._collect_feature_matrix(val_tiles)
        if train_features is None or val_features is None:
            return None
        return float(np.linalg.norm(train_features.mean(axis=0) - val_features.mean(axis=0)))

    def _ensure_group_split(self,
                            tile_groups: List[TileGroup],
                            train_tiles: List[TileMetadata],
                            val_tiles: List[TileMetadata]
                            ) -> Tuple[List[TileMetadata], List[TileMetadata], Set[int], Set[int], int]:
        """调整切片分配，力求每个重叠组按验证占比拆分到两个集合。"""
        if not tile_groups:
            return train_tiles, val_tiles, set(), set(), 0

        train_list = list(train_tiles)
        val_list = list(val_tiles)
        train_ids = {id(tile) for tile in train_list}
        val_ids = {id(tile) for tile in val_list}
        group_lookup = {group.group_id: group for group in tile_groups}

        original_val_count = len(val_list)
        adjusted_groups: Set[int] = set()
        singleton_groups: Set[int] = set()

        def remove_from_list(lst: List[TileMetadata], id_set: Set[int], tile: TileMetadata) -> bool:
            tile_id = id(tile)
            if tile_id not in id_set:
                return False
            for idx, existing in enumerate(lst):
                if existing is tile:
                    lst.pop(idx)
                    id_set.remove(tile_id)
                    return True
            return False

        def add_to_list(lst: List[TileMetadata], id_set: Set[int], tile: TileMetadata) -> None:
            tile_id = id(tile)
            if tile_id in id_set:
                return
            lst.append(tile)
            id_set.add(tile_id)

        for group in tile_groups:
            group_tiles = group.tiles
            group_size = len(group_tiles)
            if group_size <= 1:
                singleton_groups.add(group.group_id)
                continue

            desired_val = int(round(group_size * self.val_ratio))
            desired_val = max(1, min(group_size - 1, desired_val))

            train_members = [tile for tile in group_tiles if id(tile) in train_ids]
            val_members = [tile for tile in group_tiles if id(tile) in val_ids]

            if not train_members:
                if not val_members:
                    continue
                tile_to_move = self._rand.choice(val_members)
                if remove_from_list(val_list, val_ids, tile_to_move):
                    add_to_list(train_list, train_ids, tile_to_move)
                    val_members.remove(tile_to_move)
                    train_members.append(tile_to_move)
                    adjusted_groups.add(group.group_id)

            if not val_members:
                if not train_members:
                    continue
                tile_to_move = self._rand.choice(train_members)
                if remove_from_list(train_list, train_ids, tile_to_move):
                    add_to_list(val_list, val_ids, tile_to_move)
                    train_members.remove(tile_to_move)
                    val_members.append(tile_to_move)
                    adjusted_groups.add(group.group_id)

            while len(val_members) < desired_val and len(train_members) > 1:
                tile_to_move = self._rand.choice(train_members)
                if remove_from_list(train_list, train_ids, tile_to_move):
                    add_to_list(val_list, val_ids, tile_to_move)
                    train_members.remove(tile_to_move)
                    val_members.append(tile_to_move)
                    adjusted_groups.add(group.group_id)

            while len(val_members) > desired_val and len(val_members) > 1:
                tile_to_move = self._rand.choice(val_members)
                if remove_from_list(val_list, val_ids, tile_to_move):
                    add_to_list(train_list, train_ids, tile_to_move)
                    val_members.remove(tile_to_move)
                    train_members.append(tile_to_move)
                    adjusted_groups.add(group.group_id)

            if len(val_members) == 0 and len(train_members) > 1:
                tile_to_move = self._rand.choice(train_members)
                if remove_from_list(train_list, train_ids, tile_to_move):
                    add_to_list(val_list, val_ids, tile_to_move)
                    adjusted_groups.add(group.group_id)

        total_tiles = len(train_list) + len(val_list)
        target_total_val = max(1, min(total_tiles - 1, int(round(total_tiles * self.val_ratio))))

        group_val_counts: Dict[int, int] = {}
        group_train_counts: Dict[int, int] = {}
        for group in tile_groups:
            val_count = 0
            train_count = 0
            for tile in group.tiles:
                if id(tile) in val_ids:
                    val_count += 1
                elif id(tile) in train_ids:
                    train_count += 1
            group_val_counts[group.group_id] = val_count
            group_train_counts[group.group_id] = train_count

        diff = len(val_list) - target_total_val
        if diff > 0:
            movable_groups = [gid for gid, count in group_val_counts.items() if count > 1]
            while diff > 0 and movable_groups:
                gid = self._rand.choice(movable_groups)
                group_obj = group_lookup.get(gid)
                if not group_obj:
                    movable_groups.remove(gid)
                    continue
                candidates = [tile for tile in group_obj.tiles if id(tile) in val_ids]
                if len(candidates) <= 1:
                    movable_groups.remove(gid)
                    continue
                tile_to_move = self._rand.choice(candidates)
                if remove_from_list(val_list, val_ids, tile_to_move):
                    add_to_list(train_list, train_ids, tile_to_move)
                    group_val_counts[gid] -= 1
                    group_train_counts[gid] += 1
                    diff -= 1
                    adjusted_groups.add(gid)
                if group_val_counts[gid] <= 1 and gid in movable_groups:
                    movable_groups.remove(gid)

        elif diff < 0:
            movable_groups = [gid for gid, count in group_train_counts.items() if count > 1]
            while diff < 0 and movable_groups:
                gid = self._rand.choice(movable_groups)
                group_obj = group_lookup.get(gid)
                if not group_obj:
                    movable_groups.remove(gid)
                    continue
                candidates = [tile for tile in group_obj.tiles if id(tile) in train_ids]
                if len(candidates) <= 1:
                    movable_groups.remove(gid)
                    continue
                tile_to_move = self._rand.choice(candidates)
                if remove_from_list(train_list, train_ids, tile_to_move):
                    add_to_list(val_list, val_ids, tile_to_move)
                    group_train_counts[gid] -= 1
                    group_val_counts[gid] += 1
                    diff += 1
                    adjusted_groups.add(gid)
                if group_train_counts[gid] <= 1 and gid in movable_groups:
                    movable_groups.remove(gid)

        final_diff = len(val_list) - original_val_count
        return train_list, val_list, adjusted_groups, singleton_groups, final_diff

    def _assign_groups_by_ratio(self, tile_groups: List[TileGroup], target_val_tiles: int) -> Tuple[List[TileGroup], List[TileGroup]]:
        if len(tile_groups) < 2:
            raise ValueError("有效切片组数量不足，无法划分训练/验证集")

        total_tiles = self._count_tiles(tile_groups)
        target_val_tiles = max(1, min(total_tiles - 1, target_val_tiles))

        states = [-1] * (total_tiles + 1)
        prev = [-1] * (total_tiles + 1)
        states[0] = -2  # 起点标记

        for idx, group in enumerate(tile_groups):
            weight = group.tile_count
            for s in range(total_tiles - weight, -1, -1):
                if states[s] != -1 and states[s + weight] == -1:
                    states[s + weight] = idx
                    prev[s + weight] = s

        best_sum = None
        best_diff = total_tiles + 1
        for s in range(1, total_tiles):
            if states[s] == -1:
                continue
            diff = abs(s - target_val_tiles)
            if diff < best_diff:
                best_diff = diff
                best_sum = s

        if best_sum is None:
            # 回退策略：选择最小组作为验证集
            min_idx = min(range(len(tile_groups)), key=lambda i: tile_groups[i].tile_count)
            val_groups = [tile_groups[min_idx]]
            train_groups = [g for i, g in enumerate(tile_groups) if i != min_idx]
            return train_groups, val_groups

        selected_indices = set()
        cursor = best_sum
        while cursor > 0 and states[cursor] >= 0:
            idx = states[cursor]
            if idx in selected_indices:
                break
            selected_indices.add(idx)
            cursor = prev[cursor]

        val_groups = [tile_groups[i] for i in selected_indices]
        train_groups = [g for i, g in enumerate(tile_groups) if i not in selected_indices]

        if not val_groups or not train_groups:
            # 保证两侧都有数据
            min_idx = min(range(len(tile_groups)), key=lambda i: tile_groups[i].tile_count)
            val_groups = [tile_groups[min_idx]]
            train_groups = [g for i, g in enumerate(tile_groups) if i != min_idx]

        return train_groups, val_groups

    def _align_feature_distribution(self,
                                    train_groups: List[TileGroup],
                                    val_groups: List[TileGroup],
                                    target_val_tiles: int,
                                    tolerance: float = 0.02,
                                    max_iterations: int = 50) -> Tuple[List[TileGroup], List[TileGroup]]:
        train_features = self._collect_feature_matrix(train_groups)
        val_features = self._collect_feature_matrix(val_groups)
        if train_features is None or val_features is None:
            return train_groups, val_groups
        train_mean = train_features.mean(axis=0)
        val_mean = val_features.mean(axis=0)
        current_diff = float(np.linalg.norm(train_mean - val_mean))
        train_fg_vals = np.array([group.foreground_ratio or 0.0 for group in train_groups], dtype=np.float32)
        val_fg_vals = np.array([group.foreground_ratio or 0.0 for group in val_groups], dtype=np.float32)
        train_fg_mean = float(train_fg_vals.mean()) if train_fg_vals.size > 0 else 0.0
        val_fg_mean = float(val_fg_vals.mean()) if val_fg_vals.size > 0 else 0.0
        fg_diff = abs(train_fg_mean - val_fg_mean)
        if current_diff <= tolerance:
            return train_groups, val_groups
        total_tiles = self._count_tiles(train_groups) + self._count_tiles(val_groups)
        tolerance_tiles = max(1, int(total_tiles * 0.02))
        current_val_tiles = self._count_tiles(val_groups)
        best_tile_diff = abs(current_val_tiles - target_val_tiles)
        for _ in range(max_iterations):
            improved = False
            # 采样若干候选交换，寻找能明显缩小差异的组合
            for _ in range(40):
                if not train_groups or not val_groups:
                    break
                train_idx = self._rand.randrange(len(train_groups))
                val_idx = self._rand.randrange(len(val_groups))
                train_group = train_groups[train_idx]
                val_group = val_groups[val_idx]
                if train_group.visual_features is None or val_group.visual_features is None:
                    continue
                nf_train_fg_mean = train_fg_mean + ((val_group.foreground_ratio or 0.0) - (train_group.foreground_ratio or 0.0)) / max(1, len(train_groups))
                nf_val_fg_mean = val_fg_mean + ((train_group.foreground_ratio or 0.0) - (val_group.foreground_ratio or 0.0)) / max(1, len(val_groups))
                new_fg_diff = abs(nf_train_fg_mean - nf_val_fg_mean)
                if new_fg_diff > fg_diff + 0.01:
                    continue
                nf_train_mean = train_mean + (val_group.visual_features - train_group.visual_features) / max(1, len(train_groups))
                nf_val_mean = val_mean + (train_group.visual_features - val_group.visual_features) / max(1, len(val_groups))
                new_diff = float(np.linalg.norm(nf_train_mean - nf_val_mean))
                new_val_tiles = current_val_tiles - val_group.tile_count + train_group.tile_count
                new_tile_diff = abs(new_val_tiles - target_val_tiles)
                if new_diff < current_diff * 0.98 and new_tile_diff <= best_tile_diff + tolerance_tiles:
                    train_groups[train_idx], val_groups[val_idx] = val_group, train_group
                    train_mean = nf_train_mean
                    val_mean = nf_val_mean
                    current_diff = new_diff
                    train_fg_mean = nf_train_fg_mean
                    val_fg_mean = nf_val_fg_mean
                    fg_diff = new_fg_diff
                    current_val_tiles = new_val_tiles
                    best_tile_diff = min(best_tile_diff, new_tile_diff)
                    improved = True
                    break
            if not improved:
                break
            if current_diff <= tolerance:
                break
        return train_groups, val_groups

    def _fine_tune_balance(self, train_groups: List[TileGroup], 
                          val_groups: List[TileGroup], 
                          target_val_tiles: int) -> Tuple[List[TileGroup], List[TileGroup]]:
        """微调训练集和验证集的前景比例平衡（按组操作），同时保持切片数量接近目标比例"""
        if not train_groups or not val_groups:
            return train_groups, val_groups

        total_tiles = self._count_tiles(train_groups) + self._count_tiles(val_groups)
        max_swaps = max(1, min(self._count_tiles(train_groups), self._count_tiles(val_groups)) // 20)
        tolerance_tiles = max(1, int(total_tiles * 0.02))
        current_val_tiles = self._count_tiles(val_groups)
        best_tile_diff = abs(current_val_tiles - target_val_tiles)

        for _ in range(max_swaps):
            train_tiles = self._flatten_groups(train_groups)
            val_tiles = self._flatten_groups(val_groups)
            train_fg_ratio = self._mean_fg_ratio_from_tiles(train_tiles)
            val_fg_ratio = self._mean_fg_ratio_from_tiles(val_tiles)

            if abs(train_fg_ratio - val_fg_ratio) < 0.01:
                break

            swapped = False
            if train_fg_ratio > val_fg_ratio and train_groups and val_groups:
                train_high = max(train_groups, key=lambda g: g.foreground_ratio or 0)
                val_low = min(val_groups, key=lambda g: g.foreground_ratio or 0)
                new_val_tiles = current_val_tiles - val_low.tile_count + train_high.tile_count
                new_tile_diff = abs(new_val_tiles - target_val_tiles)
                if ((train_high.foreground_ratio or 0) > (val_low.foreground_ratio or 0) and
                        new_tile_diff <= best_tile_diff + tolerance_tiles):
                    train_groups.remove(train_high)
                    val_groups.remove(val_low)
                    train_groups.append(val_low)
                    val_groups.append(train_high)
                    current_val_tiles = new_val_tiles
                    best_tile_diff = min(best_tile_diff, new_tile_diff)
                    swapped = True
            elif val_groups and train_groups:
                val_high = max(val_groups, key=lambda g: g.foreground_ratio or 0)
                train_low = min(train_groups, key=lambda g: g.foreground_ratio or 0)
                new_val_tiles = current_val_tiles - val_high.tile_count + train_low.tile_count
                new_tile_diff = abs(new_val_tiles - target_val_tiles)
                if ((val_high.foreground_ratio or 0) > (train_low.foreground_ratio or 0) and
                        new_tile_diff <= best_tile_diff + tolerance_tiles):
                    val_groups.remove(val_high)
                    train_groups.remove(train_low)
                    val_groups.append(train_low)
                    train_groups.append(val_high)
                    current_val_tiles = new_val_tiles
                    best_tile_diff = min(best_tile_diff, new_tile_diff)
                    swapped = True

            if not swapped:
                break

        return train_groups, val_groups


class TileGenerator:
    """切片生成器 - 只在输出时处理实际图像"""
    
    def __init__(self, processor: MetadataProcessor):
        self.processor = processor
    
    def generate_tile(self, tile_meta: TileMetadata) -> Dict[str, np.ndarray]:
        """生成单个切片的实际图像数据"""
        tile_data = {}
        
        # 为每种类型创建空tile
        for img_type in ['A', 'B', 'D', 'label']:
            if img_type == 'label':
                tile_data[img_type] = np.zeros((self.processor.tile_size, self.processor.tile_size), dtype=np.uint8)
            else:
                # 假设3波段
                tile_data[img_type] = np.zeros((self.processor.tile_size, self.processor.tile_size, 3), dtype=np.uint8)
        
        # 合并所有源图像
        for base_name in tile_meta.source_mappings:
            img_meta = next(m for m in self.processor.images_metadata if m.base_name == base_name)
            windows = self.processor.compute_src_dst_windows(tile_meta, img_meta)
            if windows is None:
                continue
            (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = windows
            
            # 处理每种图像类型
            for img_type in ['A', 'B', 'D', 'label']:
                try:
                    file_path = img_meta.files[img_type]
                    dst_h = dst_y2 - dst_y1
                    dst_w = dst_x2 - dst_x1
                    if dst_h <= 0 or dst_w <= 0:
                        continue
                    
                    if img_type == 'label':
                        # PNG标签文件（像素与 A 同尺寸，同一像素网格）。
                        # 直接使用源窗口像素裁剪并缩放到目标子窗口大小；
                        # 同时用 A/B 的有效覆盖掩膜约束标签，避免标签落在无数据区域。
                        with Image.open(file_path).convert('L') as img:
                            img_array = np.array(img)

                        src_region = img_array[src_y1:src_y2, src_x1:src_x2]
                        if src_region.size == 0:
                            continue
                        if src_region.shape[:2] != (dst_h, dst_w):
                            img_pil = Image.fromarray(src_region)
                            img_pil = img_pil.resize((dst_w, dst_h), Image.NEAREST)
                            src_region = np.array(img_pil)

                        # 计算 A/B 的有效覆盖掩膜（>0 视为有效像素）
                        a_sub = tile_data['A'][dst_y1:dst_y2, dst_x1:dst_x2]
                        b_sub = tile_data['B'][dst_y1:dst_y2, dst_x1:dst_x2]
                        a_valid = np.any(a_sub > 0, axis=-1)
                        b_valid = np.any(b_sub > 0, axis=-1)
                        coverage_mask = a_valid & b_valid

                        # 将无覆盖区域的标签置零，避免视觉错位
                        if coverage_mask.shape == src_region.shape:
                            src_region = np.where(coverage_mask, src_region, 0)
                        else:
                            # 理论上尺寸应一致；若不一致则保守不作掩膜
                            pass

                        # 使用最大值合并（标签）
                        dst_region = tile_data[img_type][dst_y1:dst_y2, dst_x1:dst_x2]
                        tile_data[img_type][dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(dst_region, src_region)
                    else:
                        # TIF文件：按地理坐标精准重采样到目标窗口
                        with rasterio.open(file_path) as src:
                            from rasterio.windows import Window, transform as win_transform
                            window = Window(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1)
                            src_transform_win = win_transform(window, img_meta.transform)
                            src_region = src.read(window=window)  # (C,H,W)
                            if src_region.size == 0:
                                continue
                            
                            # 目标子窗口仿射
                            tile_min_x, tile_min_y, tile_max_x, tile_max_y = tile_meta.global_bounds
                            tile_transform = Affine(self.processor.target_resolution, 0, tile_min_x, 0, -self.processor.target_resolution, tile_max_y)
                            dst_transform_win = tile_transform * Affine.translation(dst_x1, dst_y1)
                            
                            channels = min(3, src_region.shape[0])
                            dst_region = np.zeros((channels, dst_h, dst_w), dtype=np.uint8)
                            for c in range(channels):
                                reproject(
                                    source=src_region[c],
                                    destination=dst_region[c],
                                    src_transform=src_transform_win,
                                    src_crs=img_meta.crs,
                                    dst_transform=dst_transform_win,
                                    dst_crs=self.processor.target_crs,
                                    resampling=Resampling.bilinear
                                )
                            data_hw3 = np.transpose(dst_region, (1, 2, 0))
                            if data_hw3.shape[-1] == 1:
                                data_hw3 = np.repeat(data_hw3, 3, axis=-1)
                            
                            valid_mask = np.any(data_hw3 > 0, axis=-1)
                            dst_region_hw3 = tile_data[img_type][dst_y1:dst_y2, dst_x1:dst_x2]
                            dst_region_hw3[valid_mask] = data_hw3[valid_mask]
                            tile_data[img_type][dst_y1:dst_y2, dst_x1:dst_x2] = dst_region_hw3
                except Exception as e:
                    print(f"警告: 处理 {base_name} 的 {img_type} 时出错: {e}")
                    continue
        
        return tile_data
    
    def save_dataset(self, train_tiles: List[TileMetadata], 
                    val_tiles: List[TileMetadata], output_dir: str):
        """保存数据集"""
        print("保存数据集...")
        
        # 创建输出目录
        splits = ['train', 'val', 'test']
        types = ['A', 'B', 'C', 'label']
        
        for split in splits:
            for img_type in types:
                os.makedirs(os.path.join(output_dir, split, img_type), exist_ok=True)
        
        # 保存训练集
        self._save_split(train_tiles, output_dir, 'train')
        
        # 保存验证集
        self._save_split(val_tiles, output_dir, 'val')

        # 复制验证集为测试集，避免重复生成切片
        self._copy_split(output_dir, 'val', 'test')
        
        print("数据集保存完成!")
    
    def _save_split(self, tiles: List[TileMetadata], output_dir: str, split: str):
        """保存一个数据集分割"""
        type_mapping = {'A': 'A', 'B': 'B', 'D': 'C', 'label': 'label'}

        for tile_meta in tqdm(tiles, desc=f"保存{split}集"):
            # 生成切片数据
            tile_data = self.generate_tile(tile_meta)
            
            # 构建文件名，附加源图像标识，避免不同源图像的重复区域被覆盖
            if tile_meta.source_mappings:
                source_tag = tile_meta.source_mappings[0]
            else:
                source_tag = "tile"
            base_name = f"{source_tag}_r{tile_meta.row}_c{tile_meta.col}"
            
            # 保存每种类型
            for src_type, dst_type in type_mapping.items():
                data = tile_data[src_type]
                
                # 确保数据类型正确
                if data.dtype != np.uint8:
                    if data.max() <= 1.0:
                        data = (data * 255).astype(np.uint8)
                    else:
                        data = data.astype(np.uint8)
                
                # 保存为PNG
                img = Image.fromarray(data)
                save_path = os.path.join(output_dir, split, dst_type, f"{base_name}.png")
                img.save(save_path)

    def _copy_split(self, output_dir: str, src_split: str, dst_split: str):
        """将已生成的切片从一个划分复制到另一个划分"""
        for img_type in ['A', 'B', 'C', 'label']:
            src_dir = os.path.join(output_dir, src_split, img_type)
            dst_dir = os.path.join(output_dir, dst_split, img_type)
            os.makedirs(dst_dir, exist_ok=True)

            for file_name in os.listdir(src_dir):
                src_path = os.path.join(src_dir, file_name)
                dst_path = os.path.join(dst_dir, file_name)

                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)


def main():
    """主函数"""
    print("开始基于元数据的地理图像合并与数据集划分...")
    
    # 设置随机种子确保可重现性
    import random
    random_seed = DEFAULT_RANDOM_SEED
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 使用默认参数
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    tile_size = DEFAULT_TILE_SIZE
    val_ratio = DEFAULT_VAL_RATIO
    black_threshold = DEFAULT_BLACK_THRESHOLD
    overlap_ratio = DEFAULT_TILE_OVERLAP_RATIO
    label_similarity_threshold = DEFAULT_LABEL_SIMILARITY_THRESHOLD
    min_label_variation = DEFAULT_MIN_LABEL_VARIATION_RATIO
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"切片大小: {tile_size}x{tile_size}")
    print(f"验证集比例: {val_ratio}")
    print(f"切片重叠比例: {overlap_ratio:.2f}")
    print(f"标签相似度阈值: {label_similarity_threshold:.2f}")
    print(f"标签最小变化阈值: {min_label_variation:.4f}")
    print(f"随机种子: {random_seed} (确保结果可重现)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 第一步：收集图像元数据
        processor = MetadataProcessor(input_dir, tile_size, black_threshold, overlap_ratio=overlap_ratio)
        processor.collect_image_metadata()
        processor.calculate_global_bounds()
        
        # 第二步：生成切片元数据
        tiles_metadata = processor.generate_tile_metadata()
        
        # 第三步：计算前景比例
        tiles_metadata = processor.calculate_foreground_ratios(tiles_metadata)
        
        # 第四步：划分数据集
        splitter = DatasetSplitter(val_ratio, black_threshold, label_similarity_threshold, min_label_variation, random_seed)
        train_tiles, val_tiles = splitter.split_tiles(processor, tiles_metadata)
        
        # 第五步：生成并保存实际图像
        generator = TileGenerator(processor)
        generator.save_dataset(train_tiles, val_tiles, output_dir)
        
        print(f"\n处理完成!")
        print(f"训练集: {len(train_tiles)} 个切片")
        print(f"验证集: {len(val_tiles)} 个切片")
        print(f"测试集: {len(val_tiles)} 个切片 (与验证集相同)")
        
    except Exception as e:
        print(f"错误: {e}")
        raise


if __name__ == "__main__":
    main()
