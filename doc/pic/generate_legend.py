import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
import os
import numpy as np
from PIL import Image

# 设置高分辨率
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 图片路径和标签
images = [
    ('l2_distance_bg_v2.png', 'L2 Distance\n$D_{L2}$'),
    ('cosine_distance_bg_v2.png', 'Cosine Distance\n$D_{cos}$'),
    ('avgpool_bg.png', 'Average Pooling\n$AvgPool$'),
    ('maxpool_bg.png', 'Max Pooling\n$MaxPool$'),
    ('gap_bg.png', 'Global Average Pooling\n$GAP$'),
    ('gmp_bg.png', 'Global Max Pooling\n$GMP$')
]

# 针对不同图片的显示前处理：
# - 余弦图：裁剪右侧 50%（去掉左侧空白），再以原尺寸显示
# - L2 图：按比例缩小后，居中贴到同尺寸透明画布，使视觉占比减小
def _load_image_for_display(img_path: str, l2_zoom: float = 0.7, cos_target_ratio: float = 0.85) -> np.ndarray:
    name = os.path.basename(img_path)
    try:
        img = Image.open(img_path).convert('RGBA')
    except Exception:
        # 回退到 mpimg 读取
        return mpimg.imread(img_path)

    if name == 'cosine_distance_bg_v2.png':
        w, h = img.size
        # 取右侧 50%
        right = img.crop((w // 2, 0, w, h))
        # 放大到占画布宽度的 cos_target_ratio（默认 85%）
        target_w = int(w * cos_target_ratio)
        scale = target_w / (w - w // 2)
        nw, nh = max(1, int((w - w // 2) * scale)), max(1, int(h * scale))
        right_big = right.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        ox = (w - nw) // 2
        oy = (h - nh) // 2
        canvas.paste(right_big, (ox, oy), mask=right_big)
        img = canvas

    if name == 'l2_distance_bg_v2.png':
        w, h = img.size
        nw, nh = max(1, int(w * l2_zoom)), max(1, int(h * l2_zoom))
        small = img.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        ox = (w - nw) // 2
        oy = (h - nh) // 2
        canvas.paste(small, (ox, oy), mask=small)
        img = canvas

    return np.array(img)

# 创建 2x3 布局
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Pooling and Distance Operations', fontsize=16, fontweight='bold', y=0.98)

# 展平axes数组以便迭代
axes = axes.flatten()

for idx, (img_name, label) in enumerate(images):
    ax = axes[idx]
    img_path = os.path.join('doc/pic', img_name)
    
    # 检查文件是否存在
    if os.path.exists(img_path):
        arr = _load_image_for_display(img_path)
        ax.imshow(arr)
    else:
        ax.text(0.5, 0.5, f'{img_name}\nNot Found', 
                ha='center', va='center', fontsize=10, color='red')
    
    ax.axis('off')
    
    # 添加标签
    ax.text(0.5, -0.15, label, transform=ax.transAxes,
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                     edgecolor='gray', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('doc/pic/operations_legend.png', 
            bbox_inches='tight', dpi=300, transparent=True)
print("✓ 图例已生成: operations_legend.png")
plt.close()

# ========== 创建紧凑的水平布局版本 ==========
fig, axes = plt.subplots(1, 6, figsize=(18, 3))

for idx, (img_name, label) in enumerate(images):
    ax = axes[idx]
    img_path = os.path.join('doc/pic', img_name)
    
    if os.path.exists(img_path):
        arr = _load_image_for_display(img_path)
        ax.imshow(arr)
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
    
    ax.axis('off')
    ax.set_title(label, fontsize=10, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('doc/pic/operations_legend_horizontal.png', 
            bbox_inches='tight', dpi=300, transparent=True)
print("✓ 水平图例已生成: operations_legend_horizontal.png")
plt.close()

# ========== 创建竖直布局版本 ==========
fig, axes = plt.subplots(6, 1, figsize=(5, 15))

for idx, (img_name, label) in enumerate(images):
    ax = axes[idx]
    img_path = os.path.join('doc/pic', img_name)
    
    if os.path.exists(img_path):
        arr = _load_image_for_display(img_path)
        ax.imshow(arr)
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
    
    ax.axis('off')
    # 标签放在左侧
    ax.text(-0.05, 0.5, label.replace('\n', ' '), 
            transform=ax.transAxes,
            ha='right', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                     edgecolor='gray', alpha=0.8))

plt.tight_layout()
plt.savefig('doc/pic/operations_legend_vertical.png', 
            bbox_inches='tight', dpi=300, transparent=True)
print("✓ 竖直图例已生成: operations_legend_vertical.png")
plt.close()

# ========== 创建紧凑的论文用版本（无标题栏，仅标签）==========
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
axes = axes.flatten()

for idx, (img_name, label) in enumerate(images):
    ax = axes[idx]
    img_path = os.path.join('doc/pic', img_name)
    
    if os.path.exists(img_path):
        arr = _load_image_for_display(img_path)
        ax.imshow(arr)
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
    
    ax.axis('off')
    
    # 简洁标签（使用子图标号）
    subfig_label = chr(97 + idx)  # a, b, c, d, e, f
    label_text = label.split('\n')[0]  # 只取第一行
    ax.text(0.02, 0.98, f'({subfig_label}) {label_text}', 
            transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='black', alpha=0.9))

plt.tight_layout(pad=0.5)
plt.savefig('doc/pic/operations_legend_paper.png', 
            bbox_inches='tight', dpi=300, transparent=True)
print("✓ 论文版图例已生成: operations_legend_paper.png")
plt.close()

print("\n所有图例已生成完成！")
print("\n生成的文件:")
print("- operations_legend.png (2x3网格，带标题)")
print("- operations_legend_horizontal.png (1x6水平排列)")
print("- operations_legend_vertical.png (6x1竖直排列)")
print("- operations_legend_paper.png (2x3论文格式，带子图标号)")

