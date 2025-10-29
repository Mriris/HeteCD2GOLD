import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# 设置高分辨率
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ========== AvgPool - 平均池化 ==========
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 创建3x3网格，显示区域平均的效果
grid_data = np.array([
    [0.9, 0.8, 0.7],
    [0.6, 0.5, 0.4],
    [0.3, 0.2, 0.1]
])

# 显示原始网格
im = ax.imshow(grid_data, extent=[0.05, 0.55, 0.25, 0.75], 
               cmap='Blues', alpha=0.8, vmin=0, vmax=1)

# 显示池化后的结果（中心值）
center_val = np.mean(grid_data)
result_rect = Rectangle((0.65, 0.35), 0.25, 0.25, 
                         facecolor=plt.cm.Blues(center_val), 
                         edgecolor='white', linewidth=2)
ax.add_patch(result_rect)

# 添加箭头
ax.annotate('', xy=(0.62, 0.5), xytext=(0.58, 0.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='orange'))

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/avgpool_bg.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ AvgPool 背景已生成: avgpool_bg.png")
plt.close()

# ========== MaxPool - 最大池化 ==========
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 创建网格，高亮最大值
grid_data = np.array([
    [0.3, 0.5, 0.4],
    [0.6, 0.9, 0.7],  # 0.9 是最大值
    [0.2, 0.4, 0.3]
])

# 显示原始网格
im = ax.imshow(grid_data, extent=[0.05, 0.55, 0.25, 0.75], 
               cmap='Reds', alpha=0.8, vmin=0, vmax=1)

# 高亮最大值位置
max_highlight = Rectangle((0.05 + 0.5/3, 0.25 + 0.5/3), 
                          0.5/3, 0.5/3,
                          facecolor='none', edgecolor='yellow', 
                          linewidth=3, linestyle='--')
ax.add_patch(max_highlight)

# 显示池化后的结果（最大值）
max_val = np.max(grid_data)
result_rect = Rectangle((0.65, 0.35), 0.25, 0.25, 
                         facecolor=plt.cm.Reds(max_val), 
                         edgecolor='white', linewidth=2)
ax.add_patch(result_rect)

# 添加箭头
ax.annotate('', xy=(0.62, 0.5), xytext=(0.58, 0.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='orange'))

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/maxpool_bg.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ MaxPool 背景已生成: maxpool_bg.png")
plt.close()

# ========== GAP - 全局平均池化 ==========
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 创建较大的特征图
np.random.seed(42)
feature_map = np.random.rand(8, 8) * 0.7 + 0.2

# 显示特征图
im = ax.imshow(feature_map, extent=[0.05, 0.65, 0.2, 0.8], 
               cmap='viridis', alpha=0.85)

# 全局平均 - 显示为单个值
avg_val = np.mean(feature_map)
# 圆形表示单个标量值
circle = plt.Circle((0.8, 0.5), 0.1, 
                    facecolor=plt.cm.viridis(avg_val), 
                    edgecolor='white', linewidth=2.5)
ax.add_patch(circle)

# 添加箭头
ax.annotate('', xy=(0.68, 0.5), xytext=(0.68, 0.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='orange'))

# 添加表示全局的虚线框
global_box = Rectangle((0.05, 0.2), 0.6, 0.6,
                       facecolor='none', edgecolor='cyan',
                       linewidth=2, linestyle=':', alpha=0.7)
ax.add_patch(global_box)

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/gap_bg.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ GAP 背景已生成: gap_bg.png")
plt.close()

# ========== GMP - 全局最大池化 ==========
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 创建特征图，有明显的峰值
np.random.seed(123)
feature_map = np.random.rand(8, 8) * 0.5 + 0.2
# 设置一个明显的最大值
feature_map[2, 5] = 0.95

# 显示特征图
im = ax.imshow(feature_map, extent=[0.05, 0.65, 0.2, 0.8], 
               cmap='plasma', alpha=0.85)

# 高亮最大值位置
max_pos = np.unravel_index(np.argmax(feature_map), feature_map.shape)
cell_h = 0.6 / 8
cell_w = 0.6 / 8
max_highlight = Rectangle((0.05 + max_pos[1] * cell_w, 
                          0.2 + (7 - max_pos[0]) * cell_h),
                         cell_w, cell_h,
                         facecolor='none', edgecolor='yellow',
                         linewidth=3, linestyle='--')
ax.add_patch(max_highlight)

# 全局最大 - 显示为单个值
max_val = np.max(feature_map)
circle = plt.Circle((0.8, 0.5), 0.1, 
                    facecolor=plt.cm.plasma(max_val), 
                    edgecolor='white', linewidth=2.5)
ax.add_patch(circle)

# 添加箭头
ax.annotate('', xy=(0.68, 0.5), xytext=(0.68, 0.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='orange'))

# 添加表示全局的虚线框
global_box = Rectangle((0.05, 0.2), 0.6, 0.6,
                       facecolor='none', edgecolor='cyan',
                       linewidth=2, linestyle=':', alpha=0.7)
ax.add_patch(global_box)

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/gmp_bg.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ GMP 背景已生成: gmp_bg.png")
plt.close()

# ========== 简化版本 - 更抽象的表示 ==========

# AvgPool 简化版 - 网格到单格
fig, ax = plt.subplots(figsize=(3, 2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 多个小方块
colors = plt.cm.Blues(np.linspace(0.3, 0.9, 9))
positions = [(i%3, i//3) for i in range(9)]
for i, (x, y) in enumerate(positions):
    rect = Rectangle((0.05 + x*0.12, 0.3 + y*0.12), 0.1, 0.1,
                     facecolor=colors[i], edgecolor='white', linewidth=1)
    ax.add_patch(rect)

# 箭头
ax.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='orange'))

# 结果方块（平均色）
avg_color = plt.cm.Blues(0.6)
result = Rectangle((0.6, 0.35), 0.25, 0.25,
                   facecolor=avg_color, edgecolor='white', linewidth=2)
ax.add_patch(result)

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/avgpool_bg_simple.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ AvgPool 简化版已生成: avgpool_bg_simple.png")
plt.close()

# MaxPool 简化版
fig, ax = plt.subplots(figsize=(3, 2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 多个小方块，一个特别亮
colors = plt.cm.Reds(np.array([0.3, 0.4, 0.35, 0.5, 0.95, 0.45, 0.4, 0.5, 0.3]))
for i, (x, y) in enumerate(positions):
    rect = Rectangle((0.05 + x*0.12, 0.3 + y*0.12), 0.1, 0.1,
                     facecolor=colors[i], edgecolor='white', linewidth=1)
    ax.add_patch(rect)
    if i == 4:  # 高亮最大值
        highlight = Rectangle((0.05 + x*0.12, 0.3 + y*0.12), 0.1, 0.1,
                             facecolor='none', edgecolor='yellow', 
                             linewidth=2.5, linestyle='--')
        ax.add_patch(highlight)

# 箭头
ax.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='orange'))

# 结果方块（最大值色）
result = Rectangle((0.6, 0.35), 0.25, 0.25,
                   facecolor=colors[4], edgecolor='white', linewidth=2)
ax.add_patch(result)

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/maxpool_bg_simple.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ MaxPool 简化版已生成: maxpool_bg_simple.png")
plt.close()

print("\n所有池化操作背景图生成完成！")
print("\n标准版本:")
print("- avgpool_bg.png (平均池化 - 区域平均)")
print("- maxpool_bg.png (最大池化 - 取最大值)")
print("- gap_bg.png (全局平均池化 - 整图平均)")
print("- gmp_bg.png (全局最大池化 - 整图最大)")
print("\n简化版本:")
print("- avgpool_bg_simple.png (简洁版平均池化)")
print("- maxpool_bg_simple.png (简洁版最大池化)")

