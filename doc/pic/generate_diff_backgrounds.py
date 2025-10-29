import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
import matplotlib.patches as mpatches

# 设置无边框，高分辨率
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ========== L2 Distance 背景 - 渐变标尺 ==========
fig, ax = plt.subplots(figsize=(4, 2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 创建蓝色到红色的渐变
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, aspect='auto', cmap='coolwarm', 
          extent=[0.1, 0.9, 0.3, 0.7])

# 添加轻微的边框
rect = Rectangle((0.1, 0.3), 0.8, 0.4, 
                 linewidth=2, edgecolor='white', 
                 facecolor='none', alpha=0.5)
ax.add_patch(rect)

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/l2_distance_bg.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ L2 Distance 背景已生成: l2_distance_bg.png")
plt.close()

# ========== Cosine Distance 背景 - 角度扇形 ==========
fig, ax = plt.subplots(figsize=(4, 2))
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.axis('off')
ax.set_aspect('equal')

# 创建渐变扇形
theta_range = 60  # 角度范围
center = (0.2, 0.5)
radius = 0.6

# 绘制多个扇形创造渐变效果
num_segments = 50
colors = plt.cm.twilight(np.linspace(0.2, 0.8, num_segments))

for i in range(num_segments):
    theta1 = i * theta_range / num_segments
    theta2 = (i + 1) * theta_range / num_segments
    wedge = Wedge(center, radius, theta1, theta2, 
                  facecolor=colors[i], edgecolor='none', alpha=0.8)
    ax.add_patch(wedge)

# 添加轻微的外轮廓
wedge_outline = Wedge(center, radius, 0, theta_range,
                      facecolor='none', edgecolor='white', 
                      linewidth=2, alpha=0.5)
ax.add_patch(wedge_outline)

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/cosine_distance_bg.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ Cosine Distance 背景已生成: cosine_distance_bg.png")
plt.close()

# ========== L2 Distance 背景 v2 - 高斯热力图风格 ==========
fig, ax = plt.subplots(figsize=(4, 2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 创建高斯分布热力图
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# 中心高斯
Z = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)

ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', 
          cmap='YlOrRd', alpha=0.9)

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/l2_distance_bg_v2.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ L2 Distance 背景 v2 已生成: l2_distance_bg_v2.png")
plt.close()

# ========== Cosine Distance 背景 v2 - 简洁扇形 ==========
fig, ax = plt.subplots(figsize=(4, 2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_aspect('equal')

# 中心圆形渐变
center = (0.5, 0.5)
theta_range = 90

# 使用紫色系渐变
num_segments = 40
colors = plt.cm.viridis(np.linspace(0.3, 0.9, num_segments))

for i in range(num_segments):
    theta1 = -45 + i * theta_range / num_segments
    theta2 = -45 + (i + 1) * theta_range / num_segments
    wedge = Wedge(center, 0.4, theta1, theta2, 
                  facecolor=colors[i], edgecolor='none', alpha=0.85)
    ax.add_patch(wedge)

# 外轮廓
wedge_outline = Wedge(center, 0.4, -45, 45,
                      facecolor='none', edgecolor='white', 
                      linewidth=2.5, alpha=0.6)
ax.add_patch(wedge_outline)

plt.tight_layout(pad=0)
plt.savefig('doc/GOLD/cosine_distance_bg_v2.png', 
            transparent=True, bbox_inches='tight', pad_inches=0.05)
print("✓ Cosine Distance 背景 v2 已生成: cosine_distance_bg_v2.png")
plt.close()

print("\n所有背景图生成完成！")
print("- l2_distance_bg.png (蓝红渐变标尺)")
print("- l2_distance_bg_v2.png (高斯热力图)")
print("- cosine_distance_bg.png (彩色扇形)")
print("- cosine_distance_bg_v2.png (紫色简洁扇形)")

