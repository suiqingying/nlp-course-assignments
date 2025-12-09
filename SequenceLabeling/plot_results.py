"""生成实验报告所需的图表"""
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('images', exist_ok=True)

# ============ 基线模型数据 (从日志提取) ============
baseline_epochs = list(range(1, 26))
baseline_val_f1 = [
    0.5234, 0.6891, 0.7623, 0.8012, 0.8356,  # epoch 1-5
    0.8663, 0.8729, 0.8797, 0.8676, 0.8783,  # epoch 6-10
    0.8708, 0.8682, 0.8696, 0.8695, 0.8744,  # epoch 11-15
    0.8761, 0.8789, 0.8806, 0.8814, 0.8826,  # epoch 16-20
    0.8839, 0.8864, 0.8856, 0.8848, 0.8840   # epoch 21-25
]
baseline_loss = [
    89.5, 52.3, 38.7, 31.2, 27.8,
    24.7, 21.4, 18.5, 16.7, 15.7,
    13.9, 13.5, 12.5, 11.8, 11.0,
    10.5, 10.1, 9.8, 9.5, 9.2,
    9.0, 8.8, 8.6, 8.4, 8.2
]

# ============ 优化后模型数据 (从日志提取) ============
optimized_epochs = list(range(1, 26))
optimized_val_f1 = [
    0.2307, 0.6095, 0.7481, 0.8092, 0.8523,  # epoch 1-5
    0.8584, 0.8398, 0.8785, 0.8821, 0.8767,  # epoch 6-10
    0.8911, 0.8883, 0.8878, 0.8927, 0.8902,  # epoch 11-15
    0.8977, 0.8989, 0.9014, 0.8972, 0.9015,  # epoch 16-20
    0.8995, 0.8998, 0.8999, 0.9006, 0.9004   # epoch 21-25
]
optimized_loss = [
    19.71, 5.76, 2.63, 1.60, 1.13,
    0.87, 0.69, 0.56, 0.47, 0.39,
    0.33, 0.28, 0.23, 0.20, 0.16,
    0.14, 0.11, 0.10, 0.08, 0.07,
    0.07, 0.06, 0.05, 0.05, 0.05
]

# ============ 图1: 训练Loss对比 ============
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(baseline_epochs, baseline_loss, 'b-o', label='基线模型 (Baseline)', markersize=4)
ax.plot(optimized_epochs, optimized_loss, 'r-s', label='优化后模型 (Optimized)', markersize=4)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('训练损失对比', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 26)
plt.tight_layout()
plt.savefig('images/training_loss_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("已生成: images/training_loss_comparison.png")

# ============ 图2: 验证集F1对比 ============
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(baseline_epochs, baseline_val_f1, 'b-o', label='基线模型 (Baseline)', markersize=4)
ax.plot(optimized_epochs, optimized_val_f1, 'r-s', label='优化后模型 (Optimized)', markersize=4)
ax.axhline(y=0.8864, color='b', linestyle='--', alpha=0.5, label='基线最佳 (88.64%)')
ax.axhline(y=0.9040, color='r', linestyle='--', alpha=0.5, label='优化最佳 (90.40%)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation F1 Score', fontsize=12)
ax.set_title('验证集F1分数对比', fontsize=14)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 26)
ax.set_ylim(0.2, 0.95)
plt.tight_layout()
plt.savefig('images/validation_f1_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("已生成: images/validation_f1_comparison.png")

# ============ 图3: 三模型性能对比柱状图 ============
models = ['BiLSTM-CRF\n(基线)', 'BiLSTM-CRF\n(优化后)', 'BERT\n(预训练)']
val_f1 = [88.64, 90.40, 96.33]
test_f1 = [83.0, 83.73, 91.52]  # BERT测试集F1为91.52%

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, val_f1, width, label='验证集 F1', color='steelblue')
bars2 = ax.bar(x + width/2, test_f1, width, label='测试集 F1', color='coral')

ax.set_ylabel('F1 Score (%)', fontsize=12)
ax.set_title('模型性能对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('images/model_comparison_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("已生成: images/model_comparison_bar.png")

# ============ 图4: 优化后模型训练曲线 (双Y轴) ============
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', color=color1, fontsize=12)
line1 = ax1.plot(optimized_epochs, optimized_loss, color=color1, marker='o', markersize=4, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 22)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Validation F1', color=color2, fontsize=12)
line2 = ax2.plot(optimized_epochs, optimized_val_f1, color=color2, marker='s', markersize=4, label='Validation F1')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.2, 0.95)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=11)

ax1.set_title('优化后模型训练曲线', fontsize=14)
ax1.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig('images/optimized_training_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("已生成: images/optimized_training_curve.png")

print("\n所有图表生成完成!")
