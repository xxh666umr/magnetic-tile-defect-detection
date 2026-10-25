# -*- coding: utf-8 -*-
"""生成磁瓦缺陷检测系统功能流程图 PNG"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(9, 12))
ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis('off')

def box(cx, cy, w, h, text, color='#dbe8fb'):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle='round,pad=0.12', fc=color, ec='#2f5597', lw=1.6))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=11.5)

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color='#404040', lw=1.8))

W = 4.4; H = 1.0
# 纵向主流程（中心 x=5）
box(5, 13.0, W, H, '启动系统')
box(5, 11.6, W, H, '加载模型 best.pt')
# 分支
box(2.7, 10.0, 3.0, H, '单图检测\n(打开图片)', '#fce4d6')
box(7.3, 10.0, 3.0, H, '批量检测\n(选择文件夹)', '#fce4d6')
# 合流
box(5, 8.5, W, H, 'YOLO 推理检测\n(输出缺陷框、类别、置信度)')
box(5, 7.0, W, H, '判定 OK / NG\n(有缺陷框→NG，无→OK)')
box(5, 5.5, W, H, '显示结果\n(框、类别、置信度、进度条)')
box(5, 4.0, W, H, '保存结果\n(带框图片 + JSON 历史记录)')
box(5, 2.5, W, H, '关闭系统后重新打开\n可查看历史记录', '#e2efda')

# 箭头
arrow(5, 12.5, 5, 12.1)      # 启动→加载
arrow(5, 11.1, 2.7, 10.5)    # 加载→单图
arrow(5, 11.1, 7.3, 10.5)    # 加载→批量
arrow(2.7, 9.5, 5, 9.0)      # 单图→推理
arrow(7.3, 9.5, 5, 9.0)      # 批量→推理
arrow(5, 8.0, 5, 7.5)        # 推理→判定
arrow(5, 6.5, 5, 6.0)        # 判定→显示
arrow(5, 5.0, 5, 4.5)        # 显示→保存
arrow(5, 3.5, 5, 3.0)        # 保存→历史

ax.set_title('磁瓦缺陷检测系统 主要功能流程图', fontsize=15, pad=10)
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = str(OUT_DIR / "flow.png")
plt.savefig(out, dpi=200, bbox_inches='tight')
print("流程图已生成:", out)
