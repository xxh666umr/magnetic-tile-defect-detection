# -*- coding: utf-8 -*-
"""
磁瓦缺陷检测 —— 模型训练脚本（英文路径版，供复现）
================================
功能：用磁瓦五类缺陷数据集（YOLO格式），在官方预训练权重
      yolov8n.pt 基础上微调，训练出属于"本人"的磁瓦五类检测模型 best.pt。

数据集五类缺陷（datasets/magnetic_tile/data.yaml）：
  0: blowhole  气孔     1: break  崩边     2: crack  裂纹
  3: fray  毛边         4: uneven  不平整
训练集 936 张 / 验证集 202 张 / 测试集 202 张

运行方式：
  python train.py
"""
from ultralytics import YOLO
import torch

# ---------------- 路径配置（统一在 config.py，可用环境变量覆盖） ----------------
from config import DATA_YAML, PRETRAINED_PATH, RUNS_DIR

DATA = str(DATA_YAML)              # 数据集配置
PRETRAINED = str(PRETRAINED_PATH)  # 官方预训练权重（训练起点）
OUTPUT_DIR = str(RUNS_DIR)         # 训练产物输出目录

def main():
    print("=" * 50)
    print("开始训练磁瓦五类缺陷检测模型")
    print("CUDA 可用:", torch.cuda.is_available(), "（当前环境为 CPU，将用 CPU 训练）")
    print("=" * 50)

    # 1. 加载官方预训练权重作为起点（迁移学习，收敛更快、效果更好）
    model = YOLO(PRETRAINED)

    # 2. 开始训练
    model.train(
        data=DATA,            # 数据集配置
        epochs=50,            # 训练轮数（磁瓦数据量小，50 轮足够收敛）
        imgsz=640,            # 输入图片尺寸
        batch=16,             # 每批张数（CPU 训练，内存不足可改成 8）
        device="cpu",         # 指定 CPU（无 GPU 环境）
        project=OUTPUT_DIR,   # 输出目录（英文路径）
        name="magnetic_tile_defect",  # 实验名称
        patience=15,          # 验证指标连续 15 轮不提升则提前停止
        seed=42,              # 固定随机种子，结果可复现
    )

    print("训练完成！最佳模型保存在：")
    print(OUTPUT_DIR + r"\magnetic_tile_defect\weights\best.pt")

if __name__ == "__main__":
    main()
