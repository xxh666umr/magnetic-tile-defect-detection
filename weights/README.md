# 模型权重

`best.pt` 因为体积原因**不放进 Git 仓库**，需要单独获取。

## 放在哪里

把权重文件放到本目录，命名为：

```
weights/best.pt
```

程序启动时会自动按以下顺序查找，找到即可运行：

1. 环境变量 `MT_MODEL_PATH`
2. `weights/best.pt`
3. `runs/magnetic_tile_defect/weights/best.pt`（自己训练时的默认输出位置）

## 从 Release 下载

在仓库的 **Releases** 页面下载 `best.pt`，解压到本目录即可。

## 自己在本地训练

```bash
python train.py
```

需要准备：

- `datasets/magnetic_tile/data.yaml` 和对应的 `images/`、`labels/` 目录（YOLO 格式）
- `weights/yolov8n.pt`（官方预训练权重，作为迁移学习起点）

训练完成后会把权重写到 `runs/magnetic_tile_defect/weights/best.pt`，直接就能被程序找到。
