# -*- coding: utf-8 -*-
"""
集中配置
========
所有路径与阈值都在这里改，避免在脚本里写死 C:\\Users\\... 这类绝对路径
（写死绝对路径的后果：换一台机器、或者别人 clone 下来，程序直接跑不起来）。

每一项都可以用环境变量覆盖，方便换机器 / 换数据目录。
"""
from __future__ import annotations

import os
from pathlib import Path

# 仓库根目录 = 本文件所在目录，这样从任何工作目录启动都能定位到项目文件
BASE_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# ---------------- 模型权重 ----------------
def _resolve_model_path() -> Path:
    """权重文件不入库（体积大），按以下顺序找：
       1) 环境变量 MT_MODEL_PATH
       2) weights/best.pt
       3) runs/magnetic_tile_defect/weights/best.pt  （训练产物的默认位置）
    """
    explicit = os.environ.get("MT_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser()
    candidates = (
        BASE_DIR / "weights" / "best.pt",
        BASE_DIR / "runs" / "magnetic_tile_defect" / "weights" / "best.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


MODEL_PATH = _resolve_model_path()

# 官方预训练权重（迁移学习的起点，只有训练时才需要）
PRETRAINED_PATH = _env_path("MT_PRETRAINED_PATH", BASE_DIR / "weights" / "yolov8n.pt")

# ---------------- 数据集 ----------------
# YOLO 格式，目录结构：data.yaml + images/{train,val} + labels/{train,val}
DATASET_DIR = _env_path("MT_DATASET_DIR", BASE_DIR / "datasets" / "magnetic_tile")
DATA_YAML = DATASET_DIR / "data.yaml"

# ---------------- 输出 ----------------
RUNS_DIR = _env_path("MT_RUNS_DIR", BASE_DIR / "runs")
RESULT_DIR = _env_path("MT_RESULT_DIR", BASE_DIR / "results")
HISTORY_FILE = _env_path("MT_HISTORY_FILE", BASE_DIR / "history.json")
SUMMARY_FILE = _env_path("MT_SUMMARY_FILE", BASE_DIR / "val_summary.json")

# ---------------- 检测参数 ----------------
CONF_THRESHOLD = float(os.environ.get("MT_CONF", "0.25"))

# 五类缺陷中文名（对应模型的类别序号）
CLASS_CN = {0: "气孔", 1: "崩边", 2: "裂纹", 3: "毛边", 4: "不平整"}
