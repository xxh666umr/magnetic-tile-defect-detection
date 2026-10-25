# -*- coding: utf-8 -*-
"""
磁瓦缺陷检测 —— 模型效果评价脚本（英文路径版）
========================================
功能：用训练好的 best.pt，在验证集 202 张图上做整体评价，输出：
  - 总体 Precision / Recall / F1 / mAP
  - 各类别召回率
  - 工件级 漏放(有缺陷被判OK) / 误杀(无缺陷被判NG) 统计
  - 判定准确率

运行方式（训练完成后）：
  python val.py
"""
from ultralytics import YOLO
import os, json, glob

# ---------------- 路径配置（统一在 config.py，可用环境变量覆盖） ----------------
from config import MODEL_PATH as _MODEL_PATH, DATA_YAML, DATASET_DIR, CONF_THRESHOLD, SUMMARY_FILE

MODEL_PATH = str(_MODEL_PATH)                      # 训练好的模型
DATA = str(DATA_YAML)                              # 数据集配置
VAL_IMAGES = str(DATASET_DIR / "images" / "val")   # 验证集图片
VAL_LABELS = str(DATASET_DIR / "labels" / "val")   # 验证集标注
CONF = CONF_THRESHOLD                              # 置信度阈值（与检测系统保持一致）

# 五类缺陷中文名（便于报告展示）
CLASS_CN = {0: "气孔", 1: "崩边", 2: "裂纹", 3: "毛边", 4: "不平整"}


def load_true_boxes(label_path):
    """读取一张图的真实标注，返回 [(类别, cx, cy, w, h)]"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def main():
    print("=" * 60)
    print("开始评价磁瓦五类检测模型（验证集 202 张）")
    print("=" * 60)

    # 1. 加载训练好的模型
    model = YOLO(MODEL_PATH)

    # 2. 用 ultralytics 自带 val 得到检测指标
    metrics = model.val(data=DATA, split="val", conf=CONF)
    print("\n【总体检测指标】")
    print(f"  Precision(精确率): {metrics.box.mp:.4f}")
    print(f"  Recall(召回率)   : {metrics.box.mr:.4f}")
    print(f"  mAP@0.5          : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95     : {metrics.box.map:.4f}")
    p, r = metrics.box.mp, metrics.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  F1               : {f1:.4f}")

    # 3. 各类别召回率
    print("\n【各类别召回率】")
    cls_names = model.names
    class_r = []
    for cls_id in range(len(cls_names)):
        row = metrics.box.class_result(cls_id)  # 返回该类 [p, r, ap50, ap]
        class_r.append(row)
        print(f"  {cls_names[cls_id]:10s}({CLASS_CN.get(cls_id,'?'):2s}): Recall={row[1]:.4f}")

    # 4. 工件级漏放/误杀统计（按图判定 OK/NG）
    print("\n【工件级判定统计】")
    val_files = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    total = len(val_files)
    true_ng = pred_ng = 0
    miss = false_kill = 0
    ok_ok = ng_ng = 0

    for img_path in val_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(VAL_LABELS, base + ".txt")
        true_boxes = load_true_boxes(label_path)
        is_true_ng = len(true_boxes) > 0       # 真实有缺陷 => NG

        results = model.predict(img_path, conf=CONF, verbose=False)
        has_pred = len(results[0].boxes) > 0   # 检出缺陷框 => 判 NG

        if is_true_ng:
            true_ng += 1
            if has_pred:
                ng_ng += 1
            else:
                miss += 1        # 漏放：有缺陷被判 OK
        else:
            if has_pred:
                false_kill += 1  # 误杀：无缺陷被判 NG
            else:
                ok_ok += 1

    print(f"  验证集图片总数 : {total} 张")
    print(f"  真实含缺陷(NG): {true_ng} 张   真实无缺陷(OK): {total - true_ng} 张")
    print(f"  漏放(有缺陷判成OK): {miss} 张  → 漏放率 {miss/true_ng:.2%}" if true_ng else "  无真实NG")
    print(f"  误杀(无缺陷判成NG): {false_kill} 张")
    correct = ok_ok + ng_ng
    print(f"  工件判定准确率 : {correct}/{total} = {correct/total:.2%}")

    # 5. 汇总保存
    summary = {
        "precision": round(metrics.box.mp, 4),
        "recall": round(metrics.box.mr, 4),
        "f1": round(f1, 4),
        "map50": round(metrics.box.map50, 4),
        "total_images": total,
        "true_ng": true_ng,
        "miss_count": miss,
        "false_kill_count": false_kill,
        "accuracy": round(correct / total, 4),
        "class_recall": {cls_names[i]: round(row[1], 4) for i, row in enumerate(class_r)},
    }
    out = str(SUMMARY_FILE)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n评价结果已保存到：{out}")


if __name__ == "__main__":
    main()
