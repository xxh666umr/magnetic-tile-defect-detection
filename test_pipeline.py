# -*- coding: utf-8 -*-
"""管线验证：加载模型，对示例图预测，输出框/类别/置信度/判定"""
from ultralytics import YOLO
import os, glob

from pathlib import Path

from config import MODEL_PATH, CONF_THRESHOLD

BASE = str(Path(__file__).resolve().parent)
MODEL = str(MODEL_PATH)
CONF = CONF_THRESHOLD
CLASS_CN = {0: "气孔", 1: "崩边", 2: "裂纹", 3: "毛边", 4: "不平整"}

if not os.path.isfile(MODEL):
    raise SystemExit(f"未找到权重：{MODEL}\n请先按 weights/README.md 放置 best.pt")

m = YOLO(MODEL)
print("模型加载成功，类别:", m.names)

# 用示例文件夹里的一张缺陷图 + 一张正常图测试
imgs = sorted(glob.glob(os.path.join(BASE, "samples", "*.jpg")))
crack = [i for i in imgs if "Crack" in i][0]
free = [i for i in imgs if "Free" in i][0]

for path, tag in [(crack, "缺陷图(Crack)"), (free, "正常图(Free)")]:
    print("\n===== " + tag + " =====")
    r = m.predict(path, conf=CONF, verbose=False)[0]
    if len(r.boxes) > 0:
        for b in r.boxes:
            cid = int(b.cls[0])
            print(f"  {CLASS_CN.get(cid, cid)} (置信度 {float(b.conf[0]):.3f})  框={[round(x,1) for x in b.xyxy[0].tolist()]}")
        print("  判定：NG（检出缺陷框）")
    else:
        print("  未检出缺陷框，判定：OK")
