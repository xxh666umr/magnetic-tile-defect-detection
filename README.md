 # 磁瓦表面缺陷检测系统（Magnetic Tile Defect Detection）

基于 **YOLOv8** 的磁瓦表面缺陷检测系统，面向工业质检场景，可对磁瓦生产过程中的常见表面缺陷进行自动识别与判定（OK / NG），支持单张检测与文件夹批量检测。

## 功能特性

- 支持**单张图片检测**与**文件夹批量检测**，带实时进度条
- 覆盖 5 类常见磁瓦缺陷：气孔（blowhole）、崩缺（break）、裂纹（crack）、磨损（fray）、不均匀（uneven）
- 自动判定 OK / NG，输出缺陷类别、置信度与位置框
- 检测历史记录本地持久化（history.json），支持随时回看
- **CPU 即可运行，无需 GPU**

## 技术栈

| 组件 | 版本 |
| ---- | ---- |
| Python | 3.12 |
| ultralytics（YOLOv8） | 8.4.137 |
| PyTorch | 2.13.0（CPU） |
| OpenCV | 4.14.0 |

## 环境要求

- Python 3.12
- 依赖库：`ultralytics`、`torch`、`opencv-python`、`pillow`、`numpy`

## 快速开始

```bash
# 1. 安装依赖
pip install ultralytics torch opencv-python pillow numpy

# 2. 运行系统
python main.py
