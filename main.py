# -*- coding: utf-8 -*-
"""
磁瓦缺陷检测系统  (GUI 主程序)
==========================================
功能：
  1. 打开单张图片 / 选择文件夹批量检测
  2. 检测显示缺陷框、类别、置信度和 OK/NG 判定
  3. 批量处理：进度条 + 每张结果 + 点击列表查看
  4. 保存结果图片与检测记录，关闭再打开仍可查看历史
  5. 异常提示：图片打不开、没选图、模型找不到等，失败不算 OK

判定规则：有符合置信度阈值的缺陷框 => NG；没有 => OK。
  "没检出"不代表一定没有缺陷，会在结果中说明。

运行方式（老师 WinPython 环境）：
  WinPython\python.exe main.py
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import json
import datetime
from ultralytics import YOLO

# ================== 配置 ==================
CONF_THRESHOLD = 0.25                 # 置信度阈值：框置信度 >= 阈值才算有效缺陷
MODEL_PATH = r"C:\Users\lenovo\Desktop\VisionAI_Lab\MT_defect_system\runs\magnetic_tile_defect\weights\best.pt"
HISTORY_FILE = r"C:\Users\lenovo\Desktop\VisionAI_Lab\MT_defect_system\history.json"
RESULT_DIR = r"C:\Users\lenovo\Desktop\VisionAI_Lab\MT_defect_system\results"

# 五类缺陷中文名（对应模型的类别序号）
CLASS_CN = {0: "气孔", 1: "崩边", 2: "裂纹", 3: "毛边", 4: "不平整"}


class DefectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("磁瓦缺陷检测系统")
        self.root.geometry("1020x720")

        # 状态变量
        self.current_image_path = None
        self.current_results = None
        self.model = None
        self.history = []
        self.batch_images = []
        self.batch_results = []

        # 创建结果/历史文件目录
        os.makedirs(RESULT_DIR, exist_ok=True)

        # 加载模型 + 读取历史
        self._load_model()
        self._load_history()

        # 搭建界面
        self._build_ui()

    # ================== 模型与历史 ==================
    def _load_model(self):
        """加载模型，模型找不到时明确提示（失败不算 OK）"""
        try:
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror("模型未找到",
                                     f"模型文件不存在：\n{MODEL_PATH}\n\n请检查模型是否已训练并放到该路径。")
                self.model = None
                return
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("模型加载失败", f"加载模型出错：{str(e)}")
            self.model = None

    def _load_history(self):
        """读取历史检测记录（关闭系统后再打开仍可查看）"""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def _save_history(self):
        """把历史记录写回文件"""
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("保存失败", f"历史记录保存出错：{str(e)}")

    # ================== 界面 ==================
    def _build_ui(self):
        # 顶部按钮栏
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        btns = [
            ("打开单张图片", self.open_single_image),
            ("选择文件夹批量检测", self.open_batch_folder),
            ("开始检测", self.detect_single),
            ("保存当前结果", self.save_current_result),
            ("查看历史记录", self.show_history_window),
        ]
        for text, cmd in btns:
            ttk.Button(top, text=text, command=cmd).pack(side=tk.LEFT, padx=4)

        # 主区域：左侧图片，右侧结果
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 左侧图片显示
        self.img_label = tk.Label(main, text="请选择一张磁瓦图片\n点击「打开单张图片」",
                                  font=("微软雅黑", 12), bg="#e8e8e8", relief=tk.SUNKEN)
        self.img_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # 右侧结果面板
        right = ttk.Frame(main, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)

        ttk.Label(right, text="检测详情", font=("微软雅黑", 12, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.result_text = tk.Text(right, height=15, wrap=tk.WORD, font=("微软雅黑", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)

        ttk.Label(right, text="批量处理进度").pack(anchor=tk.W, pady=(8, 2))
        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill=tk.X)

        ttk.Label(right, text="批量结果列表（点击可查看）").pack(anchor=tk.W, pady=(8, 2))
        self.batch_listbox = tk.Listbox(right, height=10, font=("微软雅黑", 9))
        self.batch_listbox.pack(fill=tk.BOTH, expand=True)
        self.batch_listbox.bind("<<ListboxSelect>>", self.on_batch_item_click)

    # ================== 单图检测 ==================
    def open_single_image(self):
        """选择并显示单张图片"""
        path = filedialog.askopenfilename(
            title="选择图片", filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            img = Image.open(path)
            img.thumbnail((620, 520))
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img, text="", bg="#f0f0f0")
            self.current_image_path = path
            self.current_results = None
            self._set_result(f"已加载：{os.path.basename(path)}\n点击「开始检测」查看结果")
        except Exception as e:
            # 图片打不开：明确提示，不把失败显示成 OK
            messagebox.showerror("图片打开失败", f"无法读取图片：\n{path}\n\n原因：{str(e)}")
            self._set_result(f"图片打开失败：{str(e)}")

    def detect_single(self):
        """对当前图片执行检测"""
        if not self.current_image_path:
            messagebox.showwarning("未选图片", "请先选择一张图片再点击「开始检测」")
            return
        if self.model is None:
            messagebox.showerror("模型未加载", "模型未加载，无法检测。\n请先确认 best.pt 模型文件存在。")
            return
        try:
            results = self.model.predict(self.current_image_path, conf=CONF_THRESHOLD)
            self.current_results = self._parse_results(results, self.current_image_path)

            # 绘制检测框并显示
            boxed = results[0].plot()
            boxed = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(boxed)
            img_pil.thumbnail((620, 520))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.img_label.config(image=self.tk_img, text="")

            self._show_result_text(self.current_results)
        except Exception as e:
            messagebox.showerror("检测失败", f"检测过程出错：{str(e)}")
            self._set_result(f"检测失败：{str(e)}")

    # ================== 批量检测 ==================
    def open_batch_folder(self):
        """选择文件夹并批量检测"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.batch_images = [os.path.join(folder, f) for f in os.listdir(folder)
                             if f.lower().endswith(exts)]
        if not self.batch_images:
            # 空文件夹 / 无图片：提示
            messagebox.showwarning("未找到图片", "该文件夹中没有找到 jpg/png 等图片文件。")
            return

        self.batch_listbox.delete(0, tk.END)
        self.batch_results = []
        self.progress["value"] = 0
        self.progress["maximum"] = len(self.batch_images)
        self._run_batch_detect()

    def _run_batch_detect(self):
        """循环处理批量图片，实时更新进度条和列表"""
        if self.model is None:
            messagebox.showerror("模型未加载", "模型未加载，无法批量检测。")
            return
        self.batch_results = []
        for i, img_path in enumerate(self.batch_images):
            try:
                results = self.model.predict(img_path, conf=CONF_THRESHOLD)
                res = self._parse_results(results, img_path)
            except Exception as e:
                # 单张失败也记录下来，不中断批处理，且明确标记失败
                res = self._make_error_result(img_path, str(e))
            self.batch_results.append(res)
            self.batch_listbox.insert(tk.END, f"{os.path.basename(img_path)}  →  {res['judge']}")
            self.progress["value"] = i + 1
            self.root.update()  # 刷新界面

        messagebox.showinfo("批量完成", f"批量检测结束，共处理 {len(self.batch_images)} 张图片。")

    def on_batch_item_click(self, event):
        """点击批量列表项，显示对应图片与结果"""
        sel = self.batch_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        res = self.batch_results[idx]
        if "error" in res:
            self.img_label.config(image="", text="图片读取失败", bg="#e8e8e8")
            self._set_result(f"文件：{res['filename']}\n读取失败：{res['error']}")
            return
        try:
            results = self.model.predict(res["path"], conf=CONF_THRESHOLD)
            boxed = results[0].plot()
            boxed = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(boxed)
            img_pil.thumbnail((620, 520))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.img_label.config(image=self.tk_img, text="")
            # 把当前查看的批量图设为"当前结果"，使其可被"保存当前结果"保存
            self.current_image_path = res["path"]
            self.current_results = res
            self._show_result_text(res)
        except Exception as e:
            messagebox.showerror("查看失败", f"加载该图片结果失败：{str(e)}")

    # ================== 结果解析与保存 ==================
    def _parse_results(self, results, img_path):
        """解析 YOLO 输出为结构化结果，并给出 OK/NG 判定"""
        boxes = results[0].boxes
        defects = []
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names.get(cls_id, str(cls_id))
                cls_cn = CLASS_CN.get(cls_id, cls_name)   # 转中文
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                defects.append({
                    "category": cls_cn,
                    "category_en": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in bbox],
                })
        # 判定：有缺陷框 => NG，无 => OK
        judge = "NG" if len(defects) > 0 else "OK"
        return {
            "path": img_path,
            "filename": os.path.basename(img_path),
            "judge": judge,
            "defects": defects,
            "conf_threshold": CONF_THRESHOLD,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _make_error_result(self, img_path, err):
        return {
            "path": img_path,
            "filename": os.path.basename(img_path),
            "judge": "读取失败",
            "defects": [],
            "error": err,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _show_result_text(self, res):
        """在右侧面板显示检测详情"""
        text = f"文件：{res['filename']}\n"
        text += f"时间：{res['time']}\n"
        text += f"判定：{res['judge']}\n"
        text += "-" * 30 + "\n"
        if res["judge"] == "NG":
            text += f"检出缺陷 {len(res['defects'])} 处：\n"
            for i, d in enumerate(res["defects"], 1):
                text += f"  {i}. {d['category']}  (置信度 {d['confidence']})\n"
        elif res["judge"] == "OK":
            text += "未检出符合阈值的缺陷框。\n※ 注意：「没有检出」不代表一定没有缺陷。\n"
        else:
            text += f"错误信息：{res.get('error', '未知')}\n"
        self._set_result(text)

    def _set_result(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

    def save_current_result(self):
        """保存当前结果（带框图片 + 历史记录），时间戳命名避免覆盖旧结果"""
        if not self.current_results or "error" in self.current_results:
            messagebox.showwarning("无可保存结果", "请先对一张图片成功检测后再保存。")
            return
        try:
            # 时间戳命名，多次检测不覆盖旧结果
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(RESULT_DIR, f"{ts}_{self.current_results['filename']}")

            # 重新检测并绘制检测框（保证保存结果与界面一致）
            results = self.model.predict(self.current_image_path, conf=CONF_THRESHOLD)
            cv2.imwrite(save_path, results[0].plot())

            # 写入历史记录
            record = self.current_results.copy()
            record["save_image_path"] = save_path
            self.history.append(record)
            self._save_history()

            messagebox.showinfo("保存成功", f"结果图片已保存：\n{save_path}\n\n记录已写入历史，可关闭系统后再次查看。")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存出错：{str(e)}")

    # ================== 历史记录 ==================
    def show_history_window(self):
        """打开历史记录窗口，可双击查看详情"""
        if not self.history:
            messagebox.showinfo("历史记录", "暂无历史记录。\n保存检测结果后即可在此查看。")
            return
        win = tk.Toplevel(self.root)
        win.title("历史检测记录")
        win.geometry("560x400")

        listbox = tk.Listbox(win, font=("微软雅黑", 10))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for rec in self.history:
            listbox.insert(tk.END, f"{rec['time']}  |  {rec['filename']}  |  {rec['judge']}")

        def show_detail(event):
            sel = listbox.curselection()
            if not sel:
                return
            rec = self.history[sel[0]]
            dw = tk.Toplevel(win)
            dw.title("记录详情")
            dw.geometry("420x320")
            t = tk.Text(dw, wrap=tk.WORD, font=("微软雅黑", 10))
            t.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            content = f"文件：{rec['filename']}\n时间：{rec['time']}\n判定：{rec['judge']}\n"
            content += f"结果图：{rec.get('save_image_path', '未保存')}\n" + "-" * 30 + "\n"
            if rec.get("defects"):
                for i, d in enumerate(rec["defects"], 1):
                    content += f"{i}. {d['category']}  置信度 {d['confidence']}\n"
            else:
                content += "未检出缺陷框\n"
            t.insert(tk.END, content)
            t.config(state=tk.DISABLED)

        listbox.bind("<Double-1>", show_detail)


if __name__ == "__main__":
    root = tk.Tk()
    app = DefectDetectionApp(root)
    root.mainloop()
