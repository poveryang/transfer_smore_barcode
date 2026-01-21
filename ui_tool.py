"""
相机标定UI工具
用于采集两个相机的图像，标定外参，测试坐标转换
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import os
from typing import Optional, Tuple, List
import paramiko
import json

from epiceye_camera import EpicEyeCamera
from smore_camera import SmoreCamera
from calibration import CameraCalibration
import epiceye

# 设备参数（用于发送/接收标定配置）
DEVICE_USERNAME = "smore"
DEVICE_PORT = 22101
DEVICE_PASSWORD = "smore123456"
DEVICE_CONFIG_NAME = "transfer_smore_calib.json"
DEVICE_BASE_DIR = "/usr/scanner"
UI_CONFIG_FILE = "ui_config.json"  # UI配置文件名


class CameraCalibrationUI:
    """相机标定UI工具主类"""
    
    def __init__(self, root):
        """
        初始化UI
        
        Args:
            root: tkinter根窗口
        """
        self.root = root
        self.root.title("相机外参标定工具")
        
        # 根据屏幕分辨率自动调整窗口大小
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # 基准分辨率（设计时的分辨率）
        base_width = 2560
        base_height = 1440
        
        # 计算缩放比例（取宽度和高度的较小比例，确保窗口不会超出屏幕）
        scale_x = screen_width / base_width
        scale_y = screen_height / base_height
        scale = min(scale_x, scale_y, 1.0)  # 不超过1.0，避免窗口过大
        
        # 基准窗口大小
        base_window_width = 1300
        base_window_height = 1200
        
        # 计算实际窗口大小
        window_width = int(base_window_width * scale)
        window_height = int(base_window_height * scale)
        
        # 计算窗口居中位置
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # 设置窗口大小和位置
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 相机对象
        self.camera_3d: Optional[EpicEyeCamera] = None
        self.camera_barcode: Optional[SmoreCamera] = None
        
        # 标定对象
        self.calibration = CameraCalibration()
        
        # 图像显示相关
        self.image_3d: Optional[np.ndarray] = None
        self.image_barcode: Optional[np.ndarray] = None
        self.display_image_3d: Optional[np.ndarray] = None
        self.display_image_barcode: Optional[np.ndarray] = None
        self.depth_map: Optional[np.ndarray] = None  # 3D相机的深度图
        self.saved_depth_map: Optional[np.ndarray] = None  # 标定时保存的深度图
        self.transformed_roi: Optional[Tuple[int, int, int, int]] = None  # 转换后的ROI
        
        # 四个点选择相关（左上、右上、左下、右下）
        self.selected_points: List[Optional[Tuple[int, int]]] = [None, None, None, None]  # 四个点：左上、右上、左下、右下
        self.current_point_index: int = 0  # 当前要选择的点索引（0-3）
        self.transformed_points: List[Optional[Tuple[float, float]]] = [None, None, None, None]  # 转换后的四个点
        
        
        # 创建UI
        self.create_ui()
        
        # 加载UI配置（IP地址等）
        self.load_ui_config()
        
        # 尝试加载已保存的标定参数
        self.calibration.load_calibration()
    
    def load_ui_config(self):
        """加载UI配置（IP地址等）"""
        if not os.path.exists(UI_CONFIG_FILE):
            return
        
        try:
            with open(UI_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'ip_3d' in config:
                    self.ip_3d_var.set(config['ip_3d'])
                if 'ip_barcode' in config:
                    self.ip_barcode_var.set(config['ip_barcode'])
        except Exception as e:
            print(f"加载UI配置失败: {e}")
    
    def save_ui_config(self):
        """保存UI配置（IP地址等）"""
        try:
            config = {
                'ip_3d': self.ip_3d_var.get().strip(),
                'ip_barcode': self.ip_barcode_var.get().strip()
            }
            with open(UI_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存UI配置失败: {e}")
    
    def _get_depth_from_neighborhood(self, depth_map: np.ndarray, x: int, y: int, search_radius: int = 5) -> float:
        """
        从周围区域获取有效深度值（平均值）
        
        Args:
            depth_map: 深度图
            x, y: 中心点坐标
            search_radius: 搜索半径（像素）
            
        Returns:
            float: 有效深度值的平均值，如果没有有效值则返回0
        """
        h, w = depth_map.shape[:2]
        
        # 计算搜索区域边界
        y_min = max(0, y - search_radius)
        y_max = min(h, y + search_radius + 1)
        x_min = max(0, x - search_radius)
        x_max = min(w, x + search_radius + 1)
        
        # 提取周围区域的深度值
        neighborhood = depth_map[y_min:y_max, x_min:x_max]
        
        # 过滤有效深度值（> 0）
        valid_depths = neighborhood[neighborhood > 0]
        
        if len(valid_depths) > 0:
            # 返回平均值
            return float(np.mean(valid_depths))
        else:
            return 0.0
        
    def create_ui(self):
        """创建UI界面"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 标定控制
        calib_frame = ttk.LabelFrame(control_frame, text="外参标定（棋盘格）", padding="10")
        calib_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 棋盘格参数设置
        ttk.Label(calib_frame, text="棋盘格参数:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(calib_frame, text="内部角点数 (列×行):").grid(row=1, column=0, sticky=tk.W, pady=5)
        pattern_frame = ttk.Frame(calib_frame)
        pattern_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.pattern_cols_var = tk.StringVar(value="11")
        self.pattern_rows_var = tk.StringVar(value="8")
        ttk.Entry(pattern_frame, textvariable=self.pattern_cols_var, width=5).grid(row=0, column=0)
        ttk.Label(pattern_frame, text="×").grid(row=0, column=1, padx=2)
        ttk.Entry(pattern_frame, textvariable=self.pattern_rows_var, width=5).grid(row=0, column=2)
        
        ttk.Label(calib_frame, text="方格尺寸 (mm):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.square_size_var = tk.StringVar(value="15.0")
        ttk.Entry(calib_frame, textvariable=self.square_size_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(calib_frame, text="(例如: 9×6 表示9列6行内部角点)", 
                 font=("", 8), foreground="gray").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # 检测棋盘格按钮
        ttk.Button(calib_frame, text="检测棋盘格", command=self.detect_chessboard).grid(
            row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 标定按钮
        ttk.Button(calib_frame, text="标定外参", command=self.calibrate_extrinsic).grid(
            row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 测试控制
        test_frame = ttk.LabelFrame(control_frame, text="测试外参", padding="10")
        test_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(test_frame, text="在3D相机图像上选择四个点:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        
        point_labels_frame = ttk.Frame(test_frame)
        point_labels_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        self.point_status_label = ttk.Label(point_labels_frame, text="请点击图像选择: 左上角", 
                                           font=("", 9), foreground="blue")
        self.point_status_label.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(test_frame, text="清除所有点", command=self.clear_points).grid(
            row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 转换方法选择
        ttk.Label(test_frame, text="转换方法:").grid(
            row=3, column=0, sticky=tk.W, pady=5)
        
        method_frame = ttk.Frame(test_frame)
        method_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.transform_method_var = tk.StringVar(value="3D转换")
        
        # 3D转换选项
        rb_3d = ttk.Radiobutton(method_frame, text="3D转换（推荐）", variable=self.transform_method_var, 
                       value="3D转换", command=self._on_transform_method_changed)
        rb_3d.grid(row=0, column=0, sticky=tk.W)
        
        # 单应性矩阵选项和平面深度输入放在一起
        homography_frame = ttk.Frame(method_frame)
        homography_frame.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        rb_homography = ttk.Radiobutton(homography_frame, text="单应性矩阵", variable=self.transform_method_var, 
                       value="单应性矩阵", command=self._on_transform_method_changed)
        rb_homography.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(homography_frame, text="深度(mm):").grid(
            row=0, column=1, sticky=tk.W, padx=(5, 2))
        self.plane_depth_var = tk.StringVar(value="")
        self.plane_depth_entry = ttk.Entry(homography_frame, textvariable=self.plane_depth_var, width=10)
        self.plane_depth_entry.grid(row=0, column=2, sticky=tk.W)
        
        # 提示文字
        ttk.Label(test_frame, text="(无深度图时需输入深度；单应性矩阵用于平面场景，精度较低)", 
                 font=("", 8), foreground="gray").grid(
            row=5, column=0, sticky=tk.W, pady=2)
        
        # 深度图控制
        depth_frame = ttk.Frame(test_frame)
        depth_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(depth_frame, text="重新采集深度图", command=self.capture_depth_map).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        depth_status_label = ttk.Label(depth_frame, text="", font=("", 8), foreground="gray")
        depth_status_label.grid(row=0, column=1, sticky=tk.W)
        self.depth_status_label = depth_status_label
        
        ttk.Button(test_frame, text="测试坐标转换", command=self.test_transform).grid(
            row=7, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 初始化：保持深度输入可用
        self._on_transform_method_changed()
        
        # 初始化深度图状态显示
        self._update_depth_status()
        
        # 标定参数管理
        param_frame = ttk.LabelFrame(control_frame, text="标定参数管理", padding="10")
        param_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(param_frame, text="设备IP:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.device_ip_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self.device_ip_var, width=18).grid(
            row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(param_frame, text=f"端口: {DEVICE_PORT}", font=("", 8), foreground="gray").grid(
            row=0, column=2, sticky=tk.W, pady=5, padx=(5, 0))
        
        param_frame.columnconfigure(0, weight=1)
        param_frame.columnconfigure(1, weight=1)
        
        ttk.Button(param_frame, text="保存到文件", command=self.save_calibration).grid(
            row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        
        ttk.Button(param_frame, text="从文件加载", command=self.load_calibration).grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        ttk.Button(param_frame, text="上传到设备", command=self.send_calibration_to_device).grid(
            row=2, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        
        ttk.Button(param_frame, text="从设备下载", command=self.receive_calibration_from_device).grid(
            row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # 日志显示
        status_frame = ttk.LabelFrame(control_frame, text="日志", padding="10")
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.S))
        control_frame.rowconfigure(4, weight=1)
        
        self.status_text = tk.Text(status_frame, height=42, width=30, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # 右侧图像显示区域（上下排布）
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # 3D相机图像显示区域（上方）
        camera_3d_frame = ttk.LabelFrame(image_frame, text="3D相机图像", padding="10")
        camera_3d_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        camera_3d_frame.columnconfigure(0, weight=1)
        camera_3d_frame.rowconfigure(0, weight=1)
        
        self.canvas_3d = tk.Canvas(camera_3d_frame, width=640, height=360, bg="black")
        self.canvas_3d.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas_3d.bind("<Button-1>", self.on_canvas_3d_click)
        self.canvas_3d.bind("<B1-Motion>", self.on_canvas_3d_drag)
        self.canvas_3d.bind("<ButtonRelease-1>", self.on_canvas_3d_release)
        
        # 3D相机控制区域（分为左右两部分）
        control_3d_frame = ttk.Frame(camera_3d_frame)
        control_3d_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        camera_3d_frame.columnconfigure(0, weight=1)
        control_3d_frame.columnconfigure(0, weight=1)
        control_3d_frame.columnconfigure(1, weight=1)
        
        # 左侧：IP连接和采集图像（一行）
        left_3d_frame = ttk.LabelFrame(control_3d_frame, text="采集图像", padding="5")
        left_3d_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Label(left_3d_frame, text="IP:").grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
        self.ip_3d_var = tk.StringVar(value="")
        ip_3d_entry = ttk.Entry(left_3d_frame, textvariable=self.ip_3d_var, width=18)
        ip_3d_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 3))
        # 绑定事件，当IP地址改变时自动保存（延迟保存，避免频繁写入）
        self._save_config_timer = None
        ip_3d_entry.bind("<FocusOut>", lambda e: self.save_ui_config())
        ip_3d_entry.bind("<Return>", lambda e: self.save_ui_config())
        # ttk.Label(left_3d_frame, text="(可带端口，如 169.254.188.22:5000，留空自动搜索)", font=("", 7), foreground="gray").grid(
        #     row=0, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Button(left_3d_frame, text="连接", command=self.connect_3d_camera, width=8).grid(
            row=0, column=3, padx=(0, 3))
        ttk.Button(left_3d_frame, text="采集", command=self.capture_3d_image, width=8).grid(
            row=0, column=4)
        
        # 右侧：加载图像
        right_3d_frame = ttk.LabelFrame(control_3d_frame, text="加载图像", padding="5")
        right_3d_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        ttk.Button(right_3d_frame, text="加载本地图像", command=self.load_local_image_3d).grid(
            row=0, column=0, sticky=(tk.W, tk.E))
        
        # 读码器相机图像显示区域（下方）
        camera_barcode_frame = ttk.LabelFrame(image_frame, text="读码器相机图像", padding="10")
        camera_barcode_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        camera_barcode_frame.columnconfigure(0, weight=1)
        camera_barcode_frame.rowconfigure(0, weight=1)
        
        self.canvas_barcode = tk.Canvas(camera_barcode_frame, width=640, height=360, bg="black")
        self.canvas_barcode.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 读码器相机控制区域（分为左右两部分）
        control_barcode_frame = ttk.Frame(camera_barcode_frame)
        control_barcode_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        camera_barcode_frame.columnconfigure(0, weight=1)
        control_barcode_frame.columnconfigure(0, weight=1)
        control_barcode_frame.columnconfigure(1, weight=1)
        
        # 左侧：IP连接和采集图像（一行）
        left_barcode_frame = ttk.LabelFrame(control_barcode_frame, text="采集图像", padding="5")
        left_barcode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Label(left_barcode_frame, text="IP:").grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
        self.ip_barcode_var = tk.StringVar(value="")
        ip_barcode_entry = ttk.Entry(left_barcode_frame, textvariable=self.ip_barcode_var, width=18)
        ip_barcode_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 3))
        # 绑定事件，当IP地址改变时自动保存（延迟保存，避免频繁写入）
        ip_barcode_entry.bind("<FocusOut>", lambda e: self.save_ui_config())
        ip_barcode_entry.bind("<Return>", lambda e: self.save_ui_config())
        # ttk.Label(left_barcode_frame, text="(功能待实现)", font=("", 7), foreground="gray").grid(
        #     row=0, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Button(left_barcode_frame, text="连接", command=self.connect_barcode_camera, width=8).grid(
            row=0, column=3, padx=(0, 3))
        ttk.Button(left_barcode_frame, text="采集", command=self.capture_barcode_image, width=8).grid(
            row=0, column=4)
        
        # 右侧：加载图像
        right_barcode_frame = ttk.LabelFrame(control_barcode_frame, text="加载图像", padding="5")
        right_barcode_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        ttk.Button(right_barcode_frame, text="加载本地图像", command=self.load_local_image_barcode).grid(
            row=0, column=0, sticky=(tk.W, tk.E))
        
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)
        
        self.log("UI初始化完成")
    
    def log(self, message: str):
        """在状态文本框中添加日志（优化性能，避免频繁UI更新）"""
        self.status_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.status_text.see(tk.END)
        # 移除 update_idletasks()，减少UI更新频率，提高性能
        # 只在必要时更新（如用户操作时）
    
    def log_with_update(self, message: str):
        """在状态文本框中添加日志并更新UI（用于重要消息）"""
        self.status_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def connect_3d_camera(self):
        """连接3D相机"""
        ip_3d = self.ip_3d_var.get().strip()
        
        if ip_3d:
            self.log(f"正在连接3D相机: {ip_3d}")
            self.camera_3d = EpicEyeCamera(ip=ip_3d)
        else:
            # 如果IP为空，尝试自动搜索
            self.log("未输入IP，尝试自动搜索3D相机...")
            self.camera_3d = EpicEyeCamera(ip=None)
        
        if self.camera_3d.connect():
            self.log("3D相机连接成功")
            self.ip_3d_var.set(self.camera_3d.ip)
            self.save_ui_config()  # 保存连接成功后的IP地址
            # 内参已在连接时自动加载（camera_capture.py中）
            camera_matrix, distortion = self.camera_3d.get_intrinsics()
            if camera_matrix is not None:
                self.log("3D相机内参已加载")
            else:
                self.log("警告: 3D相机内参未获取")
        else:
            self.log("3D相机连接失败")
            self.camera_3d = None
    
    def connect_barcode_camera(self):
        """连接读码器相机"""
        ip_barcode = self.ip_barcode_var.get().strip()
        
        if not ip_barcode:
            messagebox.showwarning("警告", "请输入读码器相机IP地址")
            return
        
        self.log(f"正在连接读码器相机: {ip_barcode}")
        self.camera_barcode = SmoreCamera(ip=ip_barcode)
        if self.camera_barcode.connect():
            self.log("读码器相机连接成功")
            self.save_ui_config()  # 保存连接成功后的IP地址
        else:
            self.log("读码器相机连接失败")
            self.camera_barcode = None
    
    def load_local_image_3d(self):
        """加载本地图像到3D相机显示区域"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    messagebox.showerror("错误", f"无法读取图像文件: {file_path}")
                    return
                
                self.image_3d = image.copy()
                self.update_display_3d()
                self.log(f"3D相机图像已加载: {file_path}, 尺寸: {image.shape}")
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败: {e}")
                self.log(f"加载图像失败: {e}")
    
    def load_local_image_barcode(self):
        """加载本地图像到读码器相机显示区域"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    messagebox.showerror("错误", f"无法读取图像文件: {file_path}")
                    return
                
                self.image_barcode = image.copy()
                self.update_display_barcode()
                self.log(f"读码器相机图像已加载: {file_path}, 尺寸: {image.shape}")
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败: {e}")
                self.log(f"加载图像失败: {e}")
    
    
    def capture_3d_image(self):
        """采集3D相机图像"""
        if self.camera_3d is None:
            messagebox.showwarning("警告", "请先连接3D相机")
            return
        
        self.log("采集3D相机图像...")
        # 在采集前更新UI，确保日志显示
        self.root.update_idletasks()
        
        img = self.camera_3d.capture_image(auto_reconnect=True)
        if img is not None:
            self.image_3d = img.copy()
            self.update_display_3d()
            self.log(f"3D相机图像采集成功，尺寸: {img.shape}")
        else:
            self.log("3D相机图像采集失败")
            self.log("提示: 如果网络不稳定，请检查网线连接，然后重新连接相机")
            messagebox.showerror("错误", 
                "3D相机图像采集失败\n\n"
                "可能的原因：\n"
                "1. 网络连接不稳定（请检查网线）\n"
                "2. 浏览器中打开了相机的可视化工具\n"
                "3. 其他程序正在使用相机\n\n"
                "建议：重新连接相机后重试")
    
    def capture_barcode_image(self):
        """采集读码器相机图像"""
        if self.camera_barcode is None:
            messagebox.showwarning("警告", "请先连接读码器相机")
            return
        
        self.log("采集读码器相机图像...")
        img = self.camera_barcode.capture_image()
        if img is not None:
            self.image_barcode = img.copy()
            self.update_display_barcode()
            self.log(f"读码器相机图像采集成功，尺寸: {img.shape}")
        else:
            self.log("读码器相机图像采集失败")
            messagebox.showerror("错误", "读码器相机图像采集失败")
    
    def update_display_3d(self):
        """更新3D相机图像显示（优化性能）"""
        if self.image_3d is None:
            return
        
        # 检查是否有选中的点需要绘制
        has_points = any(p is not None for p in self.selected_points)
        
        if has_points:
            # 需要绘制点时创建副本
            display_img = self.image_3d.copy()
            
            # 根据图像分辨率计算点大小和线宽（与分辨率成比例）
            h, w = self.image_3d.shape[:2]
            base_width = 1920
            base_height = 1200
            width_ratio = w / base_width
            height_ratio = h / base_height
            resolution_ratio = (width_ratio + height_ratio) / 2
            base_line_width = 4  # 与之前ROI的线宽一致
            line_width = max(2, int(base_line_width * resolution_ratio))
            point_radius = max(5, int(8 * resolution_ratio))
            
            # 定义四个点的颜色和标签（使用英文简写）
            point_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # 绿、蓝、红、青
            point_labels = ["TL", "TR", "BL", "BR"]  # Top-Left, Top-Right, Bottom-Left, Bottom-Right
            
            # 绘制已选择的点
            valid_points = []
            for i, point in enumerate(self.selected_points):
                if point is not None:
                    x, y = point
                    # 绘制点
                    cv2.circle(display_img, (x, y), point_radius, point_colors[i], -1)
                    cv2.circle(display_img, (x, y), point_radius + 2, (255, 255, 255), 2)
                    # 绘制标签
                    label_pos = (x + point_radius + 5, y - point_radius - 5)
                    cv2.putText(display_img, point_labels[i], label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_img, point_labels[i], label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, point_colors[i], 1)
                    valid_points.append((x, y, i))
            
            # 如果至少有两个点，绘制连接线（按顺序：左上->右上->右下->左下->左上）
            if len(valid_points) >= 2:
                # 按顺序连接点：左上(0) -> 右上(1) -> 右下(3) -> 左下(2) -> 左上(0)
                point_order = [0, 1, 3, 2]  # 左上、右上、右下、左下
                points_dict = {i: (x, y) for x, y, i in valid_points}
                
                # 绘制连接线
                for idx in range(len(point_order)):
                    curr_idx = point_order[idx]
                    next_idx = point_order[(idx + 1) % len(point_order)]
                    if curr_idx in points_dict and next_idx in points_dict:
                        pt1 = points_dict[curr_idx]
                        pt2 = points_dict[next_idx]
                        cv2.line(display_img, pt1, pt2, (0, 255, 255), line_width)
            
            self.display_image_3d = display_img
            self._update_canvas(self.canvas_3d, display_img)
        else:
            # 没有点时直接使用原图像，避免不必要的复制
            self.display_image_3d = self.image_3d
            self._update_canvas(self.canvas_3d, self.image_3d)
    
    def update_display_barcode(self):
        """更新读码器相机图像显示"""
        if self.image_barcode is None:
            return
        
        display_img = self.image_barcode.copy()
        self.display_image_barcode = display_img
        self._update_canvas(self.canvas_barcode, display_img)
    
    def _draw_chessboard_corners_custom(self, img: np.ndarray, pattern_size: Tuple[int, int], corners: np.ndarray, is_barcode: bool = False):
        """
        自定义绘制棋盘格角点，使点和线更粗更大
        
        Args:
            img: 要绘制的图像
            pattern_size: 棋盘格尺寸 (cols, rows)
            corners: 角点坐标
            is_barcode: 是否是读码器图像（读码器图像的点更大）
        """
        cols, rows = pattern_size
        
        # 根据图像分辨率计算点的大小（与分辨率成比例）
        h, w = img.shape[:2]
        # 基准分辨率（3D相机典型分辨率：1920x1200）
        base_width = 1920
        base_height = 1200
        
        # 计算分辨率比例（使用宽度和高度比例的平均值）
        width_ratio = w / base_width
        height_ratio = h / base_height
        resolution_ratio = (width_ratio + height_ratio) / 2
        
        # 基准点大小（对应1920x1200分辨率）
        base_point_radius = 6
        base_outline_radius = 8
        
        # 根据分辨率比例计算实际点大小
        point_radius = int(base_point_radius * resolution_ratio)
        outline_radius = int(base_outline_radius * resolution_ratio)
        
        # 确保最小尺寸
        point_radius = max(3, point_radius)
        outline_radius = max(4, outline_radius)
        
        # 只绘制角点，不绘制连接线
        # 绘制角点：起点红色，终点绿色，其他点蓝色
        for idx, corner in enumerate(corners):
            pt = tuple(corner.ravel().astype(int))
            
            # 确定点的颜色
            if idx == 0:
                # 起点：红色
                color = (0, 0, 255)  # BGR格式，红色
            elif idx == len(corners) - 1:
                # 终点：绿色
                color = (0, 255, 0)  # BGR格式，绿色
            else:
                # 其他点：蓝色
                color = (255, 0, 0)  # BGR格式，蓝色
            
            # 绘制实心圆（主色）
            cv2.circle(img, pt, point_radius, color, -1)
            # 绘制外圈（白色边框，更清晰）
            cv2.circle(img, pt, outline_radius, (255, 255, 255), 2)
    
    def _on_transform_method_changed(self):
        """当转换方法改变时，保持深度输入框可用"""
        self.plane_depth_entry.config(state="normal")
    
    def _update_depth_status(self):
        """更新深度图状态显示"""
        if self.saved_depth_map is not None:
            h, w = self.saved_depth_map.shape[:2]
            self.depth_status_label.config(text=f"已保存 ({w}x{h})", foreground="green")
        else:
            self.depth_status_label.config(text="未保存", foreground="gray")
    
    def capture_depth_map(self):
        """手动采集深度图"""
        if self.camera_3d is None:
            messagebox.showwarning("警告", "请先连接3D相机")
            return
        
        if self.image_3d is None:
            messagebox.showwarning("警告", "请先采集或加载3D相机图像")
            return
        
        self.log("开始采集深度图...")
        try:
            frame_id = epiceye.trigger_frame(ip=self.camera_3d.ip, pointcloud=True)
            if frame_id:
                depth_map = self.camera_3d.capture_depth(frame_id)
                if depth_map is not None:
                    self.saved_depth_map = depth_map.copy()
                    self.depth_map = depth_map.copy()
                    h, w = depth_map.shape[:2]
                    self.log(f"深度图采集成功，尺寸: {w}x{h}")
                    self._update_depth_status()
                    messagebox.showinfo("成功", f"深度图采集成功！\n尺寸: {w}x{h}\n\n测试外参时将使用此深度图")
                else:
                    self.log("深度图采集失败")
                    messagebox.showerror("错误", "深度图采集失败")
            else:
                self.log("无法触发拍摄")
                messagebox.showerror("错误", "无法触发拍摄获取深度图")
        except Exception as e:
            self.log(f"采集深度图时出错: {e}")
            messagebox.showerror("错误", f"采集深度图失败：{e}")
    
    def _update_canvas(self, canvas: tk.Canvas, image: np.ndarray):
        """更新画布显示（优化性能，减少不必要的计算）"""
        if image is None:
            return
        
        try:
            # 获取画布尺寸（只在必要时查询）
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 640
                canvas_height = 360
            
            h, w = image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 只在尺寸变化时进行缩放（简单缓存）
            if hasattr(canvas, '_last_size') and canvas._last_size == (new_w, new_h, w, h):
                # 尺寸没变，跳过缩放（但图像内容可能变了，所以还是需要更新）
                pass
            
            resized = cv2.resize(image, (new_w, new_h))
            canvas._last_size = (new_w, new_h, w, h)
            
            # 转换为RGB格式
            if len(resized.shape) == 3:
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                resized_rgb = resized
            
            # 转换为PIL Image然后转换为PhotoImage
            pil_image = Image.fromarray(resized_rgb)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # 更新画布
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
            canvas.image = photo  # 保持引用
            
        except Exception as e:
            print(f"更新画布出错: {e}")
    
    def on_canvas_3d_click(self, event):
        """3D相机画布点击事件 - 选择四个点"""
        if self.image_3d is None:
            return
        
        # 将画布坐标转换为图像坐标
        canvas_width = self.canvas_3d.winfo_width()
        canvas_height = self.canvas_3d.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        img_h, img_w = self.image_3d.shape[:2]
        scale = min(canvas_width / img_w, canvas_height / img_h)
        
        x = int((event.x - canvas_width / 2) / scale + img_w / 2)
        y = int((event.y - canvas_height / 2) / scale + img_h / 2)
        
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        
        # 选择四个点：左上(0)、右上(1)、左下(2)、右下(3)
        point_labels = ["左上角", "右上角", "左下角", "右下角"]
        
        # 设置当前点
        self.selected_points[self.current_point_index] = (x, y)
        self.log(f"已选择{point_labels[self.current_point_index]}: ({x}, {y})")
        
        # 移动到下一个点
        self.current_point_index = (self.current_point_index + 1) % 4
        
        # 更新状态标签
        if all(p is not None for p in self.selected_points):
            self.point_status_label.config(text="四个点已选择完成", foreground="green")
        else:
            next_label = point_labels[self.current_point_index]
            self.point_status_label.config(text=f"请点击图像选择: {next_label}", foreground="blue")
        
        # 更新显示
        self.update_display_3d()
    
    def on_canvas_3d_drag(self, event):
        """3D相机画布拖拽事件（不再使用）"""
        pass
    
    def on_canvas_3d_release(self, event):
        """3D相机画布释放事件（不再使用）"""
        pass
    
    def clear_points(self):
        """清除所有选择的点"""
        self.selected_points = [None, None, None, None]
        self.current_point_index = 0
        self.transformed_points = [None, None, None, None]
        self.point_status_label.config(text="请点击图像选择: 左上角", foreground="blue")
        self.update_display_3d()
        self.log("已清除所有点")
    
    def detect_chessboard(self):
        """检测两个图像中的棋盘格"""
        if self.image_3d is None or self.image_barcode is None:
            messagebox.showwarning("警告", "请先加载或采集两个相机的图像")
            return
        
        try:
            # 获取棋盘格参数
            pattern_cols = int(self.pattern_cols_var.get())
            pattern_rows = int(self.pattern_rows_var.get())
            pattern_size = (pattern_cols, pattern_rows)
            
            self.log(f"开始检测棋盘格，参数: {pattern_cols}×{pattern_rows}")
            
            # 检测3D相机图像中的棋盘格
            ret1, corners1 = self.calibration.detect_chessboard(self.image_3d, pattern_size)
            if ret1:
                # 在显示图像上绘制检测到的角点（不修改原始图像）
                img1_display = self.image_3d.copy()
                # 使用自定义绘制，使点和线更粗更大（3D相机图像，点较小）
                self._draw_chessboard_corners_custom(img1_display, pattern_size, corners1, is_barcode=False)
                self.display_image_3d = img1_display
                self._update_canvas(self.canvas_3d, img1_display)
                self.log(f"3D相机图像: 检测到棋盘格，角点数: {len(corners1)}")
            else:
                self.log("3D相机图像: 未检测到棋盘格")
                messagebox.showwarning("警告", "3D相机图像中未检测到棋盘格")
                return
            
            # 检测读码器相机图像中的棋盘格
            ret2, corners2 = self.calibration.detect_chessboard(self.image_barcode, pattern_size)
            if ret2:
                # 在显示图像上绘制检测到的角点（不修改原始图像）
                img2_display = self.image_barcode.copy()
                # 使用自定义绘制，使点和线更粗更大（读码器图像，点更大）
                self._draw_chessboard_corners_custom(img2_display, pattern_size, corners2, is_barcode=True)
                self.display_image_barcode = img2_display
                self._update_canvas(self.canvas_barcode, img2_display)
                self.log(f"读码器相机图像: 检测到棋盘格，角点数: {len(corners2)}")
            else:
                self.log("读码器相机图像: 未检测到棋盘格")
                messagebox.showwarning("警告", "读码器相机图像中未检测到棋盘格")
                return
            
            messagebox.showinfo("成功", "两个图像中都检测到棋盘格！\n可以点击'标定外参'进行标定")
            
        except ValueError as e:
            messagebox.showerror("错误", f"棋盘格参数格式错误: {e}")
        except Exception as e:
            messagebox.showerror("错误", f"检测棋盘格失败: {e}")
            self.log(f"检测失败: {e}")
    
    def calibrate_extrinsic(self):
        """使用棋盘格标定外参"""
        # 检查是否有两个相机的图像
        if self.image_3d is None:
            messagebox.showwarning("警告", "请先连接3D相机或加载3D相机图像")
            return
        
        if self.image_barcode is None:
            messagebox.showwarning("警告", "请先连接读码器相机或加载读码器相机图像")
            return
        
        # 获取棋盘格参数
        try:
            pattern_cols = int(self.pattern_cols_var.get())
            pattern_rows = int(self.pattern_rows_var.get())
            pattern_size = (pattern_cols, pattern_rows)
            square_size = float(self.square_size_var.get())
        except ValueError as e:
            messagebox.showerror("错误", f"棋盘格参数格式错误: {e}")
            return
        
        # 获取相机内参
        camera1_matrix = None
        camera1_distortion = None
        camera2_matrix = None
        camera2_distortion = None
        
        if self.camera_3d:
            camera1_matrix, camera1_distortion = self.camera_3d.get_intrinsics()
            if camera1_matrix is None:
                self.log("提示: 3D相机没有内参，将在标定过程中自动进行单目标定")
        else:
            # 如果没有连接相机，但加载了图像，也可以进行标定（会自动单目标定）
            self.log("提示: 3D相机未连接，如果加载了图像，将在标定过程中自动进行单目标定")
        
        if self.camera_barcode:
            # TODO: 获取读码器相机内参
            pass
        
        # 读码器相机内参是可选的
        # 如果没有内参，会在标定过程中自动进行单目标定
        if camera2_matrix is None:
            self.log("提示: 读码器相机没有内参，将在标定过程中自动标定内参")
        
        # 确保内参矩阵格式正确（在传递给标定函数之前）
        if camera1_matrix is not None:
            if isinstance(camera1_matrix, list):
                camera1_matrix = np.array(camera1_matrix, dtype=np.float64)
            else:
                camera1_matrix = np.array(camera1_matrix, dtype=np.float64)
            
            # 如果是1D数组（9个元素），reshape为3x3
            if camera1_matrix.ndim == 1:
                if len(camera1_matrix) == 9:
                    camera1_matrix = camera1_matrix.reshape(3, 3)
                    self.log("相机1内参矩阵已从1D数组reshape为3x3")
                else:
                    messagebox.showerror("错误", f"相机1内参矩阵格式错误: 期望9个元素，得到{len(camera1_matrix)}个")
                    return
            
            # 确保是3x3矩阵
            if camera1_matrix.shape != (3, 3):
                messagebox.showerror("错误", f"相机1内参矩阵尺寸错误: 期望(3,3)，得到{camera1_matrix.shape}")
                return
            
            # 记录内参矩阵信息用于调试
            self.log(f"相机1内参矩阵:\n{camera1_matrix}")
        
        if camera2_matrix is not None:
            if isinstance(camera2_matrix, list):
                camera2_matrix = np.array(camera2_matrix, dtype=np.float64)
            else:
                camera2_matrix = np.array(camera2_matrix, dtype=np.float64)
            
            if camera2_matrix.ndim == 1:
                if len(camera2_matrix) == 9:
                    camera2_matrix = camera2_matrix.reshape(3, 3)
                    self.log("相机2内参矩阵已从1D数组reshape为3x3")
        
        # 执行棋盘格标定
        self.log(f"开始棋盘格标定，参数: {pattern_cols}×{pattern_rows}, 方格尺寸: {square_size}mm")
        
        success, msg = self.calibration.calibrate_with_chessboard(
            self.image_3d, self.image_barcode,
            pattern_size, square_size,
            camera1_matrix, camera2_matrix,
            camera1_distortion, camera2_distortion
        )
        
        if success:
            self.log(f"标定成功: {msg}")
            
            # 标定成功后，尝试采集并保存深度图
            if self.camera_3d and self.image_3d is not None:
                self.log("标定成功，正在采集深度图...")
                try:
                    frame_id = epiceye.trigger_frame(ip=self.camera_3d.ip, pointcloud=True)
                    if frame_id:
                        depth_map = self.camera_3d.capture_depth(frame_id)
                        if depth_map is not None:
                            self.saved_depth_map = depth_map.copy()
                            self.log(f"深度图采集成功，尺寸: {depth_map.shape}")
                            self.log("深度图已保存，测试外参时将优先使用此深度图")
                            self._update_depth_status()
                        else:
                            self.log("深度图采集失败，测试外参时将需要重新采集")
                            self._update_depth_status()
                    else:
                        self.log("无法触发拍摄获取深度图，测试外参时将需要重新采集")
                        self._update_depth_status()
                except Exception as e:
                    self.log(f"采集深度图时出错: {e}，测试外参时将需要重新采集")
                    self._update_depth_status()
            else:
                self.log("3D相机未连接，无法采集深度图，测试外参时将需要重新采集")
                self._update_depth_status()
            
            messagebox.showinfo("成功", f"外参标定成功！\n{msg}\n可以点击'保存标定参数'保存结果")
        else:
            self.log(f"标定失败: {msg}")
            messagebox.showerror("失败", f"外参标定失败！\n{msg}")
    
    def load_calibration(self):
        """加载标定参数"""
        file_path = filedialog.askopenfilename(
            title="选择标定参数文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            if self.calibration.load_calibration(file_path):
                self.log(f"标定参数加载成功: {file_path}")
                
                # 尝试加载关联的深度图
                depth_file_path = file_path.replace('.json', '_depth.npy')
                if os.path.exists(depth_file_path):
                    try:
                        self.saved_depth_map = np.load(depth_file_path)
                        self.log(f"深度图加载成功: {depth_file_path}, 尺寸: {self.saved_depth_map.shape}")
                        self._update_depth_status()
                    except Exception as e:
                        self.log(f"深度图加载失败: {e}")
                        self.saved_depth_map = None
                        self._update_depth_status()
                else:
                    self.log("未找到关联的深度图文件")
                    self.saved_depth_map = None
                    self._update_depth_status()
            else:
                self.log(f"标定参数加载失败: {file_path}")
    
    def save_calibration(self):
        """保存标定参数"""
        if not self.calibration.is_calibrated():
            messagebox.showwarning("警告", "没有可保存的标定参数")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存标定参数",
            defaultextension=".json",
            initialfile=DEVICE_CONFIG_NAME,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            if self.calibration.save_calibration(file_path):
                self.log(f"标定参数保存成功: {file_path}")
                
                # 如果有关联的深度图，也保存深度图
                if self.saved_depth_map is not None:
                    depth_file_path = file_path.replace('.json', '_depth.npy')
                    try:
                        np.save(depth_file_path, self.saved_depth_map)
                        self.log(f"深度图保存成功: {depth_file_path}")
                    except Exception as e:
                        self.log(f"深度图保存失败: {e}")
            else:
                self.log(f"标定参数保存失败: {file_path}")

    def _get_device_ip(self) -> Optional[str]:
        ip = self.device_ip_var.get().strip()
        if not ip:
            messagebox.showwarning("警告", "请输入设备IP地址")
            return None
        return ip

    def _get_device_path(self, file_name: str) -> str:
        return f"{DEVICE_BASE_DIR}/{file_name}"

    def send_calibration_to_device(self):
        """发送标定参数到设备"""
        ip = self._get_device_ip()
        if not ip:
            return
        
        local_path = filedialog.askopenfilename(
            title="选择要上传的标定文件",
            initialfile=DEVICE_CONFIG_NAME,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not local_path:
            return
        
        file_name = os.path.basename(local_path)
        remote_path = self._get_device_path(file_name)
        if not os.path.exists(local_path):
            if not self.calibration.is_calibrated():
                messagebox.showwarning("警告", "本地没有配置文件，且当前未完成标定")
                return
            if self.calibration.save_calibration(local_path):
                self.log(f"已生成配置文件: {local_path}")
            else:
                self.log(f"配置文件保存失败: {local_path}")
                messagebox.showerror("失败", "保存配置文件失败，无法发送到设备")
                return
        
        client = None
        sftp = None
        try:
            self.log(f"正在连接设备: {ip}:{DEVICE_PORT}")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                ip,
                port=DEVICE_PORT,
                username=DEVICE_USERNAME,
                password=DEVICE_PASSWORD,
                timeout=10
            )
            sftp = client.open_sftp()
            
            # 确保目标目录存在
            try:
                sftp.stat(DEVICE_BASE_DIR)
            except Exception:
                self.log(f"设备路径不存在，尝试创建: {DEVICE_BASE_DIR}")
                sftp.mkdir(DEVICE_BASE_DIR)
            
            sftp.put(local_path, remote_path)
            self.log(f"配置文件已发送: {remote_path}")
            messagebox.showinfo("成功", "配置文件已发送到设备")
        except Exception as e:
            self.log(f"发送配置文件失败: {e}")
            messagebox.showerror("失败", f"发送配置文件失败: {e}")
        finally:
            if sftp is not None:
                sftp.close()
            if client is not None:
                client.close()

    def receive_calibration_from_device(self):
        """从设备接收标定参数"""
        ip = self._get_device_ip()
        if not ip:
            return
        
        local_path = filedialog.asksaveasfilename(
            title="保存下载的标定文件",
            defaultextension=".json",
            initialfile=DEVICE_CONFIG_NAME,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not local_path:
            return
        
        file_name = os.path.basename(local_path)
        remote_path = self._get_device_path(file_name)
        client = None
        sftp = None
        try:
            self.log(f"正在连接设备: {ip}:{DEVICE_PORT}")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                ip,
                port=DEVICE_PORT,
                username=DEVICE_USERNAME,
                password=DEVICE_PASSWORD,
                timeout=10
            )
            sftp = client.open_sftp()
            sftp.get(remote_path, local_path)
            self.log(f"配置文件已接收: {local_path}")
            
            if self.calibration.load_calibration(local_path):
                self.log("标定参数已加载到当前UI")
            else:
                self.log("标定参数加载失败")
            
            messagebox.showinfo("成功", "已从设备接收配置文件")
        except Exception as e:
            self.log(f"接收配置文件失败: {e}")
            messagebox.showerror("失败", f"接收配置文件失败: {e}")
        finally:
            if sftp is not None:
                sftp.close()
            if client is not None:
                client.close()
    
    def show_3d_camera_intrinsics(self):
        """显示3D相机内参"""
        if self.camera_3d is None:
            messagebox.showwarning("警告", "请先连接3D相机")
            return
        
        camera_matrix, distortion = self.camera_3d.get_intrinsics()
        
        if camera_matrix is None:
            self.log("3D相机内参: 未获取到内参矩阵")
            messagebox.showinfo("3D相机内参", "未获取到内参矩阵\n可能原因：\n1. 相机不支持获取内参\n2. 网络连接问题")
            return
        
        # 转换为numpy数组
        if isinstance(camera_matrix, list):
            camera_matrix = np.array(camera_matrix)
        if camera_matrix.ndim == 1:
            camera_matrix = camera_matrix.reshape(3, 3)
        
        # 构建显示信息
        info_text = "=" * 50 + "\n"
        info_text += "3D相机内参矩阵\n"
        info_text += "=" * 50 + "\n\n"
        info_text += "相机内参矩阵 (Camera Matrix):\n"
        info_text += "-" * 50 + "\n"
        info_text += f"{camera_matrix}\n\n"
        info_text += "参数说明:\n"
        info_text += f"  fx (x方向焦距): {camera_matrix[0, 0]:.4f}\n"
        info_text += f"  fy (y方向焦距): {camera_matrix[1, 1]:.4f}\n"
        info_text += f"  cx (主点x坐标): {camera_matrix[0, 2]:.4f}\n"
        info_text += f"  cy (主点y坐标): {camera_matrix[1, 2]:.4f}\n"
        
        if distortion is not None:
            if isinstance(distortion, list):
                distortion = np.array(distortion)
            info_text += "\n畸变参数 (Distortion Coefficients):\n"
            info_text += "-" * 50 + "\n"
            info_text += f"{distortion}\n"
            info_text += f"\n畸变参数数量: {len(distortion)}\n"
        else:
            info_text += "\n畸变参数: 未获取到\n"
        
        info_text += "=" * 50 + "\n"
        
        # 显示到日志
        self.log("\n" + info_text)
        
        # 显示到对话框
        messagebox.showinfo("3D相机内参", 
            f"内参矩阵:\n{camera_matrix}\n\n"
            f"fx: {camera_matrix[0, 0]:.4f}\n"
            f"fy: {camera_matrix[1, 1]:.4f}\n"
            f"cx: {camera_matrix[0, 2]:.4f}\n"
            f"cy: {camera_matrix[1, 2]:.4f}\n\n"
            f"畸变参数: {'已获取' if distortion is not None else '未获取'}")
    
    def test_transform(self):
        """测试坐标转换（可以使用相机图像或加载的本地图像）"""
        if not self.calibration.is_calibrated():
            messagebox.showwarning("警告", "请先标定外参")
            return
        
        # 检查是否选择了四个点
        if not all(p is not None for p in self.selected_points):
            messagebox.showwarning("警告", "请先在3D相机图像上选择四个点（左上、右上、左下、右下）")
            return
        
        # 检查是否有读码器相机图像（无论是从相机采集的还是加载的本地图像）
        if self.image_barcode is None:
            messagebox.showwarning("警告", "请先连接读码器相机或加载读码器相机图像")
            return
        
        self.log("开始坐标转换测试...")
        
        # 获取相机内参（优先从连接的相机获取，如果没有则使用标定时保存的内参）
        camera1_matrix = None
        camera1_distortion = None
        camera2_matrix = None
        camera2_distortion = None
        
        if self.camera_3d:
            # 如果连接了相机，从相机获取内参
            camera1_matrix, camera1_distortion = self.camera_3d.get_intrinsics()
            if camera1_matrix is None:
                self.log("警告: 3D相机内参未获取，尝试使用标定时保存的内参")
        else:
            self.log("3D相机未连接，尝试使用标定时保存的内参")
        
        # 如果没有从相机获取到内参，尝试使用标定时保存的内参
        if camera1_matrix is None:
            if self.calibration.camera1_matrix is not None:
                camera1_matrix = self.calibration.camera1_matrix
                camera1_distortion = self.calibration.camera1_distortion
                # 确保畸变参数是numpy数组
                if camera1_distortion is not None:
                    if isinstance(camera1_distortion, list):
                        camera1_distortion = np.array(camera1_distortion, dtype=np.float32)
                    elif not isinstance(camera1_distortion, np.ndarray):
                        camera1_distortion = np.array(camera1_distortion, dtype=np.float32)
                self.log("使用标定时保存的3D相机内参")
            else:
                messagebox.showerror("错误", 
                    "无法获取3D相机内参！\n\n"
                    "可能的原因：\n"
                    "1. 未连接3D相机且标定时未保存内参\n"
                    "2. 标定时使用了相机内参但未保存\n\n"
                    "解决方案：\n"
                    "1. 连接3D相机以获取内参\n"
                    "2. 重新进行标定（标定会自动保存内参）")
                return
        
        # 读码器相机内参（优先从连接的相机获取，其次使用标定时保存的内参，最后使用估算值）
        if self.camera_barcode:
            # TODO: 获取读码器相机内参
            pass
        
        # 如果没有从相机获取到内参，尝试使用标定时保存的内参
        if camera2_matrix is None:
            if self.calibration.camera2_matrix is not None:
                camera2_matrix = self.calibration.camera2_matrix.copy()
                camera2_distortion = self.calibration.camera2_distortion.copy() if self.calibration.camera2_distortion is not None else None
                self.log("使用标定时保存的读码器相机内参")
                
                # 验证保存的内参是否合理
                if isinstance(camera2_matrix, list):
                    camera2_matrix = np.array(camera2_matrix, dtype=np.float64)
                if camera2_matrix.ndim == 1:
                    camera2_matrix = camera2_matrix.reshape(3, 3)
                
                h2, w2 = self.image_barcode.shape[:2]
                fx = camera2_matrix[0, 0]
                fy = camera2_matrix[1, 1]
                cx = camera2_matrix[0, 2]
                cy = camera2_matrix[1, 2]
                
                # 检查内参是否合理
                if fx > w2 * 5 or fy > h2 * 5 or cx > w2 * 2 or cy > h2 * 2:
                    self.log(f"警告: 保存的读码器相机内参异常，使用估算值")
                    self.log(f"  焦距: fx={fx:.2f}, fy={fy:.2f}, 图像尺寸={w2}x{h2}")
                    self.log(f"  主点: cx={cx:.2f}, cy={cy:.2f}")
                    camera2_matrix = None  # 重置，使用估算值
            else:
                # 使用估算值
                self.log("读码器相机内参未获取，使用估算值")
                h, w = self.image_barcode.shape[:2]
                fx = fy = max(w, h) * 0.8
                cx, cy = w / 2, h / 2
                camera2_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float64)
                camera2_distortion = np.zeros(4)
        
        # 确保内参矩阵格式正确
        if isinstance(camera1_matrix, list):
            camera1_matrix = np.array(camera1_matrix, dtype=np.float64)
        if camera1_matrix.ndim == 1:
            camera1_matrix = camera1_matrix.reshape(3, 3)
        
        if isinstance(camera2_matrix, list):
            camera2_matrix = np.array(camera2_matrix, dtype=np.float64)
        if camera2_matrix.ndim == 1:
            camera2_matrix = camera2_matrix.reshape(3, 3)
        
        # 优先尝试使用2D平面转换（无需深度图）
        # 如果ROI在一个平面上，可以使用单应性矩阵进行转换
        use_planar_transform = True
        # 注意：平面深度的单位必须与外参矩阵中平移向量的单位一致
        # 如果标定时使用毫米作为单位（棋盘格方格尺寸），那么这里也应该是毫米
        # 记录内参矩阵信息用于调试
        self.log(f"相机1内参矩阵:\n{camera1_matrix}")
        self.log(f"相机2内参矩阵:\n{camera2_matrix}")
        self.log(f"外参矩阵:\n{self.calibration.extrinsic_matrix}")
        
        # 尝试获取深度图（优先使用保存的深度图，如果没有则重新采集）
        depth_map = None
        plane_depth = None  # 平面深度，只能从相机获取，不使用默认值
        use_planar_transform = False  # 默认不使用平面转换
        
        # 优先使用保存的深度图（如果存在且图像尺寸匹配）
        if self.saved_depth_map is not None and self.image_3d is not None:
            h_saved, w_saved = self.saved_depth_map.shape[:2]
            h_img, w_img = self.image_3d.shape[:2]
            if h_saved == h_img and w_saved == w_img:
                depth_map = self.saved_depth_map.copy()
                self.depth_map = depth_map
                self.log(f"使用保存的深度图，尺寸: {depth_map.shape}")
            else:
                self.log(f"保存的深度图尺寸({h_saved}x{w_saved})与当前图像尺寸({h_img}x{w_img})不匹配，将重新采集")
        
        # 如果没有保存的深度图或尺寸不匹配，尝试重新采集
        if depth_map is None and self.camera_3d and self.image_3d is not None:
            self.log("尝试重新采集深度图...")
            try:
                # 触发新的拍摄以获取深度图
                frame_id = epiceye.trigger_frame(ip=self.camera_3d.ip, pointcloud=True)
                if frame_id:
                    depth_map = self.camera_3d.capture_depth(frame_id)
                    if depth_map is not None:
                        self.depth_map = depth_map
                        self.log(f"深度图采集成功，尺寸: {depth_map.shape}")
                        # 如果有深度图，计算四个点的平均深度用于平面转换（作为参考）
                        valid_depths = []
                        for point in self.selected_points:
                            if point is not None:
                                x, y = point
                                if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                                    depth = depth_map[y, x]
                                    if depth > 0:
                                        valid_depths.append(depth)
                        if len(valid_depths) > 0:
                            # 使用四个点的平均深度（作为参考）
                            plane_depth = float(np.mean(valid_depths))
                            self.log(f"四个点平均深度: {plane_depth:.2f}mm (范围: {np.min(valid_depths):.2f} - {np.max(valid_depths):.2f}mm)")
                            # 检查深度值的一致性
                            depth_std = np.std(valid_depths)
                            if depth_std > 50:  # 如果标准差超过50mm，说明深度不一致
                                self.log(f"警告: 四个点深度差异较大（标准差: {depth_std:.2f}mm），建议使用3D转换方法")
                        else:
                            self.log(f"四个点深度值无效，将使用3D转换方法")
                    else:
                        self.log("深度图获取失败，将使用3D转换方法")
                else:
                    self.log("无法触发拍摄获取深度图，将使用3D转换方法")
            except Exception as e:
                self.log(f"获取深度图时出错: {e}，将使用3D转换方法")
        elif depth_map is None:
            self.log("3D相机未连接，无法获取深度值，将使用3D转换方法（需要估算深度）")
        
        # 转换四个点
        point_labels = ["TL", "TR", "BL", "BR"]  # Top-Left, Top-Right, Bottom-Left, Bottom-Right
        point_labels_cn = ["左上", "右上", "左下", "右下"]  # 用于日志显示
        self.log(f"开始转换四个点:")
        for i, point in enumerate(self.selected_points):
            if point is not None:
                self.log(f"  {point_labels_cn[i]}: {point}")
        
        # 读取UI输入的深度（平面/估计深度）
        ui_plane_depth_str = self.plane_depth_var.get().strip()

        # 检查用户选择的转换方法
        transform_method = self.transform_method_var.get()
        self.log(f"选择的转换方法: {transform_method}")
        
        # 转换四个点
        transformed_points_list = []
        use_planar_transform = False
        plane_depth = None
        
        # 如果用户选择了单应性矩阵转换，检查是否有平面深度值
        if transform_method == "单应性矩阵":
            if ui_plane_depth_str:
                try:
                    ui_plane_depth = float(ui_plane_depth_str)
                    if ui_plane_depth > 0:
                        self.log(f"使用UI输入的平面深度: {ui_plane_depth:.2f}mm（单应性矩阵转换）")
                        use_planar_transform = True
                        plane_depth = ui_plane_depth
                    else:
                        self.log("警告: UI输入的平面深度值无效（必须大于0），将使用3D转换方法")
                        use_planar_transform = False
                except ValueError:
                    self.log("警告: UI输入的平面深度值格式错误，将使用3D转换方法")
                    use_planar_transform = False
            else:
                # 如果没有输入平面深度，尝试使用从深度图获取的值
                if plane_depth is not None and plane_depth > 0:
                    self.log(f"使用从深度图获取的平面深度: {plane_depth:.2f}mm（单应性矩阵转换）")
                    use_planar_transform = True
                else:
                    self.log("警告: 单应性矩阵转换需要平面深度值，但未输入且无法从深度图获取，将使用3D转换方法")
                    use_planar_transform = False
        
        if use_planar_transform and plane_depth is not None:
            # 使用2D平面转换（单应性矩阵）
            depth_source = "UI输入" if ui_plane_depth_str else "从相机获取"
            self.log(f"使用2D平面转换（单应性矩阵），平面深度: {plane_depth:.2f}mm（{depth_source}）")
            
            # 计算单应性矩阵
            H = self.calibration.compute_homography_from_extrinsic(
                camera1_matrix, camera2_matrix, plane_depth
            )
            if H is None:
                messagebox.showerror("错误", "计算单应性矩阵失败")
                return
            
            # 转换四个点
            for i, point in enumerate(self.selected_points):
                if point is not None:
                    x, y = point
                    # 转换为齐次坐标
                    point_homo = np.array([x, y, 1.0], dtype=np.float32)
                    # 应用单应性矩阵
                    transformed_homo = H @ point_homo
                    if abs(transformed_homo[2]) < 1e-6:
                        self.log(f"警告: {point_labels_cn[i]}转换后齐次坐标第三分量接近0")
                        transformed_points_list.append(None)
                    else:
                        transformed_point = (transformed_homo[0] / transformed_homo[2], 
                                           transformed_homo[1] / transformed_homo[2])
                        transformed_points_list.append(transformed_point)
                        self.log(f"  {point_labels_cn[i]}: ({x}, {y}) -> ({transformed_point[0]:.2f}, {transformed_point[1]:.2f})")
                else:
                    transformed_points_list.append(None)
        else:
            # 使用3D转换（需要深度图）
            if depth_map is None:
                if not ui_plane_depth_str:
                    messagebox.showerror(
                        "错误",
                        "无法进行3D坐标转换：\n\n"
                        "1. 未获取深度图\n"
                        "2. 未输入深度值\n\n"
                        "请在UI中输入深度值（mm）后重试，或连接3D相机获取深度图。"
                    )
                    self.log("错误: 没有深度图且未输入深度值，无法进行3D转换")
                    return
                try:
                    estimated_depth = float(ui_plane_depth_str)
                    if estimated_depth <= 0:
                        raise ValueError()
                except ValueError:
                    messagebox.showerror(
                        "错误",
                        "深度值格式错误或小于等于0，请输入有效的深度值（mm）。"
                    )
                    self.log("错误: UI输入的深度值无效，无法进行3D转换")
                    return

                if self.image_3d is not None:
                    h_img, w_img = self.image_3d.shape[:2]
                    depth_map = np.ones((h_img, w_img), dtype=np.float32) * estimated_depth
                    self.log(f"无深度图，使用UI输入深度: {estimated_depth:.2f}mm 进行3D转换")
                else:
                    messagebox.showerror(
                        "错误",
                        "无法进行坐标转换：\n\n"
                        "1. 3D相机未连接，无法获取深度值\n"
                        "2. 没有加载3D相机图像\n\n"
                        "请连接3D相机并获取深度图，或加载3D相机图像后输入深度值。"
                    )
                    return
            
            self.log("使用3D转换方法（基于深度图）")
            # 转换四个点
            for i, point in enumerate(self.selected_points):
                if point is not None:
                    x, y = point
                    # 获取该点的深度
                    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                        depth = depth_map[y, x]
                        
                        # 如果深度无效，尝试使用周围区域的有效深度值
                        if depth <= 0:
                            depth = self._get_depth_from_neighborhood(depth_map, x, y, search_radius=5)
                            if depth > 0:
                                self.log(f"  {point_labels_cn[i]}: 点({x}, {y})深度无效，使用周围区域平均深度: {depth:.2f}mm")
                        
                        if depth > 0:
                            # 使用3D转换方法
                            transformed_point = self.calibration.transform_point_with_projectpoints(
                                np.array([x, y], dtype=np.float32),
                                camera1_matrix,
                                camera2_matrix,
                                camera1_distortion,
                                camera2_distortion,
                                depth
                            )
                            if transformed_point is not None:
                                transformed_points_list.append((float(transformed_point[0]), float(transformed_point[1])))
                                self.log(f"  {point_labels_cn[i]}: ({x}, {y}), 深度={depth:.2f}mm -> ({transformed_point[0]:.2f}, {transformed_point[1]:.2f})")
                            else:
                                transformed_points_list.append(None)
                                self.log(f"  {point_labels_cn[i]}: 转换失败")
                        else:
                            transformed_points_list.append(None)
                            self.log(f"  {point_labels_cn[i]}: 深度值无效 ({depth:.2f}mm)，周围区域也无有效深度值")
                    else:
                        transformed_points_list.append(None)
                        self.log(f"  {point_labels_cn[i]}: 坐标超出深度图范围")
                else:
                    transformed_points_list.append(None)
        
        # 检查转换结果
        valid_transformed_points = [p for p in transformed_points_list if p is not None]
        if len(valid_transformed_points) < 2:
            error_msg = "坐标转换失败\n\n可能原因：\n1. 深度图无效\n2. 某些点的深度值无效\n3. 转换后的点在相机后方\n4. 有效转换点不足（需要至少2个）\n\n请查看终端输出获取详细错误信息"
            messagebox.showerror("错误", error_msg)
            self.log("坐标转换失败，请查看终端输出获取详细错误信息")
            return
        
        self.transformed_points = transformed_points_list
        self.log(f"成功转换了{len(valid_transformed_points)}个点")
        
        # 在读码器图像上绘制转换后的四个点和连接线
        img_barcode_display = self.image_barcode.copy()
        
        # 根据图像分辨率计算点大小和线宽（与分辨率成比例）
        h_img, w_img = self.image_barcode.shape[:2]
        base_width = 1920
        base_height = 1200
        width_ratio = w_img / base_width
        height_ratio = h_img / base_height
        resolution_ratio = (width_ratio + height_ratio) / 2
        base_line_width = 4  # 与之前ROI的线宽一致
        line_width = max(2, int(base_line_width * resolution_ratio))
        point_radius = max(5, int(8 * resolution_ratio))
        
        # 定义四个点的颜色和标签（使用英文简写）
        point_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # 绿、蓝、红、青
        
        # 绘制转换后的点
        valid_points_dict = {}
        for i, transformed_point in enumerate(transformed_points_list):
            if transformed_point is not None:
                x, y = int(transformed_point[0]), int(transformed_point[1])
                # 确保点在图像范围内
                x = max(0, min(x, w_img - 1))
                y = max(0, min(y, h_img - 1))
                # 绘制点
                cv2.circle(img_barcode_display, (x, y), point_radius, point_colors[i], -1)
                cv2.circle(img_barcode_display, (x, y), point_radius + 2, (255, 255, 255), 2)
                # 绘制标签
                label_pos = (x + point_radius + 5, y - point_radius - 5)
                cv2.putText(img_barcode_display, point_labels[i], label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img_barcode_display, point_labels[i], label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, point_colors[i], 1)
                valid_points_dict[i] = (x, y)
        
        # 绘制连接线（按顺序：左上->右上->右下->左下->左上）
        if len(valid_points_dict) >= 2:
            point_order = [0, 1, 3, 2]  # 左上、右上、右下、左下
            for idx in range(len(point_order)):
                curr_idx = point_order[idx]
                next_idx = point_order[(idx + 1) % len(point_order)]
                if curr_idx in valid_points_dict and next_idx in valid_points_dict:
                    pt1 = valid_points_dict[curr_idx]
                    pt2 = valid_points_dict[next_idx]
                    cv2.line(img_barcode_display, pt1, pt2, (0, 255, 255), line_width)
        
        self.display_image_barcode = img_barcode_display
        self._update_canvas(self.canvas_barcode, img_barcode_display)
        
        # 在3D相机图像上保持点显示
        self.update_display_3d()
        
        # 构建成功消息
        points_info = "\n".join([
            f"{point_labels_cn[i]}: {self.selected_points[i]} -> {transformed_points_list[i] if transformed_points_list[i] else '转换失败'}"
            for i in range(4) if self.selected_points[i] is not None
        ])
        
        messagebox.showinfo("成功", 
            f"坐标转换成功！\n\n"
            f"转换结果:\n{points_info}\n\n"
            f"转换后的点已显示在读码器图像上")


def main():
    """主函数"""
    root = tk.Tk()
    app = CameraCalibrationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
