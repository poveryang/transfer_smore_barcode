# 相机外参标定工具

本项目用于标定迁移科技的3D相机（使用EpicEye SDK）和思谋科技的读码器相机的外参，实现将3D相机的ROI坐标映射到读码器相机的图像上。

## 功能特性

1. **3D相机图像采集**：使用EpicEye SDK采集3D相机的2D图像
2. **读码器相机图像采集**：使用SmoreCamera SDK采集读码器图像
3. **外参标定**：标定两个相机之间的外参（待实现）
4. **坐标转换映射**：将3D相机图像中的ROI坐标转换到读码器相机图像坐标系（待实现）
5. **UI工具**：提供图形界面用于预览图像、输入IP、标定外参和测试坐标转换

## 项目结构

```
transfer_smore_barcode/
├── epiceye_camera.py          # 3D相机采集模块（EpicEye SDK）
├── smore_camera.py          # 读码器相机采集模块（SmoreCamera SDK）
├── calibration.py             # 外参标定和坐标转换模块
├── ui_tool.py                 # UI工具主程序
├── run_ui.py                  # UI工具启动脚本
├── requirements.txt           # Python依赖包
├── README.md                  # 项目说明文档
└── epiceye/                   # EpicEye SDK Python包
    ├── __init__.py
    ├── epiceye.py
    └── epicraw_parser.py
```

## 版本记录

版本变更记录见 `CHANGELOG.md`。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行UI工具

```bash
python ui_tool.py
```

### 简化SDK文件路径（推荐）

只保留 DLL 即可。请在项目根目录创建 `smore_camera_sdk` 目录，并按平台放置：

```
smore_camera_sdk/
├── Win64/
│   ├── SMCamera.dll
│   └── opencv_*.dll
└── Win32/
    ├── SMCamera.dll
    └── opencv_*.dll
```

`smore_camera.py` 会根据平台自动选择 `Win64/Win32` 目录加载。

### UI工具使用说明

1. **连接相机**：
   - 在"相机IP设置"区域输入3D相机和读码器相机的IP地址
   - 点击"连接相机"按钮连接相机
   - 如果3D相机IP为空，将自动搜索网络中的相机

2. **图像预览**：
   - 勾选"开启预览"可以实时预览相机图像
   - 点击"单次采集"可以单次采集图像

3. **绘制ROI**：
   - 在3D相机图像上点击并拖拽鼠标绘制ROI区域
   - 点击"清除ROI"可以清除当前ROI

4. **标定外参**：
   - 点击"标定外参"按钮进行外参标定（功能待实现）
   - 点击"加载标定参数"可以从文件加载已保存的标定参数
   - 点击"保存标定参数"可以将标定参数保存到文件

5. **测试坐标转换**：
   - 在3D相机图像上绘制ROI后，点击"测试坐标转换"按钮
   - 系统将在读码器相机图像上显示对应的ROI区域（功能待实现）

## 模块说明

### smore_camera.py

包含读码器相机类：
- `SmoreCamera`: 思谋科技读码器相机图像采集类（基于SDK DLL）

### epiceye_camera.py

包含3D相机类：
- `EpicEyeCamera`: 迁移科技3D相机图像采集类

### calibration.py

包含外参标定和坐标转换功能：
- `CameraCalibration`: 相机外参标定和坐标转换类
  - `calibrate()`: 标定两个相机之间的外参（待实现）
  - `transform_point()`: 转换单个点坐标（待实现）
  - `transform_roi()`: 转换ROI区域（待实现）

### ui_tool.py

图形界面工具，提供：
- 相机连接和图像预览
- ROI绘制和编辑
- 外参标定操作
- 坐标转换测试

## 待实现功能

1. **外参标定算法**：实现两个相机之间的外参标定算法
2. **坐标转换算法**：实现基于外参的坐标转换算法

## 注意事项

- 确保3D相机和读码器相机在同一网络中
- 外参标定需要两个相机同时看到标定板或特征点
- 坐标转换需要深度信息，确保3D相机能够提供深度数据

## 开发计划

- [x] 3D相机图像采集模块
- [x] 读码器相机图像采集模块（SDK集成）
- [x] 外参标定模块框架（占位）
- [x] 坐标转换模块框架（占位）
- [x] UI工具基础功能
- [ ] 读码器相机SDK集成
- [ ] 外参标定算法实现
- [ ] 坐标转换算法实现
