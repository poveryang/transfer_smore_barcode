"""
相机外参标定和坐标转换模块
使用棋盘格进行标定
"""
import numpy as np
import cv2
from typing import Optional, Tuple, List
import json
import os


class CameraCalibration:
    """相机外参标定和坐标转换类"""
    
    def __init__(self):
        """初始化标定类"""
        self.extrinsic_matrix = None  # 外参矩阵（4x4齐次变换矩阵）
        self.calibration_file = "transfer_smore_calib.json"
        
        # 标定图像对列表（用于多图像标定）
        self.calibration_image_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # 保存标定时得到的内参（用于后续坐标转换）
        self.camera1_matrix = None  # 相机1内参矩阵
        self.camera1_distortion = None  # 相机1畸变参数
        self.camera2_matrix = None  # 相机2内参矩阵
        self.camera2_distortion = None  # 相机2畸变参数
        
    def load_calibration(self, file_path: Optional[str] = None) -> bool:
        """
        从文件加载标定参数
        
        Args:
            file_path: 标定参数文件路径，如果为None则使用默认路径
            
        Returns:
            bool: 加载是否成功
        """
        if file_path is None:
            file_path = self.calibration_file
        
        if not os.path.exists(file_path):
            print(f"标定文件不存在: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.extrinsic_matrix = np.array(data['extrinsic_matrix'])
                
                # 如果文件中保存了内参，也加载它们
                if 'camera1_matrix' in data:
                    self.camera1_matrix = np.array(data['camera1_matrix'])
                if 'camera1_distortion' in data:
                    self.camera1_distortion = np.array(data['camera1_distortion'])
                if 'camera2_matrix' in data:
                    self.camera2_matrix = np.array(data['camera2_matrix'])
                if 'camera2_distortion' in data:
                    self.camera2_distortion = np.array(data['camera2_distortion'])
                
                print(f"标定参数加载成功: {file_path}")
                return True
        except Exception as e:
            print(f"加载标定参数失败: {e}")
            return False
    
    def save_calibration(self, file_path: Optional[str] = None) -> bool:
        """
        保存标定参数到文件
        
        Args:
            file_path: 标定参数文件路径，如果为None则使用默认路径
            
        Returns:
            bool: 保存是否成功
        """
        if self.extrinsic_matrix is None:
            print("没有标定参数可保存")
            return False
        
        if file_path is None:
            file_path = self.calibration_file
        
        try:
            data = {
                'extrinsic_matrix': self.extrinsic_matrix.tolist()
            }
            
            # 如果保存了内参，也一起保存
            if self.camera1_matrix is not None:
                data['camera1_matrix'] = self.camera1_matrix.tolist()
            if self.camera1_distortion is not None:
                data['camera1_distortion'] = self.camera1_distortion.tolist()
            if self.camera2_matrix is not None:
                data['camera2_matrix'] = self.camera2_matrix.tolist()
            if self.camera2_distortion is not None:
                data['camera2_distortion'] = self.camera2_distortion.tolist()
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"标定参数已保存: {file_path}")
            return True
        except Exception as e:
            print(f"保存标定参数失败: {e}")
            return False
    
    def detect_chessboard(self,
                         image: np.ndarray,
                         pattern_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
        """
        检测图像中的棋盘格角点
        
        Args:
            image: 输入图像
            pattern_size: 棋盘格内部角点数量 (cols, rows)，例如 (9, 6) 表示9列6行
            
        Returns:
            Tuple[bool, np.ndarray]: (是否检测成功, 角点坐标)
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                  cv2.CALIB_CB_FAST_CHECK + 
                  cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners
        else:
            return False, None
    
    def calibrate_with_chessboard(self,
                                  image1: np.ndarray,
                                  image2: np.ndarray,
                                  pattern_size: Tuple[int, int],
                                  square_size: float,
                                  camera1_matrix: Optional[np.ndarray] = None,
                                  camera2_matrix: Optional[np.ndarray] = None,
                                  camera1_distortion: Optional[np.ndarray] = None,
                                  camera2_distortion: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        使用棋盘格进行单图像对的外参标定
        
        Args:
            image1: 相机1的图像
            image2: 相机2的图像
            pattern_size: 棋盘格内部角点数量 (cols, rows)
            square_size: 棋盘格方格的实际尺寸（单位：毫米或其他单位）
            camera1_matrix: 相机1的内参矩阵 (3, 3)，可选
            camera2_matrix: 相机2的内参矩阵 (3, 3)，可选
            camera1_distortion: 相机1的畸变参数，可选
            camera2_distortion: 相机2的畸变参数，可选
            
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        # 检测两个图像中的棋盘格
        ret1, corners1 = self.detect_chessboard(image1, pattern_size)
        ret2, corners2 = self.detect_chessboard(image2, pattern_size)
        
        if not ret1:
            return False, "相机1图像中未检测到棋盘格"
        
        if not ret2:
            return False, "相机2图像中未检测到棋盘格"
        
        # 生成3D对象点（棋盘格角点的世界坐标）
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size  # 转换为实际尺寸
        
        # 准备数据
        objpoints = [objp]
        imgpoints1 = [corners1]
        imgpoints2 = [corners2]
        
        # 处理内参
        # 如果相机1没有内参，先进行单目标定
        if camera1_matrix is None:
            print("相机1没有内参，先进行单目标定...")
            # 对相机1进行单目标定
            ret1, camera1_matrix, dist_coeffs1, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints1,
                image1.shape[:2][::-1],  # 图像尺寸
                None, None,
                flags=cv2.CALIB_FIX_PRINCIPAL_POINT  # 固定主点
            )
            
            if not ret1:
                return False, "相机1单目标定失败，无法获取内参"
            
            print(f"相机1单目标定成功，重投影误差: {ret1:.4f}")
        else:
            # 处理内参矩阵格式（可能是列表或1D数组）
            if isinstance(camera1_matrix, list):
                camera1_matrix = np.array(camera1_matrix, dtype=np.float64)
            else:
                camera1_matrix = np.array(camera1_matrix, dtype=np.float64)
            
            # 如果是1D数组（9个元素），reshape为3x3
            if camera1_matrix.ndim == 1:
                if len(camera1_matrix) == 9:
                    camera1_matrix = camera1_matrix.reshape(3, 3)
                else:
                    return False, f"相机1内参矩阵格式错误: 期望9个元素，得到{len(camera1_matrix)}个"
            
            # 确保是3x3矩阵
            if camera1_matrix.shape != (3, 3):
                return False, f"相机1内参矩阵尺寸错误: 期望(3,3)，得到{camera1_matrix.shape}"
            
            dist_coeffs1 = np.array(camera1_distortion, dtype=np.float64) if camera1_distortion is not None else np.zeros(4)
        
        # 如果相机2没有内参，先进行单目标定
        if camera2_matrix is None:
            print("相机2没有内参，先进行单目标定...")
            h2, w2 = image2.shape[:2]
            print(f"相机2图像尺寸: {w2}x{h2}")
            
            # 对相机2进行单目标定
            # 注意：单目标定需要提供初始猜测，否则可能得到异常结果
            # 使用图像尺寸估算初始内参
            fx_init = fy_init = max(w2, h2) * 0.8
            cx_init, cy_init = w2 / 2, h2 / 2
            camera_matrix_init = np.array([
                [fx_init, 0, cx_init],
                [0, fy_init, cy_init],
                [0, 0, 1]
            ], dtype=np.float64)
            
            ret2, camera2_matrix, dist_coeffs2, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints2,
                image2.shape[:2][::-1],  # 图像尺寸 (width, height)
                camera_matrix_init,  # 提供初始猜测
                None,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS  # 使用初始猜测，允许优化
            )
            
            if not ret2:
                return False, "相机2单目标定失败，无法获取内参"
            
            print(f"相机2单目标定成功，重投影误差: {ret2:.4f}")
            print(f"相机2内参矩阵（单目标定结果）:\n{camera2_matrix}")
            
            # 验证单目标定结果的合理性
            fx_result = camera2_matrix[0, 0]
            fy_result = camera2_matrix[1, 1]
            cx_result = camera2_matrix[0, 2]
            cy_result = camera2_matrix[1, 2]
            
            # 检查焦距是否在合理范围内（应该在图像尺寸的0.3-3倍之间）
            if fx_result < w2 * 0.3 or fx_result > w2 * 3 or fy_result < h2 * 0.3 or fy_result > h2 * 3:
                print(f"警告: 相机2单目标定得到的焦距异常，可能不准确")
                print(f"  焦距: fx={fx_result:.2f} (图像宽度={w2}), fy={fy_result:.2f} (图像高度={h2})")
                print(f"  使用估算的内参代替")
                # 使用估算值
                camera2_matrix = camera_matrix_init
                dist_coeffs2 = np.zeros(4)
            elif cx_result < 0 or cx_result > w2 or cy_result < 0 or cy_result > h2:
                print(f"警告: 相机2主点超出图像范围: cx={cx_result:.2f}, cy={cy_result:.2f}, 图像尺寸={w2}x{h2}")
                print(f"  使用估算的主点")
                camera2_matrix[0, 2] = cx_init
                camera2_matrix[1, 2] = cy_init
            if camera2_matrix[0, 0] > w2 * 10 or camera2_matrix[1, 1] > h2 * 10:
                print(f"警告: 相机2单目标定得到的焦距异常大，可能不准确")
                print(f"  焦距: fx={camera2_matrix[0, 0]:.2f}, fy={camera2_matrix[1, 1]:.2f}")
                print(f"  图像尺寸: {w2}x{h2}")
                print(f"  使用估算的内参代替")
                # 使用估算值
                camera2_matrix = camera_matrix_init
                dist_coeffs2 = np.zeros(4)
        else:
            # 处理内参矩阵格式（可能是列表或1D数组）
            if isinstance(camera2_matrix, list):
                camera2_matrix = np.array(camera2_matrix, dtype=np.float64)
            else:
                camera2_matrix = np.array(camera2_matrix, dtype=np.float64)
            
            # 如果是1D数组（9个元素），reshape为3x3
            if camera2_matrix.ndim == 1:
                if len(camera2_matrix) == 9:
                    camera2_matrix = camera2_matrix.reshape(3, 3)
                else:
                    return False, f"相机2内参矩阵格式错误: 期望9个元素，得到{len(camera2_matrix)}个"
            
            # 确保是3x3矩阵
            if camera2_matrix.shape != (3, 3):
                return False, f"相机2内参矩阵尺寸错误: 期望(3,3)，得到{camera2_matrix.shape}"
            
            dist_coeffs2 = np.array(camera2_distortion, dtype=np.float64) if camera2_distortion is not None else np.zeros(4)
        
        # 验证内参矩阵的有效性并打印调试信息
        print(f"相机1内参矩阵:\n{camera1_matrix}")
        print(f"相机1内参矩阵 - fx: {camera1_matrix[0, 0]:.2f}, fy: {camera1_matrix[1, 1]:.2f}, cx: {camera1_matrix[0, 2]:.2f}, cy: {camera1_matrix[1, 2]:.2f}")
        
        if camera1_matrix[2, 2] != 1.0:
            print(f"警告: 相机1内参矩阵[2,2]应为1.0，实际为{camera1_matrix[2, 2]}")
        if camera1_matrix[0, 1] != 0.0:
            print(f"警告: 相机1内参矩阵[0,1]应为0.0，实际为{camera1_matrix[0, 1]}")
        if camera1_matrix[1, 0] != 0.0:
            print(f"警告: 相机1内参矩阵[1,0]应为0.0，实际为{camera1_matrix[1, 0]}")
        if camera1_matrix[2, 0] != 0.0 or camera1_matrix[2, 1] != 0.0:
            print(f"警告: 相机1内参矩阵第三行前两个元素应为0.0")
        
        print(f"相机2内参矩阵:\n{camera2_matrix}")
        print(f"相机2内参矩阵 - fx: {camera2_matrix[0, 0]:.2f}, fy: {camera2_matrix[1, 1]:.2f}, cx: {camera2_matrix[0, 2]:.2f}, cy: {camera2_matrix[1, 2]:.2f}")
        
        if camera2_matrix[2, 2] != 1.0:
            print(f"警告: 相机2内参矩阵[2,2]应为1.0，实际为{camera2_matrix[2, 2]}")
        
        # 检查内参矩阵的合理性（焦距应该为正数，主点应该在图像范围内）
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        if camera1_matrix[0, 0] <= 0 or camera1_matrix[1, 1] <= 0:
            return False, f"相机1内参矩阵焦距无效: fx={camera1_matrix[0, 0]}, fy={camera1_matrix[1, 1]}"
        if camera1_matrix[0, 2] < 0 or camera1_matrix[0, 2] > w1 or camera1_matrix[1, 2] < 0 or camera1_matrix[1, 2] > h1:
            print(f"警告: 相机1主点可能超出图像范围: cx={camera1_matrix[0, 2]}, cy={camera1_matrix[1, 2]}, 图像尺寸={w1}x{h1}")
        
        if camera2_matrix[0, 0] <= 0 or camera2_matrix[1, 1] <= 0:
            return False, f"相机2内参矩阵焦距无效: fx={camera2_matrix[0, 0]}, fy={camera2_matrix[1, 1]}"
        if camera2_matrix[0, 2] < 0 or camera2_matrix[0, 2] > w2 or camera2_matrix[1, 2] < 0 or camera2_matrix[1, 2] > h2:
            print(f"警告: 相机2主点可能超出图像范围: cx={camera2_matrix[0, 2]}, cy={camera2_matrix[1, 2]}, 图像尺寸={w2}x{h2}")
        
        # 检查相机2内参的合理性（焦距不应该过大）
        if camera2_matrix[0, 0] > w2 * 5 or camera2_matrix[1, 1] > h2 * 5:
            print(f"严重警告: 相机2内参焦距异常大！")
            print(f"  fx={camera2_matrix[0, 0]:.2f} (图像宽度={w2})")
            print(f"  fy={camera2_matrix[1, 1]:.2f} (图像高度={h2})")
            print(f"  这可能导致坐标转换结果异常")
            # 不返回错误，但给出警告
        
        # 确定标定标志
        # 如果两个相机都有内参（无论是预先提供的还是单目标定得到的），固定内参只标定外参（精度更高）
        # 如果需要同时优化内参和外参，可以使用 CALIB_USE_INTRINSIC_GUESS
        # 这里我们固定内参，只标定外参，因为单目标定已经得到了内参
        flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参，只标定外参
        
        # 使用立体标定计算外参
        try:
            ret, camera1_matrix_new, dist_coeffs1_new, camera2_matrix_new, dist_coeffs2_new, \
            R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints1, imgpoints2,
                camera1_matrix, dist_coeffs1,
                camera2_matrix, dist_coeffs2,
                image1.shape[:2][::-1],  # 图像尺寸 (width, height)
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            )
            
            if not ret:
                return False, "立体标定失败"
            
            # 构建4x4齐次变换矩阵
            self.extrinsic_matrix = np.eye(4)
            self.extrinsic_matrix[:3, :3] = R
            self.extrinsic_matrix[:3, 3] = T.flatten()
            
            # 保存标定时使用的内参（用于后续坐标转换）
            self.camera1_matrix = camera1_matrix_new
            self.camera1_distortion = dist_coeffs1_new
            self.camera2_matrix = camera2_matrix_new
            self.camera2_distortion = dist_coeffs2_new
            
            msg = f"标定成功！重投影误差: {ret:.4f}"
            if flags != cv2.CALIB_FIX_INTRINSIC:
                msg += "\n注意: 同时标定了相机2的内参，建议使用多组图像提高精度"
            
            return True, msg
            
        except Exception as e:
            return False, f"标定过程出错: {str(e)}"
    
    def calibrate_with_multiple_images(self,
                                      image_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                      pattern_size: Tuple[int, int],
                                      square_size: float,
                                      camera1_matrix: Optional[np.ndarray] = None,
                                      camera2_matrix: Optional[np.ndarray] = None,
                                      camera1_distortion: Optional[np.ndarray] = None,
                                      camera2_distortion: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        使用多组图像进行外参标定（精度更高）
        
        Args:
            image_pairs: 图像对列表，每个元素是(image1, image2)
            pattern_size: 棋盘格内部角点数量 (cols, rows)
            square_size: 棋盘格方格的实际尺寸
            camera1_matrix: 相机1的内参矩阵 (3, 3)，可选
            camera2_matrix: 相机2的内参矩阵 (3, 3)，可选
            camera1_distortion: 相机1的畸变参数，可选
            camera2_distortion: 相机2的畸变参数，可选
            
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        if len(image_pairs) == 0:
            return False, "没有提供图像对"
        
        if camera1_matrix is None or camera2_matrix is None:
            return False, "需要两个相机的内参矩阵"
        
        # 生成3D对象点
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints = []
        imgpoints1 = []
        imgpoints2 = []
        
        success_count = 0
        
        # 检测所有图像对中的棋盘格
        for i, (img1, img2) in enumerate(image_pairs):
            ret1, corners1 = self.detect_chessboard(img1, pattern_size)
            ret2, corners2 = self.detect_chessboard(img2, pattern_size)
            
            if ret1 and ret2:
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                success_count += 1
            else:
                print(f"图像对 {i+1} 中未检测到棋盘格")
        
        if success_count == 0:
            return False, "所有图像对中都未检测到棋盘格"
        
        if success_count < 3:
            return False, f"成功检测的图像对数量不足（{success_count} < 3），建议至少3组图像"
        
        camera1_matrix = np.array(camera1_matrix, dtype=np.float64)
        camera2_matrix = np.array(camera2_matrix, dtype=np.float64)
        dist_coeffs1 = np.array(camera1_distortion, dtype=np.float64) if camera1_distortion is not None else np.zeros(4)
        dist_coeffs2 = np.array(camera2_distortion, dtype=np.float64) if camera2_distortion is not None else np.zeros(4)
        
        # 使用立体标定
        try:
            ret, camera1_matrix_new, dist_coeffs1_new, camera2_matrix_new, dist_coeffs2_new, \
            R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints1, imgpoints2,
                camera1_matrix, dist_coeffs1,
                camera2_matrix, dist_coeffs2,
                image_pairs[0][0].shape[:2][::-1],
                flags=cv2.CALIB_FIX_INTRINSIC,
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            )
            
            if not ret:
                return False, "立体标定失败"
            
            # 构建4x4齐次变换矩阵
            self.extrinsic_matrix = np.eye(4)
            self.extrinsic_matrix[:3, :3] = R
            self.extrinsic_matrix[:3, 3] = T.flatten()
            
            return True, f"标定成功！使用了{success_count}组图像，重投影误差: {ret:.4f}"
            
        except Exception as e:
            return False, f"标定过程出错: {str(e)}"
    
    def transform_point_with_projectpoints(self,
                                          point: np.ndarray,
                                          camera1_matrix: np.ndarray,
                                          camera2_matrix: np.ndarray,
                                          camera1_distortion: Optional[np.ndarray] = None,
                                          camera2_distortion: Optional[np.ndarray] = None,
                                          depth: Optional[float] = None) -> Optional[np.ndarray]:
        """
        使用cv2.projectPoints进行坐标转换（与标定时使用的投影方法一致，包含畸变校正）
        
        Args:
            point: 相机1图像中的点坐标 (x, y)
            camera1_matrix: 相机1的内参矩阵 (3, 3)
            camera2_matrix: 相机2的内参矩阵 (3, 3)
            camera1_distortion: 相机1的畸变参数，可选
            camera2_distortion: 相机2的畸变参数，可选
            depth: 点的深度值（单位：毫米）
            
        Returns:
            np.ndarray: 相机2图像中的点坐标 (x, y)，失败返回None
        """
        if self.extrinsic_matrix is None:
            print("外参矩阵未标定")
            return None
        
        if depth is None or depth <= 0:
            print("错误: 需要深度信息才能进行坐标转换")
            return None
        
        try:
            # 1. 将相机1图像坐标转换为相机1的3D坐标（考虑畸变）
            fx1, fy1 = camera1_matrix[0, 0], camera1_matrix[1, 1]
            cx1, cy1 = camera1_matrix[0, 2], camera1_matrix[1, 2]
            
            # 去畸变（如果需要）
            point_2d = np.array([[point[0], point[1]]], dtype=np.float32)
            
            # 确保畸变参数是numpy数组格式
            if camera1_distortion is not None:
                if isinstance(camera1_distortion, list):
                    camera1_distortion = np.array(camera1_distortion, dtype=np.float32)
                elif not isinstance(camera1_distortion, np.ndarray):
                    camera1_distortion = np.array(camera1_distortion, dtype=np.float32)
                else:
                    camera1_distortion = camera1_distortion.astype(np.float32)
            
            if camera1_distortion is not None and len(camera1_distortion) > 0 and np.any(camera1_distortion != 0):
                # 使用undistortPoints去畸变，返回归一化坐标（在相机坐标系中，z=1时的x,y坐标）
                point_undistorted = cv2.undistortPoints(
                    point_2d, camera1_matrix, camera1_distortion, P=None
                )
                # undistortPoints返回的是归一化坐标
                u_norm = point_undistorted[0, 0, 0]
                v_norm = point_undistorted[0, 0, 1]
            else:
                # 没有畸变，直接计算归一化坐标
                u_norm = (point[0] - cx1) / fx1
                v_norm = (point[1] - cy1) / fy1
            
            # 3D坐标（在相机1坐标系中）
            x = u_norm * depth
            y = v_norm * depth
            z = depth
            
            point_3d_cam1 = np.array([[x, y, z]], dtype=np.float32)
            
            # 2. 使用外参矩阵将3D坐标从相机1坐标系转换到相机2坐标系
            point_3d_homo = np.append(point_3d_cam1[0], 1.0)
            point_3d_cam2_homo = self.extrinsic_matrix @ point_3d_homo
            point_3d_cam2 = point_3d_cam2_homo[:3].reshape(1, 1, 3)
            
            if point_3d_cam2[0, 0, 2] <= 0:
                print("警告: 点在相机2后方，无法投影")
                return None
            
            # 3. 使用cv2.projectPoints将相机2的3D坐标投影到图像坐标系（考虑畸变）
            # 这是与标定时使用的相同方法！
            rvec = np.zeros((3, 1), dtype=np.float32)  # 旋转向量（单位矩阵，因为已经在相机2坐标系中）
            tvec = np.zeros((3, 1), dtype=np.float32)  # 平移向量（零向量，因为已经在相机2坐标系中）
            
            # 确保畸变参数是numpy数组格式
            if camera2_distortion is not None:
                if isinstance(camera2_distortion, list):
                    dist_coeffs = np.array(camera2_distortion, dtype=np.float32)
                elif not isinstance(camera2_distortion, np.ndarray):
                    dist_coeffs = np.array(camera2_distortion, dtype=np.float32)
                else:
                    dist_coeffs = camera2_distortion.astype(np.float32)
            else:
                dist_coeffs = np.zeros(4, dtype=np.float32)
            
            # 确保dist_coeffs是1D数组（OpenCV要求）
            if dist_coeffs.ndim > 1:
                dist_coeffs = dist_coeffs.flatten()
            
            projected_points, _ = cv2.projectPoints(
                point_3d_cam2,
                rvec, tvec,
                camera2_matrix,
                dist_coeffs
            )
            
            return projected_points[0, 0]
        except Exception as e:
            print(f"坐标转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def transform_point(self, 
                       point: np.ndarray,
                       camera1_matrix: np.ndarray,
                       camera2_matrix: np.ndarray,
                       camera1_distortion: Optional[np.ndarray] = None,
                       camera2_distortion: Optional[np.ndarray] = None,
                       depth: Optional[float] = None) -> Optional[np.ndarray]:
        """
        将相机1图像中的点坐标转换到相机2图像坐标系
        
        Args:
            point: 相机1图像中的点坐标 (x, y)
            camera1_matrix: 相机1的内参矩阵 (3, 3)
            camera2_matrix: 相机2的内参矩阵 (3, 3)
            camera1_distortion: 相机1的畸变参数，可选
            camera2_distortion: 相机2的畸变参数，可选
            depth: 点的深度值，如果为None则无法转换
            
        Returns:
            np.ndarray: 相机2图像中的点坐标 (x, y)，失败返回None
        """
        if self.extrinsic_matrix is None:
            print("外参矩阵未标定")
            return None
        
        if depth is None or depth <= 0:
            print("错误: 需要深度信息才能进行坐标转换")
            return None
        
        try:
            # 1. 将相机1图像坐标转换为相机1的3D坐标
            fx1, fy1 = camera1_matrix[0, 0], camera1_matrix[1, 1]
            cx1, cy1 = camera1_matrix[0, 2], camera1_matrix[1, 2]
            
            x = (point[0] - cx1) * depth / fx1
            y = (point[1] - cy1) * depth / fy1
            z = depth
            
            point_3d_cam1 = np.array([x, y, z])
            
            # 2. 使用外参矩阵将3D坐标从相机1坐标系转换到相机2坐标系
            point_3d_homo = np.append(point_3d_cam1, 1.0)
            point_3d_cam2_homo = self.extrinsic_matrix @ point_3d_homo
            point_3d_cam2 = point_3d_cam2_homo[:3]
            
            # 3. 将相机2的3D坐标投影到相机2图像坐标系
            fx2, fy2 = camera2_matrix[0, 0], camera2_matrix[1, 1]
            cx2, cy2 = camera2_matrix[0, 2], camera2_matrix[1, 2]
            
            if point_3d_cam2[2] <= 0:
                print("警告: 点在相机2后方，无法投影")
                return None
            
            u = fx2 * point_3d_cam2[0] / point_3d_cam2[2] + cx2
            v = fy2 * point_3d_cam2[1] / point_3d_cam2[2] + cy2
            
            return np.array([u, v])
        except Exception as e:
            print(f"坐标转换失败: {e}")
            return None
    
    def transform_roi(self,
                     roi: Tuple[int, int, int, int],
                     camera1_matrix: np.ndarray,
                     camera2_matrix: np.ndarray,
                     camera1_distortion: Optional[np.ndarray] = None,
                     camera2_distortion: Optional[np.ndarray] = None,
                     depth_map: Optional[np.ndarray] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        将相机1图像中的ROI区域转换到相机2图像坐标系
        
        Args:
            roi: ROI区域 (x, y, width, height)
            camera1_matrix: 相机1的内参矩阵 (3, 3)
            camera2_matrix: 相机2的内参矩阵 (3, 3)
            camera1_distortion: 相机1的畸变参数，可选
            camera2_distortion: 相机2的畸变参数，可选
            depth_map: 相机1的深度图，用于坐标转换
            
        Returns:
            Tuple[int, int, int, int]: 相机2图像中的ROI区域 (x, y, width, height)，失败返回None
        """
        try:
            if self.extrinsic_matrix is None:
                print("外参矩阵未标定")
                return None
            
            if depth_map is None:
                print("错误: 需要深度图才能转换ROI")
                return None
            
            x, y, w, h = roi
            
            # 计算ROI的四个角点
            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
            
            # 将四个角点分别转换到相机2坐标系
            # 使用cv2.projectPoints方法（与标定时使用的投影方法一致）
            transformed_corners = []
            print(f"开始转换ROI的4个角点（使用cv2.projectPoints方法，与标定时一致）:")
            for i, corner in enumerate(corners):
                u, v = int(corner[0]), int(corner[1])
                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    depth = depth_map[v, u]
                    if depth > 0:
                        # 调试信息：打印每个角点的转换过程
                        print(f"  角点{i+1}: 图像坐标({u}, {v}), 深度={depth:.2f}mm")
                        # 使用与标定时相同的投影方法
                        transformed_point = self.transform_point_with_projectpoints(
                            corner, camera1_matrix, camera2_matrix,
                            camera1_distortion, camera2_distortion, depth
                        )
                        if transformed_point is not None:
                            print(f"    转换后坐标: ({transformed_point[0]:.2f}, {transformed_point[1]:.2f})")
                            transformed_corners.append(transformed_point)
                        else:
                            print(f"    转换失败")
                    else:
                        print(f"  角点{i+1}: 深度值无效 ({depth:.2f}mm)")
                else:
                    print(f"  角点{i+1}: 坐标超出深度图范围 ({u}, {v})")
            
            if len(transformed_corners) < 2:
                print(f"错误: 只有{len(transformed_corners)}个有效转换点（需要至少2个），无法计算ROI")
                print(f"  可能原因：")
                print(f"  1. 某些角点的深度值无效")
                print(f"  2. 转换后的点在相机2后方")
                print(f"  3. 坐标转换过程中出现异常")
                return None
            
            transformed_corners = np.array(transformed_corners)
            
            # 计算新的ROI边界框
            min_x = int(np.min(transformed_corners[:, 0]))
            min_y = int(np.min(transformed_corners[:, 1]))
            max_x = int(np.max(transformed_corners[:, 0]))
            max_y = int(np.max(transformed_corners[:, 1]))
            
            new_x = max(0, min_x)
            new_y = max(0, min_y)
            new_w = max_x - min_x
            new_h = max_y - min_y
            
            print(f"转换成功: ROI边界框 x={new_x}, y={new_y}, w={new_w}, h={new_h}")
            return (new_x, new_y, new_w, new_h)
        except Exception as e:
            print(f"transform_roi异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_homography_from_extrinsic(self,
                                          camera1_matrix: np.ndarray,
                                          camera2_matrix: np.ndarray,
                                          plane_depth: float = 1000.0,
                                          plane_normal: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        从外参矩阵和内参矩阵计算单应性矩阵（用于平面场景的2D到2D转换）
        
        Args:
            camera1_matrix: 相机1的内参矩阵 (3, 3)
            camera2_matrix: 相机2的内参矩阵 (3, 3)
            plane_depth: 平面的深度值（单位：毫米），默认1000mm
            plane_normal: 平面的法向量（在相机1坐标系中），如果为None则假设平面垂直于相机光轴
        
        Returns:
            np.ndarray: 单应性矩阵 (3, 3)，失败返回None
        """
        if self.extrinsic_matrix is None:
            print("外参矩阵未标定")
            return None
        
        try:
            # 如果未指定平面法向量，假设平面垂直于相机1的光轴（z轴）
            if plane_normal is None:
                plane_normal = np.array([0, 0, 1], dtype=np.float64)  # z轴方向
            
            # 归一化法向量
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            
            # 计算平面方程：n^T * X = d，其中d是平面到原点的距离
            # 假设平面在深度plane_depth处
            d = plane_depth
            
            # 提取外参矩阵的旋转和平移
            R = self.extrinsic_matrix[:3, :3]
            t = self.extrinsic_matrix[:3, 3]
            
            # 单应性矩阵计算公式：H = K2 * (R - t * n^T / d) * K1^(-1)
            # 其中：
            # - K1, K2 是内参矩阵（单位：像素）
            # - R 是旋转矩阵（无量纲）
            # - t 是平移向量（单位：与标定时使用的单位一致，通常是毫米）
            # - n 是平面法向量（在相机1坐标系中）
            # - d 是平面到相机1的距离（单位：与t一致，通常是毫米）
            # 
            # 注意：t/d 是无量纲的，所以单位是一致的
            
            K1_inv = np.linalg.inv(camera1_matrix)
            n_T = plane_normal.reshape(1, 3)
            
            # 计算 t * n^T / d
            # 这里 t 是列向量 (3,1)，n^T 是行向量 (1,3)，结果是 (3,3) 矩阵
            t_nT = np.outer(t, plane_normal) / d
            
            # 计算 R - t * n^T / d
            R_modified = R - t_nT
            
            # 检查计算结果的合理性
            if np.any(np.isnan(R_modified)) or np.any(np.isinf(R_modified)):
                print(f"错误: R_modified包含NaN或Inf值")
                print(f"  R:\n{R}")
                print(f"  t:\n{t}")
                print(f"  plane_normal:\n{plane_normal}")
                print(f"  d: {d}")
                return None
            
            # 计算单应性矩阵
            H = camera2_matrix @ R_modified @ K1_inv
            
            # 归一化单应性矩阵（将第三行归一化，使H[2,2]=1，提高数值稳定性）
            # 单应性矩阵是齐次的，可以任意缩放，通常归一化使H[2,2]=1
            if abs(H[2, 2]) > 1e-10:
                H = H / H[2, 2]
            else:
                print(f"警告: 单应性矩阵H[2,2]接近0 ({H[2, 2]:.2e})，可能导致数值不稳定")
                return None
            
            # 调试信息（减少输出，只在需要时打印）
            print(f"单应性矩阵计算:")
            print(f"  平面深度: {plane_depth}mm")
            print(f"  平面法向量: {plane_normal}")
            print(f"  旋转矩阵R:\n{R}")
            print(f"  平移向量t: {t} (单位应与标定时一致，通常是毫米)")
            print(f"  t的模长: {np.linalg.norm(t):.2f}mm")
            print(f"  t/d比例: {np.linalg.norm(t)/d:.6f} (应该 < 1，否则可能有问题)")
            if np.linalg.norm(t)/d > 1:
                print(f"  警告: t/d比例 > 1，这可能表示平面深度值不准确或外参矩阵有问题")
            print(f"  t * n^T / d:\n{t_nT}")
            print(f"  R_modified:\n{R_modified}")
            print(f"  相机1内参K1:\n{camera1_matrix}")
            print(f"  相机2内参K2:\n{camera2_matrix}")
            print(f"  单应性矩阵H:\n{H}")
            
            # 检查单应性矩阵的条件数（如果太大，说明矩阵接近奇异）
            cond_num = np.linalg.cond(H)
            if cond_num > 1e10:
                print(f"警告: 单应性矩阵条件数很大 ({cond_num:.2e})，可能导致数值不稳定")
            
            return H
        except Exception as e:
            print(f"计算单应性矩阵失败: {e}")
            return None
    
    def transform_roi_planar(self,
                            roi: Tuple[int, int, int, int],
                            camera1_matrix: np.ndarray,
                            camera2_matrix: np.ndarray,
                            plane_depth: float = 1000.0,
                            camera1_distortion: Optional[np.ndarray] = None,
                            camera2_distortion: Optional[np.ndarray] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        使用单应性矩阵将相机1图像中的ROI区域转换到相机2图像坐标系（平面场景，无需深度图）
        
        Args:
            roi: ROI区域 (x, y, width, height)
            camera1_matrix: 相机1的内参矩阵 (3, 3)
            camera2_matrix: 相机2的内参矩阵 (3, 3)
            plane_depth: 平面的深度值（单位：毫米），默认1000mm
            camera1_distortion: 相机1的畸变参数，可选
            camera2_distortion: 相机2的畸变参数，可选
            
        Returns:
            Tuple[int, int, int, int]: 相机2图像中的ROI区域 (x, y, width, height)，失败返回None
        """
        if self.extrinsic_matrix is None:
            print("外参矩阵未标定")
            return None
        
        try:
            # 注意：单应性矩阵应该在去畸变的空间中计算
            # 如果相机有畸变，我们需要使用去畸变后的内参矩阵
            # 但为了简化，我们假设内参矩阵已经考虑了畸变的影响
            # 实际上，单应性矩阵公式中的K1和K2应该是在去畸变空间中的内参
            
            # 计算单应性矩阵
            # 注意：这里的camera1_matrix和camera2_matrix应该是在去畸变空间中的内参
            # 如果相机有畸变，理论上应该使用去畸变后的内参，但通常使用原始内参也可以
            H = self.compute_homography_from_extrinsic(camera1_matrix, camera2_matrix, plane_depth)
            if H is None:
                return None
            
            print(f"单应性矩阵计算完成，平面深度: {plane_depth:.2f}mm")
            
            x, y, w, h = roi
            
            # 计算ROI的四个角点（图像坐标）
            corners_distorted = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
            
            # 对输入点进行去畸变（如果需要）
            if camera1_distortion is not None and np.any(camera1_distortion != 0):
                # 确保畸变参数格式正确
                if isinstance(camera1_distortion, list):
                    dist_coeffs1 = np.array(camera1_distortion, dtype=np.float32)
                else:
                    dist_coeffs1 = np.array(camera1_distortion, dtype=np.float32)
                if dist_coeffs1.ndim > 1:
                    dist_coeffs1 = dist_coeffs1.flatten()
                
                # 去畸变：将图像坐标转换为去畸变后的图像坐标
                # 使用P=camera1_matrix，这样返回的是去畸变后的图像坐标（而不是归一化坐标）
                corners_undistorted = cv2.undistortPoints(
                    corners_distorted.reshape(-1, 1, 2),
                    camera1_matrix,
                    dist_coeffs1,
                    P=camera1_matrix  # 使用原内参矩阵，得到去畸变后的图像坐标
                )
                corners_undistorted = corners_undistorted.reshape(-1, 2)
                
                # 验证去畸变后的坐标是否合理
                if np.any(np.isnan(corners_undistorted)) or np.any(np.isinf(corners_undistorted)):
                    print("错误: 去畸变后的坐标包含NaN或Inf值")
                    return None
                print(f"原始ROI角点（畸变）:\n{corners_distorted}")
                print(f"去畸变后的ROI角点:\n{corners_undistorted}")
            else:
                corners_undistorted = corners_distorted
                print(f"原始ROI角点（无畸变）:\n{corners_undistorted}")
            
            # 转换为齐次坐标
            corners_homo = np.ones((corners_undistorted.shape[0], 3), dtype=np.float32)
            corners_homo[:, :2] = corners_undistorted
            corners_homo = corners_homo.T  # (3, 4)
            
            # 使用单应性矩阵转换角点
            print(f"使用单应性矩阵转换...")
            transformed_corners_homo = H @ corners_homo
            print(f"转换后齐次坐标:\n{transformed_corners_homo}")
            
            # 检查是否有异常大的值
            if np.any(np.abs(transformed_corners_homo[2, :]) < 1e-6):
                print("警告: 转换后的齐次坐标第三分量接近0，可能导致数值不稳定")
                return None
            
            transformed_corners_undistorted = transformed_corners_homo[:2, :] / transformed_corners_homo[2, :]
            transformed_corners_undistorted = transformed_corners_undistorted.T  # (4, 2)
            
            print(f"转换后角点坐标（去畸变后）:\n{transformed_corners_undistorted}")
            
            # 对输出点应用畸变（如果需要，恢复到相机2的实际图像坐标）
            if camera2_distortion is not None and np.any(camera2_distortion != 0):
                # 确保畸变参数格式正确
                if isinstance(camera2_distortion, list):
                    dist_coeffs2 = np.array(camera2_distortion, dtype=np.float32)
                else:
                    dist_coeffs2 = np.array(camera2_distortion, dtype=np.float32)
                if dist_coeffs2.ndim > 1:
                    dist_coeffs2 = dist_coeffs2.flatten()
                
                # 将去畸变后的坐标转换为归一化坐标
                # 然后应用畸变，再投影回图像坐标
                # 注意：这里我们需要反向操作，但实际上单应性矩阵已经考虑了内参
                # 所以转换后的坐标应该已经是图像坐标了
                # 如果相机2有畸变，我们需要应用畸变
                # 但通常单应性矩阵是在去畸变空间中工作的，所以这里可能需要特殊处理
                # 为了简化，我们先假设输出也是去畸变的坐标
                transformed_corners = transformed_corners_undistorted
                print(f"注意: 相机2有畸变参数，但单应性矩阵在去畸变空间中计算")
                print(f"转换后的坐标可能需要进一步处理以匹配实际图像")
            else:
                transformed_corners = transformed_corners_undistorted
            
            print(f"最终转换后角点坐标:\n{transformed_corners}")
            
            # 检查转换后的坐标是否合理
            # 使用主点坐标估算图像尺寸（主点通常在图像中心附近）
            estimated_w = int(camera2_matrix[0, 2] * 2)  # 主点x坐标的2倍作为估算宽度
            estimated_h = int(camera2_matrix[1, 2] * 2)  # 主点y坐标的2倍作为估算高度
            
            # 检查坐标是否在合理范围内（允许一定的超出，因为估算可能不准确）
            max_x = np.max(transformed_corners[:, 0])
            min_x = np.min(transformed_corners[:, 0])
            max_y = np.max(transformed_corners[:, 1])
            min_y = np.min(transformed_corners[:, 1])
            
            print(f"转换后坐标范围: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
            print(f"估算图像尺寸: {estimated_w}x{estimated_h}")
            
            # 如果坐标明显超出图像范围，给出警告但不返回错误（因为可能是估算不准确）
            if min_x < -estimated_w * 0.5 or max_x > estimated_w * 1.5 or min_y < -estimated_h * 0.5 or max_y > estimated_h * 1.5:
                print(f"警告: 转换后的坐标可能超出图像范围")
                print(f"  这可能是因为：")
                print(f"  1. 平面深度值不准确（当前: {plane_depth:.2f}mm）")
                print(f"  2. ROI不在假设的平面上")
                print(f"  3. 外参矩阵精度不足")
                # 不返回错误，继续处理，但结果可能不准确
            
            # 计算新的ROI边界框
            min_x = int(np.min(transformed_corners[:, 0]))
            min_y = int(np.min(transformed_corners[:, 1]))
            max_x = int(np.max(transformed_corners[:, 0]))
            max_y = int(np.max(transformed_corners[:, 1]))
            
            new_x = max(0, min_x)
            new_y = max(0, min_y)
            new_w = max(1, max_x - min_x)
            new_h = max(1, max_y - min_y)
            
            print(f"最终ROI边界框: x={new_x}, y={new_y}, w={new_w}, h={new_h}")
            return (new_x, new_y, new_w, new_h)
        except Exception as e:
            print(f"平面转换失败: {e}")
            return None
    
    def is_calibrated(self) -> bool:
        """检查是否已标定"""
        return self.extrinsic_matrix is not None
