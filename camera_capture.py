"""
相机图像采集模块
- 3D相机（迁移科技EpicEye）图像采集
- 读码器相机（思谋科技）图像采集
"""
import time
import socket
import struct
import cv2
import numpy as np
from typing import Optional, Tuple
import epiceye


class EpicEyeCamera:
    """迁移科技3D相机（EpicEye）图像采集类"""
    
    def __init__(self, ip: Optional[str] = None):
        """
        初始化3D相机
        
        Args:
            ip: 相机IP地址（可带端口，如 "169.254.188.22:5000"），如果为None则自动搜索
        """
        self.ip = ip
        self.info = None
        self.camera_matrix = None
        self.distortion = None
    
    def _normalize_ip(self, ip: str) -> str:
        """
        规范化IP地址，如果未指定端口则添加默认端口5000
        
        Args:
            ip: IP地址字符串
            
        Returns:
            str: 规范化后的IP地址（带端口）
        """
        if not ip:
            return ip
        
        # 如果IP中已经包含端口号（包含冒号），直接返回
        if ':' in ip:
            return ip
        
        # 如果没有端口号，添加默认端口5000
        return f"{ip}:5000"
        
    def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        连接相机（带重试机制）
        
        Args:
            max_retries: 最大重试次数，默认3次
            retry_delay: 重试间隔（秒），默认2秒
        
        Returns:
            bool: 连接是否成功
        """
        if self.ip is None:
            # 自动搜索相机
            found_cameras = epiceye.search_camera()
            if found_cameras is None or len(found_cameras) == 0:
                print("未找到相机，请检查网络连接")
                return False
            self.ip = found_cameras[0]["ip"]
            print(f"自动搜索到相机，IP: {self.ip}")
        else:
            # 规范化IP地址（添加默认端口）
            original_ip = self.ip
            self.ip = self._normalize_ip(self.ip)
            if original_ip != self.ip:
                print(f"IP地址已规范化: {original_ip} -> {self.ip}")
        
        # 尝试连接（带重试机制）
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"连接失败，等待 {retry_delay} 秒后重试 ({attempt}/{max_retries-1})...")
                print("提示: 如果浏览器中打开了相机的可视化工具，请先关闭浏览器标签页")
                time.sleep(retry_delay)
            
            # 获取相机信息
            self.info = epiceye.get_info(self.ip)
            if self.info is not None:
                # 连接成功，获取相机内参
                self.camera_matrix = epiceye.get_camera_matrix(self.ip)
                self.distortion = epiceye.get_distortion(self.ip)
                
                print(f"相机连接成功: {self.info.get('model', 'Unknown')}, "
                      f"分辨率: {self.info.get('width')}x{self.info.get('height')}")
                return True
        
        # 所有重试都失败
        print(f"无法获取相机信息，IP={self.ip}")
        print("可能的原因:")
        print("  1. 相机未开机或网络连接异常")
        print("  2. 浏览器中打开了相机的可视化工具，占用了连接（请关闭浏览器标签页后重试）")
        print("  3. 其他程序正在使用相机")
        return False
    
    def _check_connection(self) -> bool:
        """
        检查相机连接状态
        
        Returns:
            bool: 连接是否正常
        """
        if self.ip is None:
            return False
        # 快速检查连接状态
        info = epiceye.get_info(self.ip)
        return info is not None
    
    def capture_image(self, auto_reconnect: bool = True) -> Optional[np.ndarray]:
        """
        采集2D图像（参考示例代码流程）
        
        Args:
            auto_reconnect: 如果连接失败，是否自动重连，默认True
        
        Returns:
            np.ndarray: 图像数据（BGR格式），失败返回None
        """
        if self.ip is None:
            print("相机未连接")
            return None
        
        # 在采集前检查连接状态
        if not self._check_connection():
            print("警告: 相机连接状态异常")
            if auto_reconnect:
                print("尝试重新连接相机...")
                if not self.connect(max_retries=2, retry_delay=1.0):
                    print("重新连接失败，请检查网络连接和相机状态")
                    return None
                print("重新连接成功")
            else:
                print("请先连接相机")
                return None
        
        # 参考示例代码：即使只获取图像，也使用pointcloud=True来触发拍摄
        # 这样可以确保相机正确触发，即使后续只获取图像数据
        # 触发拍摄（带重试机制）
        frame_id = None
        max_trigger_retries = 3
        trigger_retry_delay = 1.0
        
        for attempt in range(max_trigger_retries):
            frame_id = epiceye.trigger_frame(ip=self.ip, pointcloud=True)
            if frame_id is not None:
                break
            if attempt < max_trigger_retries - 1:
                print(f"触发拍摄失败，等待 {trigger_retry_delay} 秒后重试 ({attempt+1}/{max_trigger_retries-1})...")
                print("提示: 如果浏览器中打开了相机的可视化工具，请先关闭浏览器标签页")
                time.sleep(trigger_retry_delay)
        
        if frame_id is None:
            print("触发拍摄失败，可能的原因:")
            print("  1. 相机连接已断开，请重新连接")
            print("  2. 浏览器中打开了相机的可视化工具，占用了连接（请关闭浏览器标签页后重试）")
            print("  3. 其他程序正在使用相机")
            print("  4. 网络连接不稳定")
            return None
        
        # 获取图像（重试机制，增加重试次数和延迟）
        image = None
        max_retries = 10
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            image = epiceye.get_image(ip=self.ip, frame_id=frame_id)
            if image is not None:
                break
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        if image is None:
            print("获取图像失败")
            return None
        
        # 转换10bit图像到8bit（如果需要）
        if image.dtype == np.uint16:
            image = cv2.convertScaleAbs(image, alpha=(255.0 / 1024.0))
        
        # 如果是灰度图，转换为BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image
    
    def capture_depth(self, frame_id: Optional[str] = None, auto_reconnect: bool = True) -> Optional[np.ndarray]:
        """
        获取深度图
        
        Args:
            frame_id: 帧ID，如果为None则触发新的拍摄
            auto_reconnect: 如果连接失败，是否自动重连，默认True
        
        Returns:
            np.ndarray: 深度数据，失败返回None
        """
        if self.ip is None:
            print("相机未连接")
            return None
        
        # 在采集前检查连接状态
        if not self._check_connection():
            print("警告: 相机连接状态异常")
            if auto_reconnect:
                print("尝试重新连接相机...")
                if not self.connect(max_retries=2, retry_delay=1.0):
                    print("重新连接失败，请检查网络连接和相机状态")
                    return None
                print("重新连接成功")
            else:
                print("请先连接相机")
                return None
        
        # 如果没有提供frame_id，先触发拍摄（需要点云数据以获取深度）
        if frame_id is None:
            frame_id = epiceye.trigger_frame(ip=self.ip, pointcloud=True)
            if frame_id is None:
                print("触发拍摄失败")
                return None
        
        # 获取深度图（重试机制）
        depth = None
        for _ in range(5):
            depth = epiceye.get_depth(ip=self.ip, frame_id=frame_id)
            if depth is not None:
                break
            time.sleep(0.05)
        
        if depth is None:
            print("获取深度图失败")
            return None
        
        return depth
    
    def get_camera_info(self) -> Optional[dict]:
        """获取相机信息"""
        return self.info
    
    def get_intrinsics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取相机内参
        
        Returns:
            Tuple[camera_matrix, distortion]: 相机矩阵和畸变参数
        """
        return self.camera_matrix, self.distortion


class BarcodeReaderCamera:
    """思谋科技读码器相机图像采集类"""
    
    def __init__(self, ip: Optional[str] = None, port: Optional[int] = None):
        """
        初始化读码器相机
        
        Args:
            ip: 相机IP地址（可带端口，如 "192.168.1.100:8080"），如果为None则使用默认值
            port: 相机端口号，如果为None且ip中未指定端口，则使用默认端口8080
        """
        self.server_ip, self.server_port = self._parse_ip_port(ip, port)
        self.sock: Optional[socket.socket] = None
        self.connected = False
        self.info = None
        
    def _parse_ip_port(self, ip: Optional[str], port: Optional[int]) -> Tuple[str, int]:
        """
        解析IP地址和端口号
        
        Args:
            ip: IP地址字符串（可能包含端口）
            port: 端口号
            
        Returns:
            Tuple[str, int]: (IP地址, 端口号)
        """
        default_port = 8080
        
        if ip is None:
            return "localhost", default_port
        
        # 如果IP中包含端口号（包含冒号）
        if ':' in ip:
            parts = ip.rsplit(':', 1)  # 从右边分割，只分割一次
            parsed_ip = parts[0]
            try:
                parsed_port = int(parts[1])
                return parsed_ip, parsed_port
            except ValueError:
                print(f"警告: IP地址中的端口号无效，使用默认端口{default_port}")
                return parsed_ip, default_port
        
        # 如果提供了独立的port参数，使用它
        if port is not None:
            return ip, port
        
        # 否则使用默认端口
        return ip, default_port
    
    def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        连接相机（带重试机制）
        
        Args:
            max_retries: 最大重试次数，默认3次
            retry_delay: 重试间隔（秒），默认2秒
        
        Returns:
            bool: 连接是否成功
        """
        # 如果已经连接，先关闭旧连接
        if self.connected and self.sock is not None:
            self.close()
        
        # 尝试连接（带重试机制）
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"连接失败，等待 {retry_delay} 秒后重试 ({attempt}/{max_retries-1})...")
                time.sleep(retry_delay)
            
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)  # 设置超时时间
                self.sock.connect((self.server_ip, self.server_port))
                self.connected = True
                self.info = {
                    'ip': self.server_ip,
                    'port': self.server_port,
                    'type': 'BarcodeReader'
                }
                print(f"读码器相机连接成功: {self.server_ip}:{self.server_port}")
                return True
            except socket.timeout:
                print(f"连接超时: {self.server_ip}:{self.server_port}")
                self.connected = False
            except socket.error as e:
                print(f"连接错误: {e}")
                self.connected = False
            except Exception as e:
                print(f"连接异常: {e}")
                self.connected = False
            
            # 清理失败的socket
            if self.sock is not None:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
        
        # 所有重试都失败
        print(f"无法连接到读码器相机服务器: {self.server_ip}:{self.server_port}")
        print("可能的原因:")
        print("  1. 相机服务器未启动或网络连接异常")
        print("  2. IP地址或端口号配置错误")
        print("  3. 防火墙阻止了连接")
        return False
    
    def _check_connection(self) -> bool:
        """
        检查相机连接状态
        
        Returns:
            bool: 连接是否正常
        """
        return self.connected and self.sock is not None
    
    def send_command(self, cmd: str) -> bool:
        """
        发送命令到相机服务器
        
        Args:
            cmd: 命令字符串（如 "GET_FRAME", "SET_EXP_PARAMS 3000 128"）
        
        Returns:
            bool: 发送是否成功
        """
        if not self._check_connection():
            print("未连接到相机服务器")
            return False
        
        try:
            data = cmd.encode('utf-8')
            length = len(data)
            len_pack = struct.pack('!I', length)  # 大端序，4字节
            self.sock.sendall(len_pack)
            self.sock.sendall(data)
            return True
        except socket.error as e:
            print(f"发送命令错误: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"发送命令异常: {e}")
            self.connected = False
            return False
    
    def _recv_all(self, size: int) -> Optional[bytes]:
        """
        接收指定长度的数据（确保接收完整）
        
        Args:
            size: 要接收的字节数
        
        Returns:
            bytes: 接收到的数据，失败返回None
        """
        if not self._check_connection():
            return None
        
        buf = b''
        try:
            while len(buf) < size:
                data = self.sock.recv(size - len(buf))
                if not data:
                    return None
                buf += data
            return buf
        except socket.error as e:
            print(f"接收数据错误: {e}")
            self.connected = False
            return None
        except Exception as e:
            print(f"接收数据异常: {e}")
            self.connected = False
            return None
    
    def receive_image(self) -> Optional[np.ndarray]:
        """
        接收图像数据
        
        Returns:
            np.ndarray: 图像数据（BGR格式），失败返回None
        """
        if not self._check_connection():
            print("未连接到相机服务器")
            return None
        
        try:
            # 先接收4字节的图像大小
            size_data = self._recv_all(4)
            if size_data is None or len(size_data) != 4:
                print("接收图像大小失败")
                return None
            
            size = struct.unpack('!I', size_data)[0]  # 大端序，4字节无符号整数
            
            if size == 0:
                print("图像大小为0")
                return None
            
            # 接收图像数据
            img_data = self._recv_all(size)
            if img_data is None or len(img_data) != size:
                print(f"接收图像数据不完整: 期望{size}字节，实际{len(img_data) if img_data else 0}字节")
                return None
            
            # 解码图像
            img_array = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("图像解码失败")
                return None
            
            return frame
            
        except Exception as e:
            print(f"接收图像错误: {e}")
            self.connected = False
            return None
    
    def set_exposure_params(self, exposure_time: int, gain: int) -> bool:
        """
        设置曝光参数
        
        Args:
            exposure_time: 曝光时间（微秒）
            gain: 增益值（0-255）
        
        Returns:
            bool: 设置是否成功
        """
        cmd = f"SET_EXP_PARAMS {exposure_time} {gain}"
        return self.send_command(cmd)
    
    def capture_image(self, auto_reconnect: bool = True) -> Optional[np.ndarray]:
        """
        采集图像
        
        Args:
            auto_reconnect: 如果连接失败，是否自动重连，默认True
        
        Returns:
            np.ndarray: 图像数据（BGR格式），失败返回None
        """
        # 检查连接状态
        if not self._check_connection():
            if auto_reconnect:
                print("连接已断开，尝试重新连接...")
                if not self.connect(max_retries=2, retry_delay=1.0):
                    print("重新连接失败")
                    return None
                print("重新连接成功")
            else:
                print("请先连接相机")
                return None
        
        # 发送获取图像命令
        if not self.send_command("GET_FRAME"):
            return None
        
        # 接收图像
        frame = self.receive_image()
        
        if frame is None:
            print("采集图像失败")
            return None
        
        return frame
    
    def close(self):
        """关闭连接"""
        if self.sock is not None:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False
    
    def get_camera_info(self) -> Optional[dict]:
        """获取相机信息"""
        return self.info


if __name__ == "__main__":
    # 测试3D相机采集
    camera = EpicEyeCamera()
    if camera.connect():
        img = camera.capture_image()
        if img is not None:
            cv2.imwrite("test_capture.png", img)
            print(f"图像已保存，尺寸: {img.shape}")
