"""
思谋读码器相机采图模块（基于 SmoreCamera SDK 示例逻辑）
"""
from __future__ import annotations

import os
import platform
import time
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_int,
    c_uint32,
    c_uint64,
    c_ubyte,
    c_void_p,
)
from enum import IntEnum
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameInfo(Structure):
    _fields_ = [
        ("format", c_int),
        ("bits", c_int),
        ("uBytes", c_uint32),
        ("iWidth", c_int),
        ("iHeight", c_int),
        ("iChannel", c_int),
        ("uFrameID", c_uint64),
    ]


class smFrameBuffer(Structure):
    _fields_ = [
        ("frame", FrameInfo),
        ("pBuffer", POINTER(c_ubyte)),
        ("pBufferRGB", POINTER(c_ubyte)),
    ]


class ErrorCode(IntEnum):
    SM_OK = 1
    SM_ERROR = 0
    SM_CONNECT_ERROR = -1
    SM_INVALID_CONNECT = -2
    SM_GET_FRAME_ERROR = -3
    SM_INVALID_HANDLE = -4
    SM_VS_BAR_MORE_GET_ERROR = -5
    SM_SET_NOT = -6
    SM_Call_NOT = -7
    SM_NO_CAMERA = -8
    SM_OFFLINE_CAMERA = -9


class CameraType(IntEnum):
    NNone = -1
    VN = 0
    VS = 1


class smTransmit(IntEnum):
    MainRunOnce = 0
    MainRunCycle = 1
    StopRun = 2
    StatisticsClear = 3
    Get = 4
    IOTrigger = 5
    NetTrigger = 6


class smAlgTool(Structure):
    _fields_ = [
        ("rtype", c_int),
        ("rvalue", POINTER(c_void_p) * 300),
        ("count", c_int),
        ("result", c_bool),
    ]


class smProjectResult(Structure):
    _fields_ = [
        ("tools", POINTER(smAlgTool) * 20),
        ("toolCount", c_int),
        ("overallResult", c_bool),
        ("projectTolCount", c_int),
        ("projectNGCount", c_int),
        ("totalCost", c_int),
    ]


_SMCAMERA_DLL = None
_DLL_DIR_HANDLES = []


def _resolve_dll_path(dll_path: Optional[str]) -> str:
    if dll_path:
        if os.path.isdir(dll_path):
            return os.path.join(dll_path, "SMCamera.dll")
        return dll_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dll_arch = "Win64" if platform.architecture()[0] == "64bit" else "Win32"
    return os.path.join(script_dir, "smore_camera_sdk", dll_arch, "SMCamera.dll")


def _add_dll_search_dir(dll_path: str) -> None:
    dll_dir = os.path.dirname(dll_path)
    if os.path.isdir(dll_dir) and hasattr(os, "add_dll_directory"):
        _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir))


def _load_dll(dll_path: Optional[str] = None) -> CDLL:
    global _SMCAMERA_DLL
    if _SMCAMERA_DLL is not None:
        return _SMCAMERA_DLL

    resolved_path = _resolve_dll_path(dll_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"未找到 SMCamera.dll: {resolved_path}")

    _add_dll_search_dir(resolved_path)
    smcamera_dll = CDLL(resolved_path)

    smcamera_dll.smOpenByIp.argtypes = [c_char_p, POINTER(c_uint32), c_int]
    smcamera_dll.smOpenByIp.restype = c_int

    smcamera_dll.smClose.argtypes = [POINTER(c_uint32)]
    smcamera_dll.smClose.restype = c_int

    smcamera_dll.smSendTransmit.argtypes = [POINTER(c_uint32), c_int]
    smcamera_dll.smSendTransmit.restype = c_int

    smcamera_dll.smGetFrameResult.argtypes = [
        POINTER(c_uint32),
        POINTER(POINTER(smFrameBuffer)),
        POINTER(POINTER(smProjectResult)),
    ]
    smcamera_dll.smGetFrameResult.restype = c_int

    smcamera_dll.smReleaseFrame.argtypes = [POINTER(c_uint32)]
    smcamera_dll.smReleaseFrame.restype = c_int

    smcamera_dll.smGetInfo.argtypes = [
        POINTER(c_uint32),
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_int),
    ]
    smcamera_dll.smGetInfo.restype = c_int

    _SMCAMERA_DLL = smcamera_dll
    return smcamera_dll


class SmoreCamera:
    """思谋科技读码器相机图像采集类（基于 SDK DLL）"""

    def __init__(
        self,
        ip: str,
        camera_type: CameraType = CameraType.VS,
        dll_path: Optional[str] = None,
    ):
        """
        Args:
            ip: 相机IP地址（示例: "169.254.207.24"）
            camera_type: 相机类型（VN/VS）
            dll_path: SMCamera.dll 路径，可选
        """
        self.ip = ip
        self.camera_type = camera_type
        self.handle = c_uint32()
        self.connected = False
        self._dll = _load_dll(dll_path)

    def connect(self) -> bool:
        """连接相机"""
        if not self.ip:
            print("未提供相机IP地址")
            return False

        ret = self._dll.smOpenByIp(self.ip.encode("utf-8"), byref(self.handle), self.camera_type.value)
        if ret != ErrorCode.SM_OK.value:
            print(f"{self.ip} 连接失败，错误码: {ret}")
            self.connected = False
            return False

        self.connected = True
        print(f"{self.ip} 连接相机成功")
        return True

    def _ensure_connected(self, auto_reconnect: bool) -> bool:
        if self.connected:
            return True
        if not auto_reconnect:
            print("相机未连接")
            return False
        print("相机未连接，尝试重新连接...")
        return self.connect()

    def _frame_to_bgr(self, frame_buffer: smFrameBuffer) -> Optional[np.ndarray]:
        if frame_buffer is None:
            return None

        width = frame_buffer.frame.iWidth
        height = frame_buffer.frame.iHeight
        channels = frame_buffer.frame.iChannel

        if width <= 0 or height <= 0:
            return None

        # if frame_buffer.pBufferRGB:
        #     buffer_size = width * height * 3
        #     rgb_flat = np.ctypeslib.as_array(frame_buffer.pBufferRGB, shape=(buffer_size,))
        #     if rgb_flat.size == 0:
        #         return None
        #     rgb_image = rgb_flat.reshape((height, width, 3))
        #     return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if frame_buffer.pBuffer:
            buffer_size = width * height * max(1, channels)
            raw_flat = np.ctypeslib.as_array(frame_buffer.pBuffer, shape=(buffer_size,))
            if raw_flat.size == 0:
                return None
            raw_image = raw_flat.reshape((height, width, max(1, channels)))
            if channels == 1:
                return cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            return raw_image.copy()

        return None

    def _capture_frame(self, max_retries: int, retry_delay: float) -> Optional[np.ndarray]:
        for attempt in range(max_retries):
            ret = self._dll.smSendTransmit(byref(self.handle), smTransmit.MainRunOnce.value)
            if ret != ErrorCode.SM_OK.value:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None

            frame_buffer = POINTER(smFrameBuffer)()
            project_result = POINTER(smProjectResult)()
            try:
                ret = self._dll.smGetFrameResult(
                    byref(self.handle),
                    byref(frame_buffer),
                    byref(project_result),
                )
                if ret != ErrorCode.SM_OK.value:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None

                image = self._frame_to_bgr(frame_buffer.contents if frame_buffer else None)
                if image is None:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                return image
            finally:
                self._dll.smReleaseFrame(byref(self.handle))
        return None

    def capture_image(
        self,
        auto_reconnect: bool = True,
        max_retries: int = 100,
        retry_delay: float = 1,
        drop_frames: int = 2,
    ) -> Optional[np.ndarray]:
        """
        采集图像（基于示例逻辑：触发 -> 获取帧）
        """
        if not self._ensure_connected(auto_reconnect):
            return None

        for _ in range(max(0, drop_frames)):
            self._capture_frame(max_retries=3, retry_delay=retry_delay)

        image = self._capture_frame(max_retries=max_retries, retry_delay=retry_delay)
        if image is None:
            print("获取图像失败或图像为空")
        return image

    def get_resolution(self) -> Optional[Tuple[int, int, int]]:
        """获取图像宽高和通道数"""
        if not self._ensure_connected(auto_reconnect=False):
            return None

        width = c_int()
        height = c_int()
        channel = c_int()
        ret = self._dll.smGetInfo(byref(self.handle), byref(width), byref(height), byref(channel))
        if ret != ErrorCode.SM_OK.value:
            print(f"获取分辨率失败，错误码: {ret}")
            return None
        return width.value, height.value, channel.value

    def close(self) -> None:
        """关闭相机连接"""
        if not self.connected:
            return
        ret = self._dll.smClose(byref(self.handle))
        if ret == ErrorCode.SM_OK.value:
            print("关闭相机成功")
        self.connected = False

    def get_camera_info(self) -> dict:
        """获取相机信息"""
        return {"ip": self.ip, "type": self.camera_type.name}


def _test_capture() -> None:
    ip = input("请输入读码器相机IP: ").strip()
    if not ip:
        print("未输入IP，测试结束")
        return

    camera = SmoreCamera(ip=ip)
    if not camera.connect():
        return

    image = camera.capture_image()
    if image is None:
        print("采集失败")
        camera.close()
        return

    output_path = "test_smore_capture.png"
    cv2.imwrite(output_path, image)
    print(f"采集成功，已保存: {output_path}, 尺寸: {image.shape}")
    camera.close()


if __name__ == "__main__":
    _test_capture()
