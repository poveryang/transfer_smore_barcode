"""
简单打包脚本：使用 PyInstaller 命令行方式打包
这种方式更稳定，避免 spec 文件的复杂配置问题
"""
import os
import sys
import subprocess
import shutil
import glob

def main():
    """主函数：执行打包流程"""
    print("=" * 60)
    print("使用命令行方式打包 Python 程序为 exe")
    print("=" * 60)
    
    # 检查 PyInstaller 是否已安装
    try:
        import PyInstaller
        print(f"✓ PyInstaller 已安装，版本: {PyInstaller.__version__}")
    except ImportError:
        print("✗ PyInstaller 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller 安装完成")
    
    # 清理之前的构建文件
    print("\n清理之前的构建文件...")
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"✓ 已删除 {dir_name}/")
            except Exception as e:
                print(f"✗ 删除 {dir_name}/ 失败: {e}")
    
    # 收集需要包含的数据文件
    datas = []
    
    # 包含配置文件
    if os.path.exists('ui_config.json'):
        datas.append('--add-data')
        datas.append('ui_config.json;.')
    
    # 收集 smore_camera_sdk 中的 DLL 文件
    for arch in ['Win64', 'Win32']:
        dll_dir = os.path.join('smore_camera_sdk', arch)
        if os.path.exists(dll_dir):
            dll_files = glob.glob(os.path.join(dll_dir, '*.dll'))
            if dll_files:
                # 将所有 DLL 文件添加到数据文件
                for dll_file in dll_files:
                    datas.append('--add-data')
                    datas.append(f'{dll_file};{dll_dir}')
    
    # 构建 PyInstaller 命令
    # 使用命令行方式，避免 spec 文件的复杂配置
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=transfer_smore_barcode",
        "--onefile",  # 单文件模式
        "--console",  # 先使用控制台模式以便调试，成功后可以改为 --windowed
        "--clean",
        "--noconfirm",
        # 不压缩，避免启动问题
        "--noupx",
        # 隐藏导入
        "--hidden-import=epiceye",
        "--hidden-import=epiceye.epiceye",
        "--hidden-import=epiceye.epicraw_parser",
        "--hidden-import=epiceye_camera",
        "--hidden-import=smore_camera",
        "--hidden-import=calibration",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL.ImageTk",
        "--hidden-import=paramiko",
        "--hidden-import=requests",
        "--hidden-import=tkinter",
        # 排除不需要的模块
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--exclude-module=jupyter",
        "--exclude-module=IPython",
        "--exclude-module=notebook",
    ]
    
    # 添加数据文件
    cmd.extend(datas)
    
    # 添加主程序文件
    cmd.append('run_ui.py')
    
    print("\n开始打包...")
    print("-" * 60)
    print("执行命令:")
    print(" ".join(cmd))
    print("-" * 60)
    
    try:
        subprocess.check_call(cmd)
        print("-" * 60)
        print("\n✓ 打包完成！")
        print(f"\n可执行文件位置: dist/transfer_smore_barcode.exe")
        print("\n提示：")
        print("  - 首次运行可能需要一些时间")
        print("  - 如果遇到 DLL 缺失错误，请确保 smore_camera_sdk 文件夹中的 DLL 已正确包含")
        print("  - 可以将 dist/transfer_smore_barcode.exe 单独分发给用户")
        print("  - 如果需要调试，可以将 --windowed 改为 --console 查看错误信息")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 打包失败: {e}")
        print("\n请检查：")
        print("  1. Python 版本是否兼容（建议 Python 3.8-3.11）")
        print("  2. 所有依赖是否已正确安装")
        print("  3. PyInstaller 版本是否最新（pip install --upgrade pyinstaller）")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
