#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-BaIoT 自编码器训练系统 - GUI主入口

这是系统的主程序入口文件，负责整合所有模块化组件并启动GUI应用程序。
该文件加载所有必要的模块，初始化配置，并启动主窗口。
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.components.main_window import MainWindow


def main():
    """
    主函数，启动应用程序
    """
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
