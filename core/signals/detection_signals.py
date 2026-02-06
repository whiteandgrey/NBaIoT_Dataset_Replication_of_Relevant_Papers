from PyQt5.QtCore import QObject, pyqtSignal


class IntrusionDetectionSignals(QObject):
    """
    入侵检测与评估线程信号类
    
    用于在入侵检测评估线程和主线程之间传递消息，包括评估开始、
    进度更新、数据更新、评估完成、错误、日志、状态更新、
    文件生成和保存完成等信号。
    """
    started = pyqtSignal(str)           # 评估开始信号
    progress = pyqtSignal(dict)         # 进度更新信号
    data_updated = pyqtSignal(dict)     # 数据更新信号
    completed = pyqtSignal(dict)        # 评估完成信号
    error = pyqtSignal(str)             # 错误信号
    log = pyqtSignal(str)               # 日志信号
    status_update = pyqtSignal(str)     # 状态更新信号
    file_generated = pyqtSignal(str)    # 文件生成信号
    save_completed = pyqtSignal(str)    # 保存完成信号
