from PyQt5.QtCore import QObject, pyqtSignal


class TrainingSignals(QObject):
    """
    训练线程信号类
    
    用于在训练线程和主线程之间传递消息，包括训练开始、进度更新、
    Epoch完成、阶段完成、设备完成、错误、训练完成、日志和状态更新等信号。
    """
    started = pyqtSignal(str)           # 训练开始信号
    progress = pyqtSignal(dict)         # 进度更新信号
    epoch_completed = pyqtSignal(dict)  # Epoch完成信号
    phase_completed = pyqtSignal(dict)  # 阶段完成信号
    device_completed = pyqtSignal(dict) # 设备完成信号
    error = pyqtSignal(str)             # 错误信号
    finished = pyqtSignal(dict)         # 训练完成信号
    log = pyqtSignal(str)               # 日志信号
    status_update = pyqtSignal(str)     # 状态更新信号
