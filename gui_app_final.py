"""
N-BaIoT自编码器训练系统 - 图形用户界面版 (最终修复版)
GUI-based N-BaIoT Autoencoder Training System (Final Fixed Version)

修复内容:
1. 修复图表标题显示方框问题
2. 修复最终训练阶段KeyError: 'training_time'错误
3. 优化实时曲线显示，限制历史数据点数量

依赖: PyQt5, TensorFlow, matplotlib, numpy, pandas
安装: pip install PyQt5 matplotlib tensorflow numpy pandas scikit-learn seaborn scipy joblib
"""
import sys
import os
import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Callable

# 设置环境变量（必须在导入TensorFlow之前）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONWARNINGS'] = 'ignore'

# 导入配置
from config import Config

# 尝试从配置文件读取GPU设置（在导入TensorFlow之前）
def load_config_from_file():
    """从配置文件加载设置"""
    config_file = "config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                Config.USE_GPU = config.get('use_gpu', False)
                Config.GPU_DEVICES = config.get('gpu_devices', "0")
                Config.GPU_MEMORY_LIMIT = config.get('gpu_memory_limit')
                print(f"📝 Loaded config from file: use_gpu={Config.USE_GPU}, gpu_devices={Config.GPU_DEVICES}")
        except Exception as e:
            print(f"⚠️ Failed to load config file: {e}")

# 加载配置文件
load_config_from_file()

# 设置环境（必须在导入TensorFlow之前）
Config.setup_environment()

# 尝试导入GUI库
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QFormLayout, QTabWidget, QGroupBox, QLabel,
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
        QPushButton, QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
        QHeaderView, QFileDialog, QMessageBox, QStatusBar, QToolBar,
        QAction, QSplitter, QFrame, QSlider, QRadioButton, QButtonGroup,
        QListWidget, QListWidgetItem, QScrollArea, QSizePolicy, QDialog
    )
    from PyQt5.QtCore import (
        Qt, QTimer, pyqtSignal, QObject, QThread, pyqtSlot, QMutex,
        QSize, QRect
    )
    from PyQt5.QtGui import (
        QFont, QColor, QPalette, QPixmap, QIcon, QPainter, QPen,
        QLinearGradient, QBrush
    )
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️ PyQt5未安装，请运行: pip install PyQt5")

# 尝试导入图表库
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    # 设置matplotlib使用系统默认字体，避免中文显示问题
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib未安装，请运行: pip install matplotlib")

# 导入TensorFlow和项目模块
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_processor import NBaIoTDataProcessor
from model import Autoencoder
from trainer import AutoencoderTrainer
from visualizer import ScientificVisualizer


# ============================================================================
# 自定义信号类用于线程通信
# ============================================================================
class TrainingSignals(QObject):
    """训练线程信号类"""
    started = pyqtSignal(str)           # 训练开始信号
    progress = pyqtSignal(dict)         # 进度更新信号
    epoch_completed = pyqtSignal(dict)  # Epoch完成信号
    phase_completed = pyqtSignal(dict)  # 阶段完成信号
    device_completed = pyqtSignal(dict) # 设备完成信号
    error = pyqtSignal(str)             # 错误信号
    finished = pyqtSignal(dict)         # 训练完成信号
    log = pyqtSignal(str)               # 日志信号
    status_update = pyqtSignal(str)     # 状态更新信号


# ============================================================================
# 自定义Keras回调 - 用于控制训练流程
# ============================================================================
class TrainingControlCallback(Callback):
    """
    自定义回调用于控制训练流程
    实现可靠的暂停/停止功能
    """
    
    def __init__(self, worker_signals, worker_ref):
        """
        初始化回调
        
        Args:
            worker_signals: 训练信号对象
            worker_ref: 对TrainingWorker的弱引用，用于检查状态
        """
        super().__init__()
        self.worker_signals = worker_signals
        self.worker_ref = worker_ref
        self.epoch_data = {}
        
    def on_epoch_begin(self, epoch, logs=None):
        """每个epoch开始时检查停止状态"""
        worker = self.worker_ref()
        if worker is not None:
            if worker.should_stop:
                self.model.stop_training = True
                self.worker_signals.log.emit("🛑 停止信号已收到，正在停止训练...")
    
    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时检查暂停状态并发送数据"""
        worker = self.worker_ref()
        if worker is None:
            return
            
        # 发送epoch完成信号
        if logs:
            self.epoch_data = {
                'epoch': epoch + 1,
                'train_loss': float(logs.get('loss', 0)),
                'val_loss': float(logs.get('val_loss', logs.get('loss', 0))),
                'phase': getattr(worker, 'current_phase', 'Training'),
                'total_epochs': getattr(worker, 'total_epochs', 100)
            }
            self.worker_signals.epoch_completed.emit(self.epoch_data)
        
        # 检查暂停状态
        if worker.is_paused:
            self.worker_signals.status_update.emit("已暂停 - 等待恢复...")
            while worker.is_paused and not worker.should_stop:
                time.sleep(0.1)
                
        # 检查停止状态
        if worker.should_stop:
            self.model.stop_training = True
            
        return super().on_epoch_end(epoch, logs)
    
    def on_batch_end(self, batch, logs=None):
        """每个batch结束时检查状态"""
        worker = self.worker_ref()
        if worker is None:
            return
            
        if worker.should_stop:
            self.model.stop_training = True
            return
            
        if worker.is_paused:
            while worker.is_paused and not worker.should_stop:
                time.sleep(0.05)


# ============================================================================
# 训练工作线程
# ============================================================================
class TrainingWorker(QThread):
    """训练工作线程 - 在后台执行训练任务"""
    
    def __init__(self, config: Dict, signals: TrainingSignals):
        super().__init__()
        self.config = config
        self.signals = signals
        self.is_paused = False
        self.should_stop = False
        self.is_running = False
        self.mutex = QMutex()
        
        # 训练阶段跟踪
        self.current_phase = "初始训练"
        self.total_epochs = 100
        
    def run(self):
        """执行训练"""
        self.is_running = True
        self.should_stop = False
        self.is_paused = False
        
        try:
            # 设置环境
            self._setup_environment()
            
            # 初始化数据处理器
            data_processor = NBaIoTDataProcessor(Config)
            
            # 获取要训练的设备列表
            devices_to_train = self._get_devices_to_train(data_processor)
            
            if not devices_to_train:
                self.signals.error.emit("没有选择要训练的设备")
                self.is_running = False
                return
            
            # 初始化可视化器
            visualizer = ScientificVisualizer(Config)
            
            # 记录总结果
            all_results = []
            total_start_time = time.time()
            
            # 遍历设备进行训练
            for i, device_name in enumerate(devices_to_train):
                if self.should_stop:
                    break
                    
                self._wait_if_paused()
                
                if self.should_stop:
                    break
                    
                self.signals.status_update.emit(f"正在训练: {device_name} ({i+1}/{len(devices_to_train)})")
                self.signals.started.emit(device_name)
                
                # 训练单个设备
                result = self._train_device(
                    device_name, data_processor, visualizer
                )
                
                if result:
                    all_results.append(result)
                
                if self.should_stop:
                    break
                    
                self.signals.device_completed.emit({
                    'device': device_name,
                    'result': result,
                    'progress': (i + 1) / len(devices_to_train) * 100
                })
            
            # 训练完成
            total_time = time.time() - total_start_time
            
            self.signals.finished.emit({
                'results': all_results,
                'total_time': total_time,
                'total_devices': len(all_results)
            })
            
        except Exception as e:
            import traceback
            error_msg = f"训练错误: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        finally:
            self.is_running = False
    
    def _wait_if_paused(self):
        """等待恢复"""
        while self.is_paused and not self.should_stop:
            time.sleep(0.1)
    
    def _setup_environment(self):
        """设置环境"""
        Config.USE_GPU = self.config.get('use_gpu', False)
        Config.GPU_DEVICES = self.config.get('gpu_devices', "0")
        Config.GPU_MEMORY_LIMIT = self.config.get('gpu_memory_limit')
        Config.DATA_ROOT = self.config.get('data_root', Config.DATA_ROOT)
        Config.OUTPUT_DIR = self.config.get('output_dir', Config.OUTPUT_DIR)
        
        Config.DEFAULT_LEARNING_RATE = self.config.get('learning_rate', 0.001)
        Config.DEFAULT_EPOCHS = self.config.get('epochs', 100)
        Config.DEFAULT_BATCH_SIZE = self.config.get('batch_size', 64)
        
        Config.ENCODER_RATIOS = self.config.get('encoder_ratios', [0.75, 0.50, 0.33, 0.25])
        Config.DECODER_RATIOS = self.config.get('decoder_ratios', [0.33, 0.50, 0.75, 1.0])
        Config.ACTIVATION = self.config.get('activation', 'relu')
        Config.USE_BATCH_NORM = self.config.get('use_batch_norm', False)
        Config.DROPOUT_RATE = self.config.get('dropout_rate', 0.0)
        
        Config.EARLY_STOPPING_PATIENCE = self.config.get('early_stopping_patience', 15)
        Config.REDUCE_LR_PATIENCE = self.config.get('reduce_lr_patience', 10)
        
        Config.LEARNING_RATES = self.config.get('learning_rates', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        Config.EPOCHS_OPTIONS = self.config.get('epochs_options', [50, 100, 150, 200])
        
        # 文件保存配置
        Config.SAVE_LOG_FILE = self.config.get('save_log_file', True)
        Config.SAVE_MODEL = self.config.get('save_model', True)
        Config.SAVE_BEST_MODEL_ONLY = self.config.get('save_best_model_only', True)
        Config.SAVE_TRAINING_HISTORY = self.config.get('save_training_history', True)
        Config.SAVE_HYPERPARAMETER_TUNING_RESULTS = self.config.get('save_hyperparam_results', True)
        Config.SAVE_SCALER = self.config.get('save_scaler', True)
        Config.SAVE_TENSORBOARD_LOGS = self.config.get('save_tensorboard', False)
        Config.PLOT_SAVE = self.config.get('plot_save', True)
        
        # 图表类型配置
        Config.PLOT_TRAINING_LOSS_CURVE = self.config.get('plot_training_loss_curve', True)
        Config.PLOT_TRAINING_MAE_CURVE = self.config.get('plot_training_mae_curve', True)
        Config.PLOT_TRAINING_LR_CURVE = self.config.get('plot_training_lr_curve', True)
        Config.PLOT_HYPERPARAM_HEATMAP = self.config.get('plot_hyperparam_heatmap', True)
        Config.PLOT_HYPERPARAM_CONTOUR = self.config.get('plot_hyperparam_contour', True)
        Config.PLOT_HYPERPARAM_3D = self.config.get('plot_hyperparam_3d', False)
        Config.PLOT_LOSS_DISTRIBUTION = self.config.get('plot_loss_distribution', True)
        Config.PLOT_LOSS_HISTOGRAM = self.config.get('plot_loss_histogram', True)
        Config.PLOT_LOSS_BOX_PLOT = self.config.get('plot_loss_boxplot', True)
        Config.PLOT_LOSS_VIOLIN_PLOT = self.config.get('plot_loss_violin', True)
        Config.PLOT_PERFORMANCE_METRICS = self.config.get('plot_performance_metrics', True)
        Config.PLOT_LEARNING_RATE_SCHEDULE = self.config.get('plot_lr_schedule', True)
        Config.PLOT_GRADIENT_FLOW = self.config.get('plot_gradient_flow', False)
        Config.PLOT_DATA_DISTRIBUTION = self.config.get('plot_data_distribution', True)
        Config.PLOT_FEATURE_CORRELATION = self.config.get('plot_feature_correlation', False)
        Config.PLOT_PCA_VISUALIZATION = self.config.get('plot_pca_visualization', False)
        Config.PLOT_TRAINING_TIME_ANALYSIS = self.config.get('plot_training_time_analysis', True)
        Config.PLOT_EPOCH_TIME_DISTRIBUTION = self.config.get('plot_epoch_time_distribution', True)
        Config.PLOT_DEVICE_COMPARISON = self.config.get('plot_device_comparison', True)
        Config.PLOT_PHASE_COMPARISON = self.config.get('plot_phase_comparison', True)
        Config.PLOT_PERFORMANCE_RANKING = self.config.get('plot_performance_ranking', True)
        Config.PLOT_COMPREHENSIVE_SUMMARY = self.config.get('plot_comprehensive_summary', True)
        Config.PLOT_TRAINING_REPORT = self.config.get('plot_training_report', True)
        
        # 注意：Config.setup_environment()已在导入时调用，这里不再重复调用
        # 只需要设置TensorFlow和目录
        Config.setup_tensorflow()
        Config.setup_directories()
        
    def _get_devices_to_train(self, data_processor) -> List[str]:
        """获取要训练的设备列表"""
        selected_devices = self.config.get('selected_devices', [])
        
        if not selected_devices:
            return data_processor.get_available_devices()
        
        available = data_processor.get_available_devices()
        valid_devices = [d for d in selected_devices if d in available]
        
        if not valid_devices:
            return available
        
        return valid_devices
    
    def _train_device(self, device_name: str, data_processor, visualizer) -> Optional[Dict]:
        """训练单个设备"""
        import weakref
        
        # 加载数据
        self.signals.log.emit(f"📥 正在加载 {device_name} 的数据...")
        data = data_processor.load_device_data(device_name)
        
        if data is None:
            self.signals.error.emit(f"❌ 无法加载 {device_name} 的数据")
            return None
        
        # 划分数据
        if Config.TIME_ORDERED:
            DStrn, DSopt, DStst = data_processor.split_data_chronologically(data)
        else:
            DStrn, DSopt, DStst = data_processor.split_data_randomly(data)
        
        # 预处理数据
        self.signals.log.emit(f"🔧 正在预处理数据...")
        DStrn_processed = data_processor.preprocess_data(DStrn, fit_scaler=True)
        DSopt_processed = data_processor.preprocess_data(DSopt, fit_scaler=False)
        
        # 创建训练数据
        (X_train, y_train), (X_val, y_val) = data_processor.create_numpy_datasets(
            DStrn_processed, DSopt_processed
        )
        
        # 获取数据信息
        data_info = {
            'device_name': device_name,
            'n_features': data.shape[1],
            'n_samples': len(data),
            'train_samples': len(DStrn),
            'val_samples': len(DSopt),
            'test_samples': len(DStst),
        }
        
        # 使用实际数据的特征维度来构建模型
        actual_input_dim = data.shape[1]
        self.signals.log.emit(f"📐 实际数据特征维度: {actual_input_dim}")
        
        # 创建训练器
        trainer = AutoencoderTrainer(Config, device_name)
        
        # 创建回调对象
        control_callback = TrainingControlCallback(self.signals, weakref.ref(self))
        
        # 阶段1: 初始训练
        self.signals.log.emit(f"🚀 开始初始训练...")
        self.signals.status_update.emit(f"阶段1: 初始训练")
        self.current_phase = "初始训练"
        
        initial_start_time = time.time()
        initial_result = self._train_with_callback(
            trainer, X_train, y_train, X_val, y_val,
            phase_name="初始训练",
            total_epochs=Config.DEFAULT_EPOCHS,
            control_callback=control_callback,
            input_dim=actual_input_dim
        )
        initial_training_time = time.time() - initial_start_time
        
        # 保存初始训练历史
        trainer.training_history['initial_train'] = {
            'history': initial_result['history'],
            'training_time': initial_training_time,
            'best_val_loss': initial_result['best_val_loss']
        }
        
        self.signals.log.emit(f"✅ 初始训练完成。最佳损失: {initial_result['best_val_loss']:.6f}")
        
        if self.should_stop:
            return None
        
        # 阶段2: 超参数调优
        self.signals.log.emit(f"🔍 开始超参数调优...")
        self.signals.status_update.emit(f"阶段2: 超参数调优")
        self.current_phase = "超参数调优"
        
        tuning_result = self._train_with_hyperparameter_tuning(
            trainer, X_train, y_train, X_val, y_val,
            control_callback=control_callback,
            input_dim=actual_input_dim
        )
        
        self.signals.log.emit(f"✅ 超参数调优完成。最佳参数: LR={tuning_result['lr']:.6f}, Epochs={tuning_result['epochs']}")
        
        if self.should_stop:
            return None
        
        # 阶段3: 最终训练
        self.signals.log.emit(f"🎯 开始最终训练...")
        self.signals.status_update.emit(f"阶段3: 最终训练")
        self.current_phase = "最终训练"
        
        # 合并训练和验证数据
        X_combined = np.concatenate([X_train, X_val], axis=0)
        y_combined = np.concatenate([y_train, y_val], axis=0)
        
        final_start_time = time.time()
        final_result = self._train_with_callback(
            trainer, X_combined, y_combined, None, None,
            phase_name="最终训练",
            total_epochs=trainer.training_history['best_params']['epochs'] if trainer.training_history.get('best_params') else Config.DEFAULT_EPOCHS,
            control_callback=control_callback,
            input_dim=actual_input_dim
        )
        final_training_time = time.time() - final_start_time
        
        # 保存最终训练历史
        trainer.training_history['final_train'] = {
            'history': final_result['history'],
            'training_time': final_training_time,
            'best_val_loss': final_result['best_val_loss']
        }
        
        self.signals.log.emit(f"✅ 最终训练完成。最终损失: {final_result['best_val_loss']:.6f}")
        
        # 生成可视化
        self.signals.log.emit(f"📊 生成可视化图表...")
        device_start_time = time.time()
        visualizer.generate_all_plots(trainer, device_name, data_info)

        # 保存scaler
        if Config.SAVE_SCALER:
            scaler_path = os.path.join(trainer.device_output_dir, 'scaler.pkl')
            data_processor.save_scaler(scaler_path)
        
        # 返回结果（安全访问training_history中的数据）
        final_train_data = trainer.training_history.get('final_train') or {}
        return {
            'device_name': device_name,
            'best_params': trainer.training_history.get('best_params'),
            'best_val_loss': trainer.training_history['best_val_loss'],
            'final_train_loss': final_result['best_val_loss'],
            'training_time': final_train_data.get('training_time', 0),
            'data_info': data_info,
            'model_path': os.path.join(trainer.device_output_dir, 'final_model.h5')
        }
    
    def _train_with_callback(self, trainer, X_train, y_train, X_val, y_val, 
                             phase_name: str, total_epochs: int,
                             control_callback: TrainingControlCallback = None,
                             input_dim: int = None) -> Dict:
        """带回调的训练"""
        import weakref
        
        self.current_phase = phase_name
        self.total_epochs = total_epochs
        
        # 创建模型（使用实际输入维度）
        autoencoder = Autoencoder(Config)
        model = autoencoder.build(input_dim=input_dim)
        trainer.model = model
        
        # 编译模型
        lr = trainer.config.DEFAULT_LEARNING_RATE
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae']
        )
        
        # 创建回调
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=trainer.config.EARLY_STOPPING_PATIENCE,
                mode='min',
                min_delta=trainer.config.MIN_DELTA,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=trainer.config.REDUCE_LR_FACTOR,
                patience=trainer.config.REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                mode='min',
                verbose=0
            )
        ]
        
        if control_callback is None:
            control_callback = TrainingControlCallback(self.signals, weakref.ref(self))
        callbacks.append(control_callback)
        
        # 准备训练数据
        if X_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=total_epochs,
            batch_size=trainer.config.DEFAULT_BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )
        
        # 获取最佳验证损失
        history_dict = history.history
        if 'val_loss' in history_dict:
            best_val_loss = min(history_dict['val_loss'])
        else:
            best_val_loss = history_dict['loss'][-1] if history_dict['loss'] else float('inf')
        
        return {
            'history': history_dict,
            'best_val_loss': best_val_loss
        }
    
    def _train_with_hyperparameter_tuning(self, trainer, X_train, y_train, X_val, y_val,
                                           control_callback: TrainingControlCallback = None,
                                           input_dim: int = None):
        """超参数调优（带图表更新）- 修复版，包含training_time"""
        import weakref
        
        results = []
        best_val_loss = float('inf')
        best_params = None
        
        # 遍历超参数组合
        for lr in trainer.config.LEARNING_RATES:
            for epochs in trainer.config.EPOCHS_OPTIONS:
                if self.should_stop:
                    break
                    
                tuning_start_time = time.time()
                
                self.signals.log.emit(f"🧪 测试: LR={lr:.6f}, Epochs={epochs}")
                self.signals.status_update.emit(f"超参数调优: LR={lr:.6f}, Epochs={epochs}")
                
                # 重新创建模型（使用实际输入维度）
                autoencoder = Autoencoder(Config)
                model = autoencoder.build(input_dim=input_dim)
                trainer.model = model
                
                # 编译模型
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='mse',
                    metrics=['mae']
                )
                
                # 创建回调
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=trainer.config.EARLY_STOPPING_PATIENCE,
                        mode='min',
                        min_delta=trainer.config.MIN_DELTA,
                        restore_best_weights=True,
                        verbose=0
                    )
                ]
                
                if control_callback is None:
                    control_callback = TrainingControlCallback(self.signals, weakref.ref(self))
                
                # 设置当前阶段
                old_phase = self.current_phase
                self.current_phase = f"超参数调优 (LR={lr:.4f})"
                
                # 训练模型
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=trainer.config.DEFAULT_BATCH_SIZE,
                    callbacks=callbacks + [control_callback],
                    verbose=0
                )
                
                # 恢复阶段名称
                self.current_phase = old_phase
                
                # 计算训练时间
                tuning_time = time.time() - tuning_start_time
                
                # 获取结果
                history_dict = history.history
                val_loss = min(history_dict['val_loss']) if 'val_loss' in history_dict else history_dict['loss'][-1]
                
                # 记录结果（包含training_time）
                result = {
                    'lr': lr,
                    'epochs': epochs,
                    'val_loss': val_loss,
                    'training_time': tuning_time  # 修复：添加training_time字段
                }
                results.append(result)
                
                # 发送epoch完成信号
                if history_dict['loss']:
                    self.signals.epoch_completed.emit({
                        'epoch': len(history_dict['loss']),
                        'train_loss': history_dict['loss'][-1],
                        'val_loss': val_loss,
                        'phase': "超参数调优",
                        'total_epochs': epochs
                    })
                
                # 更新最佳参数
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {'lr': lr, 'epochs': epochs}
                
                # 发送进度
                total_combinations = len(trainer.config.LEARNING_RATES) * len(trainer.config.EPOCHS_OPTIONS)
                current = results.index(result) + 1
                progress = current / total_combinations * 100
                self.signals.progress.emit({'progress': progress, 'loss': val_loss})
                
                self.signals.log.emit(f"   结果: 验证损失={val_loss:.6f}, 时间={tuning_time:.2f}秒")
                
                self._wait_if_paused()
                
                if self.should_stop:
                    break
        
        # 保存调优结果
        trainer.training_history['hyperparameter_tuning'] = results
        trainer.training_history['best_params'] = best_params
        trainer.training_history['best_val_loss'] = best_val_loss
        
        # 保存到文件
        tuning_results_path = os.path.join(trainer.device_output_dir, 'hyperparameter_tuning.json')
        with open(tuning_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return best_params if best_params else {'lr': trainer.config.DEFAULT_LEARNING_RATE, 'epochs': trainer.config.DEFAULT_EPOCHS}
    
    def pause(self):
        """暂停训练"""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
        self.signals.status_update.emit("已暂停 - 点击继续恢复训练")
        
    def resume(self):
        """恢复训练"""
        self.mutex.lock()
        self.is_paused = False
        self.mutex.unlock()
        self.signals.status_update.emit("正在恢复训练...")
        
    def stop(self):
        """停止训练"""
        self.mutex.lock()
        self.should_stop = True
        self.is_paused = False
        self.mutex.unlock()
        self.signals.status_update.emit("正在停止...")


# ============================================================================
# 实时图表组件（最终修复版）
# ============================================================================
class RealTimeChart(QWidget):
    """实时训练曲线图表组件（修复标题显示和历史数据过多问题）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.max_data_points = 200  # 限制最大数据点数量，避免曲线过多
        self.setup_plot()
        
    def setup_plot(self):
        """设置图表"""
        if not MATPLOTLIB_AVAILABLE:
            self.setStyleSheet("background-color: #2d2d2d;")
            return
            
        # 创建图表
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.patch.set_facecolor('#1e1e1e')
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        
        # 设置布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 初始化数据
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.current_phase = ""
        self.best_val_loss = float('inf')
        
        # 设置图表样式
        self.setup_chart_style()
        
    def setup_chart_style(self):
        """设置图表样式"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # 使用dark_background样式
        plt.style.use('dark_background')
        
        # 初始化图表
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        
        # 设置坐标轴颜色
        self.ax.spines['bottom'].set_color('#ffffff')
        self.ax.spines['top'].set_color('#ffffff')
        self.ax.spines['left'].set_color('#ffffff')
        self.ax.spines['right'].set_color('#ffffff')
        self.ax.tick_params(axis='x', colors='#ffffff')
        self.ax.tick_params(axis='y', colors='#ffffff')
        self.ax.yaxis.label.set_color('#ffffff')
        self.ax.xaxis.label.set_color('#ffffff')
        self.ax.title.set_color('#ffffff')
        
        # 初始化线条（使用更亮的颜色）
        self.train_line, = self.ax.plot([], [], color='#00BFFF', linewidth=2, 
                                        label='Training Loss', marker='o', markersize=3)
        self.val_line, = self.ax.plot([], [], color='#FF6B6B', linewidth=2, 
                                       label='Validation Loss', marker='s', markersize=3)
        
        # 初始化当前点标记
        self.train_point, = self.ax.plot([], [], 'o', color='#00BFFF', markersize=10, 
                                          label='Current Train')
        self.val_point, = self.ax.plot([], [], 's', color='#FF6B6B', markersize=10, 
                                         label='Current Val')
        
        # 使用英文标签
        self.ax.set_xlabel('Epoch', fontsize=12, color='white')
        self.ax.set_ylabel('Loss (MSE)', fontsize=12, color='white')
        self.ax.set_title('Training Progress - Real-time Loss Curve', 
                         fontsize=14, color='white', fontweight='bold')
        self.ax.legend(loc='upper right', facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        
    def update_chart(self, epoch: int, train_loss: float, val_loss: float, 
                     phase: str = "Training", total_epochs: int = 100):
        """更新图表数据（优化版：限制历史数据点数量）"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        else:
            self.val_losses.append(train_loss)
        
        # 限制数据点数量：如果超过最大限制，移除最旧的数据
        if len(self.epochs) > self.max_data_points:
            self.epochs = self.epochs[-self.max_data_points:]
            self.train_losses = self.train_losses[-self.max_data_points:]
            self.val_losses = self.val_losses[-self.max_data_points:]
        
        # 更新线条数据
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # 更新当前点标记（高亮显示最新点）
        if self.epochs:
            current_epoch = self.epochs[-1]
            current_train = self.train_losses[-1]
            current_val = self.val_losses[-1] if len(self.val_losses) > len(self.epochs) - 1 else current_train
            
            self.train_point.set_data([current_epoch], [current_train])
            self.val_point.set_data([current_epoch], [current_val])
        
        # 更新标题
        phase_display = {
            "初始训练": "Initial Training",
            "超参数调优": "Hyperparameter Tuning", 
            "最终训练": "Final Training",
            "Training": "Training"
        }.get(phase, phase)
        
        self.ax.set_title(f'{phase_display} - Loss Curve (Epoch {epoch}/{total_epochs})', 
                         fontsize=14, color='white', fontweight='bold')
        
        # 动态调整坐标轴
        if len(self.epochs) > 1:
            x_max = max(self.epochs) * 1.1
            self.ax.set_xlim(0, max(100, x_max))
            
            all_losses = self.train_losses + self.val_losses
            if all_losses:
                y_max = max(all_losses) * 1.2
                y_min = min(all_losses) * 0.8
                self.ax.set_ylim(max(0, y_min), y_max)
        
        # 刷新图表
        self.canvas.draw_idle()
        
    def clear_chart(self):
        """清空图表"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.train_point.set_data([], [])
        self.val_point.set_data([], [])
        
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Training Progress - Real-time Loss Curve', 
                         fontsize=14, color='white', fontweight='bold')
        
        self.canvas.draw_idle()
        
    def resizeEvent(self, event):
        """响应窗口大小变化"""
        super().resizeEvent(event)
        if MATPLOTLIB_AVAILABLE:
            self.figure.tight_layout()


# ============================================================================
# 配置面板组件
# ============================================================================
class ConfigPanel(QWidget):
    """配置面板组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        
        # 添加各个配置页面
        self.tab_widget.addTab(self.create_basic_config(), "基础配置")
        self.tab_widget.addTab(self.create_model_config(), "模型架构")
        self.tab_widget.addTab(self.create_training_config(), "训练参数")
        self.tab_widget.addTab(self.create_device_config(), "设备选择")
        self.tab_widget.addTab(self.create_save_config(), "保存选项")
        self.tab_widget.addTab(self.create_advanced_config(), "高级选项")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
    def create_basic_config(self) -> QWidget:
        """创建基础配置页面"""
        widget = QWidget()
        layout = QFormLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 数据路径
        self.data_root_edit = QLineEdit()
        self.data_root_edit.setText(Config.DATA_ROOT)
        self.data_root_edit.setPlaceholderText("N-BaIoT数据集根目录路径")
        
        data_root_btn = QPushButton("浏览...")
        data_root_btn.clicked.connect(self.browse_data_root)
        
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_root_edit)
        data_layout.addWidget(data_root_btn)
        
        layout.addRow(QLabel("📁 数据根目录:"), data_layout)
        
        # 输出目录
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(Config.OUTPUT_DIR)
        self.output_dir_edit.setPlaceholderText("训练结果输出目录")
        
        output_dir_btn = QPushButton("浏览...")
        output_dir_btn.clicked.connect(self.browse_output_dir)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_dir_btn)
        
        layout.addRow(QLabel("📂 输出目录:"), output_layout)
        
        # GPU设置
        self.use_gpu_check = QCheckBox("启用GPU加速")
        self.use_gpu_check.setChecked(Config.USE_GPU)
        layout.addRow(QLabel("🖥️ GPU设置:"), self.use_gpu_check)
        
        # GPU内存限制
        self.gpu_memory_spin = QSpinBox()
        self.gpu_memory_spin.setRange(0, 32768)
        self.gpu_memory_spin.setSuffix(" MB")
        self.gpu_memory_spin.setValue(Config.GPU_MEMORY_LIMIT if Config.GPU_MEMORY_LIMIT else 0)
        self.gpu_memory_spin.setSpecialValueText("无限制")
        layout.addRow(QLabel("💾 GPU内存限制:"), self.gpu_memory_spin)
        
        # 特征维度
        self.feature_dim_spin = QSpinBox()
        self.feature_dim_spin.setRange(1, 1000)
        self.feature_dim_spin.setValue(Config.FEATURE_DIM)
        layout.addRow(QLabel("📊 特征维度:"), self.feature_dim_spin)
        
        widget.setLayout(layout)
        return widget
        
    def create_model_config(self) -> QWidget:
        """创建模型架构配置页面"""
        widget = QWidget()
        layout = QFormLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 激活函数
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(['relu', 'leaky_relu', 'tanh', 'sigmoid'])
        self.activation_combo.setCurrentText(Config.ACTIVATION)
        layout.addRow(QLabel("🔥 激活函数:"), self.activation_combo)
        
        # 批量归一化
        self.use_batch_norm_check = QCheckBox("启用")
        self.use_batch_norm_check.setChecked(Config.USE_BATCH_NORM)
        layout.addRow(QLabel("📦 批量归一化:"), self.use_batch_norm_check)
        
        # Dropout率
        self.dropout_rate_spin = QDoubleSpinBox()
        self.dropout_rate_spin.setRange(0, 1)
        self.dropout_rate_spin.setSingleStep(0.05)
        self.dropout_rate_spin.setValue(Config.DROPOUT_RATE)
        layout.addRow(QLabel("🎲 Dropout率:"), self.dropout_rate_spin)
        
        # L2正则化
        self.l2_reg_spin = QDoubleSpinBox()
        self.l2_reg_spin.setRange(0, 1)
        self.l2_reg_spin.setSingleStep(0.0001)
        self.l2_reg_spin.setDecimals(6)
        self.l2_reg_spin.setValue(Config.L2_REGULARIZATION)
        layout.addRow(QLabel("📐 L2正则化:"), self.l2_reg_spin)
        
        # 编码器比例
        self.encoder_ratios_edit = QLineEdit()
        self.encoder_ratios_edit.setText(str(Config.ENCODER_RATIOS))
        self.encoder_ratios_edit.setPlaceholderText("[0.75, 0.50, 0.33, 0.25]")
        layout.addRow(QLabel("🔢 编码器维度比例:"), self.encoder_ratios_edit)
        
        # 解码器比例
        self.decoder_ratios_edit = QLineEdit()
        self.decoder_ratios_edit.setText(str(Config.DECODER_RATIOS))
        self.decoder_ratios_edit.setPlaceholderText("[0.33, 0.50, 0.75, 1.0]")
        layout.addRow(QLabel("🔢 解码器维度比例:"), self.decoder_ratios_edit)
        
        widget.setLayout(layout)
        return widget
        
    def create_training_config(self) -> QWidget:
        """创建训练参数配置页面"""
        widget = QWidget()
        layout = QFormLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 默认学习率
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-6, 1)
        self.learning_rate_spin.setSingleStep(1e-4)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setValue(Config.DEFAULT_LEARNING_RATE)
        layout.addRow(QLabel("📈 默认学习率:"), self.learning_rate_spin)
        
        # 默认批大小
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(Config.DEFAULT_BATCH_SIZE)
        layout.addRow(QLabel("📦 默认批大小:"), self.batch_size_spin)
        
        # 默认训练轮数
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(Config.DEFAULT_EPOCHS)
        layout.addRow(QLabel("🔄 默认训练轮数:"), self.epochs_spin)
        
        # 早停耐心值
        self.early_stopping_spin = QSpinBox()
        self.early_stopping_spin.setRange(1, 100)
        self.early_stopping_spin.setValue(Config.EARLY_STOPPING_PATIENCE)
        layout.addRow(QLabel("⏰ 早停耐心值:"), self.early_stopping_spin)
        
        # 学习率调整耐心值
        self.reduce_lr_spin = QSpinBox()
        self.reduce_lr_spin.setRange(1, 100)
        self.reduce_lr_spin.setValue(Config.REDUCE_LR_PATIENCE)
        layout.addRow(QLabel("📉 LR调整耐心值:"), self.reduce_lr_spin)
        
        # 学习率调整因子
        self.reduce_lr_factor_spin = QDoubleSpinBox()
        self.reduce_lr_factor_spin.setRange(0.01, 1)
        self.reduce_lr_factor_spin.setSingleStep(0.05)
        self.reduce_lr_factor_spin.setValue(Config.REDUCE_LR_FACTOR)
        layout.addRow(QLabel("📉 LR调整因子:"), self.reduce_lr_factor_spin)
        
        # 数据划分
        self.time_ordered_check = QCheckBox("按时间顺序划分数据")
        self.time_ordered_check.setChecked(Config.TIME_ORDERED)
        layout.addRow(QLabel("📊 数据划分方式:"), self.time_ordered_check)
        
        # 随机种子
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 2**31-1)
        self.random_seed_spin.setValue(Config.RANDOM_SEED)
        layout.addRow(QLabel("🎲 随机种子:"), self.random_seed_spin)
        
        # 可视化设置
        self.plot_save_check = QCheckBox("保存图表到文件")
        self.plot_save_check.setChecked(Config.PLOT_SAVE)
        layout.addRow(QLabel("📊 可视化设置:"), self.plot_save_check)
        
        widget.setLayout(layout)
        return widget
        
    def create_device_config(self) -> QWidget:
        """创建设备选择配置页面"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        info_label = QLabel("选择要训练的IoT设备（可多选）:")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info_label)
        
        self.device_list = QListWidget()
        self.device_list.setSelectionMode(QListWidget.MultiSelection)
        
        all_devices = Config.ALL_DEVICES
        for device in all_devices:
            item = QListWidgetItem(device)
            self.device_list.addItem(item)
            if device in Config.SELECTED_DEVICES:
                item.setSelected(True)
        
        layout.addWidget(self.device_list)
        
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_devices)
        
        deselect_all_btn = QPushButton("全不选")
        deselect_all_btn.clicked.connect(self.deselect_all_devices)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        
        layout.addLayout(button_layout)
        
        self.device_stats_label = QLabel(f"共 {len(all_devices)} 个设备")
        layout.addWidget(self.device_stats_label)
        
        widget.setLayout(layout)
        return widget
        
    def create_save_config(self) -> QWidget:
        """创建保存选项配置页面"""
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(10)
        
        # 文件保存选项
        file_group = QGroupBox("📁 文件保存选项")
        file_layout = QGridLayout()
        
        self.save_log_check = QCheckBox("保存训练日志")
        self.save_log_check.setChecked(Config.SAVE_LOG_FILE)
        file_layout.addWidget(self.save_log_check, 0, 0)
        
        self.save_model_check = QCheckBox("保存模型文件")
        self.save_model_check.setChecked(Config.SAVE_MODEL)
        file_layout.addWidget(self.save_model_check, 0, 1)
        
        self.save_best_model_only_check = QCheckBox("仅保存最佳模型")
        self.save_best_model_only_check.setChecked(Config.SAVE_BEST_MODEL_ONLY)
        file_layout.addWidget(self.save_best_model_only_check, 1, 0)
        
        self.save_training_history_check = QCheckBox("保存训练历史")
        self.save_training_history_check.setChecked(Config.SAVE_TRAINING_HISTORY)
        file_layout.addWidget(self.save_training_history_check, 1, 1)
        
        self.save_hyperparam_results_check = QCheckBox("保存超参数调优结果")
        self.save_hyperparam_results_check.setChecked(Config.SAVE_HYPERPARAMETER_TUNING_RESULTS)
        file_layout.addWidget(self.save_hyperparam_results_check, 2, 0)
        
        self.save_scaler_check = QCheckBox("保存数据标准化器")
        self.save_scaler_check.setChecked(Config.SAVE_SCALER)
        file_layout.addWidget(self.save_scaler_check, 2, 1)
        
        self.save_tensorboard_check = QCheckBox("保存TensorBoard日志")
        self.save_tensorboard_check.setChecked(Config.SAVE_TENSORBOARD_LOGS)
        file_layout.addWidget(self.save_tensorboard_check, 3, 0)
        
        self.plot_save_check = QCheckBox("保存可视化图表")
        self.plot_save_check.setChecked(Config.PLOT_SAVE)
        file_layout.addWidget(self.plot_save_check, 3, 1)
        
        # 连接信号：当"保存可视化图表"状态改变时，启用/禁用图表选项
        self.plot_save_check.stateChanged.connect(self.toggle_plot_options)
        
        file_group.setLayout(file_layout)
        content_layout.addWidget(file_group)
        
        # 训练曲线图表
        training_curves_group = QGroupBox("📈 训练曲线图表")
        training_curves_layout = QGridLayout()
        
        self.plot_loss_curve_check = QCheckBox("训练损失曲线")
        self.plot_loss_curve_check.setChecked(Config.PLOT_TRAINING_LOSS_CURVE)
        training_curves_layout.addWidget(self.plot_loss_curve_check, 0, 0)
        
        self.plot_mae_curve_check = QCheckBox("训练MAE曲线")
        self.plot_mae_curve_check.setChecked(Config.PLOT_TRAINING_MAE_CURVE)
        training_curves_layout.addWidget(self.plot_mae_curve_check, 0, 1)
        
        self.plot_lr_curve_check = QCheckBox("学习率变化曲线")
        self.plot_lr_curve_check.setChecked(Config.PLOT_TRAINING_LR_CURVE)
        training_curves_layout.addWidget(self.plot_lr_curve_check, 1, 0)
        
        training_curves_group.setLayout(training_curves_layout)
        content_layout.addWidget(training_curves_group)
        
        # 超参数调优图表
        hyperparam_group = QGroupBox("🔍 超参数调优图表")
        hyperparam_layout = QGridLayout()
        
        self.plot_hyperparam_heatmap_check = QCheckBox("超参数热图")
        self.plot_hyperparam_heatmap_check.setChecked(Config.PLOT_HYPERPARAM_HEATMAP)
        hyperparam_layout.addWidget(self.plot_hyperparam_heatmap_check, 0, 0)
        
        self.plot_hyperparam_contour_check = QCheckBox("超参数等高线图")
        self.plot_hyperparam_contour_check.setChecked(Config.PLOT_HYPERPARAM_CONTOUR)
        hyperparam_layout.addWidget(self.plot_hyperparam_contour_check, 0, 1)
        
        self.plot_hyperparam_3d_check = QCheckBox("超参数3D图")
        self.plot_hyperparam_3d_check.setChecked(Config.PLOT_HYPERPARAM_3D)
        hyperparam_layout.addWidget(self.plot_hyperparam_3d_check, 1, 0)
        
        hyperparam_group.setLayout(hyperparam_layout)
        content_layout.addWidget(hyperparam_group)
        
        # 损失分析图表
        loss_analysis_group = QGroupBox("📊 损失分析图表")
        loss_analysis_layout = QGridLayout()
        
        self.plot_loss_distribution_check = QCheckBox("损失分布图")
        self.plot_loss_distribution_check.setChecked(Config.PLOT_LOSS_DISTRIBUTION)
        loss_analysis_layout.addWidget(self.plot_loss_distribution_check, 0, 0)
        
        self.plot_loss_histogram_check = QCheckBox("损失直方图")
        self.plot_loss_histogram_check.setChecked(Config.PLOT_LOSS_HISTOGRAM)
        loss_analysis_layout.addWidget(self.plot_loss_histogram_check, 0, 1)
        
        self.plot_loss_boxplot_check = QCheckBox("损失箱线图")
        self.plot_loss_boxplot_check.setChecked(Config.PLOT_LOSS_BOX_PLOT)
        loss_analysis_layout.addWidget(self.plot_loss_boxplot_check, 1, 0)
        
        self.plot_loss_violin_check = QCheckBox("损失小提琴图")
        self.plot_loss_violin_check.setChecked(Config.PLOT_LOSS_VIOLIN_PLOT)
        loss_analysis_layout.addWidget(self.plot_loss_violin_check, 1, 1)
        
        loss_analysis_group.setLayout(loss_analysis_layout)
        content_layout.addWidget(loss_analysis_group)
        
        # 模型性能图表
        performance_group = QGroupBox("⚡ 模型性能图表")
        performance_layout = QGridLayout()
        
        self.plot_performance_metrics_check = QCheckBox("性能指标图")
        self.plot_performance_metrics_check.setChecked(Config.PLOT_PERFORMANCE_METRICS)
        performance_layout.addWidget(self.plot_performance_metrics_check, 0, 0)
        
        self.plot_lr_schedule_check = QCheckBox("学习率调度图")
        self.plot_lr_schedule_check.setChecked(Config.PLOT_LEARNING_RATE_SCHEDULE)
        performance_layout.addWidget(self.plot_lr_schedule_check, 0, 1)
        
        self.plot_gradient_flow_check = QCheckBox("梯度流图")
        self.plot_gradient_flow_check.setChecked(Config.PLOT_GRADIENT_FLOW)
        performance_layout.addWidget(self.plot_gradient_flow_check, 1, 0)
        
        performance_group.setLayout(performance_layout)
        content_layout.addWidget(performance_group)
        
        # 数据分析图表
        data_analysis_group = QGroupBox("🔬 数据分析图表")
        data_analysis_layout = QGridLayout()
        
        self.plot_data_distribution_check = QCheckBox("数据分布图")
        self.plot_data_distribution_check.setChecked(Config.PLOT_DATA_DISTRIBUTION)
        data_analysis_layout.addWidget(self.plot_data_distribution_check, 0, 0)
        
        self.plot_feature_corr_check = QCheckBox("特征相关性图")
        self.plot_feature_corr_check.setChecked(Config.PLOT_FEATURE_CORRELATION)
        data_analysis_layout.addWidget(self.plot_feature_corr_check, 0, 1)
        
        self.plot_pca_check = QCheckBox("PCA可视化")
        self.plot_pca_check.setChecked(Config.PLOT_PCA_VISUALIZATION)
        data_analysis_layout.addWidget(self.plot_pca_check, 1, 0)
        
        data_analysis_group.setLayout(data_analysis_layout)
        content_layout.addWidget(data_analysis_group)
        
        # 时间分析图表
        time_analysis_group = QGroupBox("⏱️ 时间分析图表")
        time_analysis_layout = QGridLayout()
        
        self.plot_training_time_check = QCheckBox("训练时间分析")
        self.plot_training_time_check.setChecked(Config.PLOT_TRAINING_TIME_ANALYSIS)
        time_analysis_layout.addWidget(self.plot_training_time_check, 0, 0)
        
        self.plot_epoch_time_check = QCheckBox("Epoch时间分布")
        self.plot_epoch_time_check.setChecked(Config.PLOT_EPOCH_TIME_DISTRIBUTION)
        time_analysis_layout.addWidget(self.plot_epoch_time_check, 0, 1)
        
        time_analysis_group.setLayout(time_analysis_layout)
        content_layout.addWidget(time_analysis_group)
        
        # 比较图表
        comparison_group = QGroupBox("🔎 比较图表")
        comparison_layout = QGridLayout()
        
        self.plot_device_comparison_check = QCheckBox("设备比较图")
        self.plot_device_comparison_check.setChecked(Config.PLOT_DEVICE_COMPARISON)
        comparison_layout.addWidget(self.plot_device_comparison_check, 0, 0)
        
        self.plot_phase_comparison_check = QCheckBox("训练阶段比较")
        self.plot_phase_comparison_check.setChecked(Config.PLOT_PHASE_COMPARISON)
        comparison_layout.addWidget(self.plot_phase_comparison_check, 0, 1)
        
        self.plot_performance_ranking_check = QCheckBox("性能排名图")
        self.plot_performance_ranking_check.setChecked(Config.PLOT_PERFORMANCE_RANKING)
        comparison_layout.addWidget(self.plot_performance_ranking_check, 1, 0)
        
        comparison_group.setLayout(comparison_layout)
        content_layout.addWidget(comparison_group)
        
        # 综合报告图表
        report_group = QGroupBox("📋 综合报告图表")
        report_layout = QGridLayout()
        
        self.plot_comprehensive_summary_check = QCheckBox("综合总结图")
        self.plot_comprehensive_summary_check.setChecked(Config.PLOT_COMPREHENSIVE_SUMMARY)
        report_layout.addWidget(self.plot_comprehensive_summary_check, 0, 0)
        
        self.plot_training_report_check = QCheckBox("训练报告")
        self.plot_training_report_check.setChecked(Config.PLOT_TRAINING_REPORT)
        report_layout.addWidget(self.plot_training_report_check, 0, 1)
        
        report_group.setLayout(report_layout)
        content_layout.addWidget(report_group)
        
        # 初始化图表选项的启用状态
        self.toggle_plot_options(self.plot_save_check.isChecked())
        
        content_layout.addStretch()
        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)
        
        main_layout.addWidget(scroll)
        widget.setLayout(main_layout)
        return widget
        
    def create_advanced_config(self) -> QWidget:
        """创建高级配置页面"""
        widget = QWidget()
        layout = QFormLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        self.lr_space_edit = QLineEdit()
        self.lr_space_edit.setText(str(Config.LEARNING_RATES))
        self.lr_space_edit.setPlaceholderText("[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]")
        layout.addRow(QLabel("📈 LR搜索空间:"), self.lr_space_edit)
        
        self.epochs_space_edit = QLineEdit()
        self.epochs_space_edit.setText(str(Config.EPOCHS_OPTIONS))
        self.epochs_space_edit.setPlaceholderText("[50, 100, 150, 200]")
        layout.addRow(QLabel("🔄 Epoch搜索空间:"), self.epochs_space_edit)
        
        self.batch_space_edit = QLineEdit()
        self.batch_space_edit.setText(str(Config.BATCH_SIZES))
        self.batch_space_edit.setPlaceholderText("[32, 64, 128]")
        layout.addRow(QLabel("📦 Batch搜索空间:"), self.batch_space_edit)
        
        self.output_activation_combo = QComboBox()
        self.output_activation_combo.addItems(['None', 'sigmoid', 'tanh', 'relu'])
        self.output_activation_combo.setCurrentText(str(Config.OUTPUT_ACTIVATION) if Config.OUTPUT_ACTIVATION else 'None')
        layout.addRow(QLabel("🎯 输出激活函数:"), self.output_activation_combo)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['adam', 'rmsprop', 'sgd'])
        self.optimizer_combo.setCurrentText(Config.OPTIMIZER)
        layout.addRow(QLabel("⚙️ 优化器:"), self.optimizer_combo)
        
        widget.setLayout(layout)
        return widget
    
    def browse_data_root(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择N-BaIoT数据集目录",
            self.data_root_edit.text()
        )
        if directory:
            self.data_root_edit.setText(directory)
            
    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择输出目录",
            self.output_dir_edit.text()
        )
        if directory:
            self.output_dir_edit.setText(directory)
    
    def toggle_plot_options(self, enabled):
        """
        切换图表选项的启用状态
        
        Args:
            enabled: 是否启用图表选项
        """
        # 收集所有图表类型的复选框
        plot_checkboxes = [
            # 训练曲线图表
            self.plot_loss_curve_check,
            self.plot_mae_curve_check,
            self.plot_lr_curve_check,
            # 超参数调优图表
            self.plot_hyperparam_heatmap_check,
            self.plot_hyperparam_contour_check,
            self.plot_hyperparam_3d_check,
            # 损失分析图表
            self.plot_loss_distribution_check,
            self.plot_loss_histogram_check,
            self.plot_loss_boxplot_check,
            self.plot_loss_violin_check,
            # 模型性能图表
            self.plot_performance_metrics_check,
            self.plot_lr_schedule_check,
            self.plot_gradient_flow_check,
            # 数据分析图表
            self.plot_data_distribution_check,
            self.plot_feature_corr_check,
            self.plot_pca_check,
            # 时间分析图表
            self.plot_training_time_check,
            self.plot_epoch_time_check,
            # 比较图表
            self.plot_device_comparison_check,
            self.plot_phase_comparison_check,
            self.plot_performance_ranking_check,
            # 综合报告图表
            self.plot_comprehensive_summary_check,
            self.plot_training_report_check
        ]
        
        # 启用或禁用所有图表选项
        for checkbox in plot_checkboxes:
            checkbox.setEnabled(enabled)
            # 如果禁用，取消勾选
            if not enabled:
                checkbox.setChecked(False)
    
    def select_all_devices(self):
        for i in range(self.device_list.count()):
            self.device_list.item(i).setSelected(True)
            
    def deselect_all_devices(self):
        for i in range(self.device_list.count()):
            self.device_list.item(i).setSelected(False)
    
    def get_config(self) -> Dict:
        def parse_list(text: str, default: List):
            try:
                return eval(text)
            except:
                return default
        
        selected_devices = []
        for i in range(self.device_list.count()):
            if self.device_list.item(i).isSelected():
                selected_devices.append(self.device_list.item(i).text())
        
        return {
            'data_root': self.data_root_edit.text(),
            'output_dir': self.output_dir_edit.text(),
            'use_gpu': self.use_gpu_check.isChecked(),
            'gpu_memory_limit': self.gpu_memory_spin.value() if self.gpu_memory_spin.value() > 0 else None,
            'feature_dim': self.feature_dim_spin.value(),
            'activation': self.activation_combo.currentText(),
            'use_batch_norm': self.use_batch_norm_check.isChecked(),
            'dropout_rate': self.dropout_rate_spin.value(),
            'l2_regularization': self.l2_reg_spin.value(),
            'encoder_ratios': parse_list(self.encoder_ratios_edit.text(), Config.ENCODER_RATIOS),
            'decoder_ratios': parse_list(self.decoder_ratios_edit.text(), Config.DECODER_RATIOS),
            'learning_rate': self.learning_rate_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'epochs': self.epochs_spin.value(),
            'early_stopping_patience': self.early_stopping_spin.value(),
            'reduce_lr_patience': self.reduce_lr_spin.value(),
            'reduce_lr_factor': self.reduce_lr_factor_spin.value(),
            'time_ordered': self.time_ordered_check.isChecked(),
            'random_seed': self.random_seed_spin.value(),
            'selected_devices': selected_devices,
            'learning_rates': parse_list(self.lr_space_edit.text(), Config.LEARNING_RATES),
            'epochs_options': parse_list(self.epochs_space_edit.text(), Config.EPOCHS_OPTIONS),
            'batch_sizes': parse_list(self.batch_space_edit.text(), Config.BATCH_SIZES),
            'output_activation': None if self.output_activation_combo.currentText() == 'None' else self.output_activation_combo.currentText(),
            'optimizer': self.optimizer_combo.currentText(),
            # 文件保存选项
            'save_log_file': self.save_log_check.isChecked(),
            'save_model': self.save_model_check.isChecked(),
            'save_best_model_only': self.save_best_model_only_check.isChecked(),
            'save_training_history': self.save_training_history_check.isChecked(),
            'save_hyperparam_results': self.save_hyperparam_results_check.isChecked(),
            'save_scaler': self.save_scaler_check.isChecked(),
            'save_tensorboard': self.save_tensorboard_check.isChecked(),
            'plot_save': self.plot_save_check.isChecked()
        }
        
        # 图表类型选项（当plot_save为False时，所有图表类型都返回False）
        plot_enabled = self.plot_save_check.isChecked()
        plot_options = {
            'plot_training_loss_curve': self.plot_loss_curve_check.isChecked() if plot_enabled else False,
            'plot_training_mae_curve': self.plot_mae_curve_check.isChecked() if plot_enabled else False,
            'plot_training_lr_curve': self.plot_lr_curve_check.isChecked() if plot_enabled else False,
            'plot_hyperparam_heatmap': self.plot_hyperparam_heatmap_check.isChecked() if plot_enabled else False,
            'plot_hyperparam_contour': self.plot_hyperparam_contour_check.isChecked() if plot_enabled else False,
            'plot_hyperparam_3d': self.plot_hyperparam_3d_check.isChecked() if plot_enabled else False,
            'plot_loss_distribution': self.plot_loss_distribution_check.isChecked() if plot_enabled else False,
            'plot_loss_histogram': self.plot_loss_histogram_check.isChecked() if plot_enabled else False,
            'plot_loss_boxplot': self.plot_loss_boxplot_check.isChecked() if plot_enabled else False,
            'plot_loss_violin': self.plot_loss_violin_check.isChecked() if plot_enabled else False,
            'plot_performance_metrics': self.plot_performance_metrics_check.isChecked() if plot_enabled else False,
            'plot_lr_schedule': self.plot_lr_schedule_check.isChecked() if plot_enabled else False,
            'plot_gradient_flow': self.plot_gradient_flow_check.isChecked() if plot_enabled else False,
            'plot_data_distribution': self.plot_data_distribution_check.isChecked() if plot_enabled else False,
            'plot_feature_correlation': self.plot_feature_corr_check.isChecked() if plot_enabled else False,
            'plot_pca_visualization': self.plot_pca_check.isChecked() if plot_enabled else False,
            'plot_training_time_analysis': self.plot_training_time_check.isChecked() if plot_enabled else False,
            'plot_epoch_time_distribution': self.plot_epoch_time_check.isChecked() if plot_enabled else False,
            'plot_device_comparison': self.plot_device_comparison_check.isChecked() if plot_enabled else False,
            'plot_phase_comparison': self.plot_phase_comparison_check.isChecked() if plot_enabled else False,
            'plot_performance_ranking': self.plot_performance_ranking_check.isChecked() if plot_enabled else False,
            'plot_comprehensive_summary': self.plot_comprehensive_summary_check.isChecked() if plot_enabled else False,
            'plot_training_report': self.plot_training_report_check.isChecked() if plot_enabled else False
        }
        
        # 合并两个字典
        config_dict.update(plot_options)
        
        return config_dict
    
    def load_config(self, config: Dict):
        if 'data_root' in config:
            self.data_root_edit.setText(config['data_root'])
        if 'output_dir' in config:
            self.output_dir_edit.setText(config['output_dir'])
        if 'use_gpu' in config:
            self.use_gpu_check.setChecked(config['use_gpu'])
        if 'feature_dim' in config:
            self.feature_dim_spin.setValue(config['feature_dim'])
        if 'selected_devices' in config:
            for i in range(self.device_list.count()):
                item = self.device_list.item(i)
                item.setSelected(item.text() in config['selected_devices'])


# ============================================================================
# 训练控制面板
# ============================================================================
class TrainingControlPanel(QWidget):
    """训练控制面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.training_worker = None
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # 开始按钮
        self.start_btn = QPushButton("▶ 开始训练")
        self.start_btn.setMinimumSize(120, 50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_training)
        layout.addWidget(self.start_btn)
        
        # 暂停按钮
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setMinimumSize(100, 50)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 14px;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.pause_btn.clicked.connect(self.pause_training)
        layout.addWidget(self.pause_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setMinimumSize(100, 50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_training)
        layout.addWidget(self.stop_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumSize(200, 30)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setMinimumSize(150, 30)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.signals = TrainingSignals()
        
    def start_training(self):
        """开始训练"""
        if self.main_window:
            config = self.main_window.config_panel.get_config()
            
            if not config['selected_devices']:
                QMessageBox.warning(self, "警告", "请至少选择一个要训练的设备！")
                return
            
            # 如果已有训练 worker，先断开所有旧的信号连接
            if self.training_worker is not None:
                self._disconnect_signals()
            
            self.training_worker = TrainingWorker(config, self.signals)
            
            # 连接信号（每次都是新的连接）
            self.signals.started.connect(self.on_training_started)
            self.signals.progress.connect(self.on_progress_update)
            self.signals.epoch_completed.connect(self.on_epoch_completed)
            self.signals.phase_completed.connect(self.on_phase_completed)
            self.signals.device_completed.connect(self.on_device_completed)
            self.signals.finished.connect(self.on_training_finished)
            self.signals.error.connect(self.on_training_error)
            self.signals.log.connect(self.on_log_received)
            self.signals.status_update.connect(self.on_status_update)
            
            self.training_worker.start()
            
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.pause_btn.setText("⏸ 暂停")
            self.status_label.setText("训练中...")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    padding: 5px;
                    font-weight: bold;
                }
            """)
            
            if self.main_window.chart:
                self.main_window.chart.clear_chart()
    
    def _disconnect_signals(self):
        """断开所有信号连接，防止重复连接导致的重复日志"""
        try:
            self.signals.started.disconnect(self.on_training_started)
            self.signals.progress.disconnect(self.on_progress_update)
            self.signals.epoch_completed.disconnect(self.on_epoch_completed)
            self.signals.phase_completed.disconnect(self.on_phase_completed)
            self.signals.device_completed.disconnect(self.on_device_completed)
            self.signals.finished.disconnect(self.on_training_finished)
            self.signals.error.disconnect(self.on_training_error)
            self.signals.log.disconnect(self.on_log_received)
            self.signals.status_update.disconnect(self.on_status_update)
        except (TypeError, RuntimeError):
            # 如果某些信号没有被连接，忽略错误
            pass
    
    def pause_training(self):
        """暂停训练"""
        if self.training_worker and self.training_worker.isRunning():
            if self.pause_btn.text() == "⏸ 暂停":
                self.training_worker.pause()
                self.pause_btn.setText("▶ 继续")
                self.status_label.setText("已暂停")
                self.status_label.setStyleSheet("""
                    QLabel {
                        background-color: #FF9800;
                        color: white;
                        border-radius: 5px;
                        padding: 5px;
                        font-weight: bold;
                    }
                """)
            else:
                self.training_worker.resume()
                self.pause_btn.setText("⏸ 暂停")
                self.status_label.setText("训练中...")
                self.status_label.setStyleSheet("""
                    QLabel {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 5px;
                        padding: 5px;
                        font-weight: bold;
                    }
                """)
    
    def stop_training(self):
        """停止训练"""
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认", "确定要停止训练吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_worker.stop()
                self.status_label.setText("正在停止...")
                self.status_label.setStyleSheet("""
                    QLabel {
                        background-color: #f44336;
                        color: white;
                        border-radius: 5px;
                        padding: 5px;
                        font-weight: bold;
                    }
                """)
    
    @pyqtSlot(str)
    def on_training_started(self, device_name: str):
        self.main_window.log_widget.append(f"\n{'='*60}")
        self.main_window.log_widget.append(f"🚀 开始训练设备: {device_name}")
        self.main_window.log_widget.append(f"{'='*60}\n")
        
        if self.main_window.chart:
            self.main_window.chart.clear_chart()
    
    @pyqtSlot(dict)
    def on_progress_update(self, progress: dict):
        self.progress_bar.setValue(int(progress.get('progress', 0)))
    
    @pyqtSlot(dict)
    def on_epoch_completed(self, data: dict):
        epoch = data.get('epoch', 0)
        train_loss = data.get('train_loss', 0)
        val_loss = data.get('val_loss', 0)
        phase = data.get('phase', '训练')
        total_epochs = data.get('total_epochs', 100)
        
        if self.main_window.chart:
            self.main_window.chart.update_chart(epoch, train_loss, val_loss, phase, total_epochs)
        
        progress = (epoch / total_epochs) * 100
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(f"Epoch {epoch}/{total_epochs}")
        
        if self.main_window:
            self.main_window.epoch_label.setText(f"Epoch: {epoch}/{total_epochs}")
            self.main_window.loss_label.setText(f"训练损失: {train_loss:.6f}")
            if val_loss < float('inf'):
                self.main_window.best_loss_label.setText(f"验证损失: {val_loss:.6f}")
            self.main_window.phase_label.setText(f"阶段: {phase}")
    
    @pyqtSlot(dict)
    def on_phase_completed(self, data: dict):
        phase = data.get('phase', '')
        loss = data.get('loss', 0)
        self.main_window.log_widget.append(f"✅ {phase}完成，最佳损失: {loss:.6f}")
    
    @pyqtSlot(dict)
    def on_device_completed(self, data: dict):
        device = data.get('device', '')
        result = data.get('result')
        progress = data.get('progress', 0)
        
        self.main_window.log_widget.append(f"\n{'#'*60}")
        self.main_window.log_widget.append(f"✅ 设备 {device} 训练完成")
        
        if result:
            self.main_window.log_widget.append(f"   最佳验证损失: {result.get('best_val_loss', 0):.6f}")
            self.main_window.log_widget.append(f"   最终训练损失: {result.get('final_train_loss', 0):.6f}")
            self.main_window.log_widget.append(f"   训练时间: {result.get('training_time', 0):.2f}秒")
        
        self.main_window.log_widget.append(f"{'#'*60}\n")
        
        if self.main_window:
            self.main_window.update_overall_progress(progress)
    
    @pyqtSlot(dict)
    def on_training_finished(self, data: dict):
        results = data.get('results', [])
        total_time = data.get('total_time', 0)
        
        self.main_window.log_widget.append(f"\n{'='*60}")
        self.main_window.log_widget.append(f"🎉 全部训练完成！")
        self.main_window.log_widget.append(f"{'='*60}")
        self.main_window.log_widget.append(f"总设备数: {len(results)}")
        self.main_window.log_widget.append(f"总训练时间: {total_time:.2f}秒")
        self.main_window.log_widget.append(f"输出目录: {Config.OUTPUT_DIR}")
        
        # 生成设备比较图表
        if len(results) > 1 and Config.PLOT_SAVE:
            try:
                from visualizer import ScientificVisualizer
                visualizer = ScientificVisualizer(Config)
                visualizer.plot_device_comparison(results)
                visualizer.plot_performance_ranking(results)
                self.main_window.log_widget.append(f"✅ 设备比较图表已生成")
            except Exception as e:
                self.main_window.log_widget.append(f"⚠️ 生成比较图表时出错: {str(e)}")
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("完成")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        
        QMessageBox.information(
            self, "训练完成", 
            f"训练完成！\n\n"
            f"总设备数: {len(results)}\n"
            f"总训练时间: {total_time:.2f}秒\n"
            f"结果已保存到: {Config.OUTPUT_DIR}"
        )
    
    @pyqtSlot(str)
    def on_training_error(self, error: str):
        self.main_window.log_widget.append(f"\n❌ 错误: {error}")
        self.main_window.log_widget.append("\n" + "="*60)
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("错误")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        
        QMessageBox.critical(self, "训练错误", error)
    
    @pyqtSlot(str)
    def on_log_received(self, log: str):
        self.main_window.log_widget.append(log)
    
    @pyqtSlot(str)
    def on_status_update(self, status: str):
        self.status_label.setText(status)


# ============================================================================
# 入侵检测与评估信号类
# ============================================================================
class IntrusionDetectionSignals(QObject):
    """入侵检测与评估线程信号类"""
    started = pyqtSignal(str)           # 评估开始信号
    progress = pyqtSignal(dict)         # 进度更新信号
    data_updated = pyqtSignal(dict)     # 数据更新信号
    completed = pyqtSignal(dict)        # 评估完成信号
    error = pyqtSignal(str)             # 错误信号
    log = pyqtSignal(str)               # 日志信号
    status_update = pyqtSignal(str)     # 状态更新信号
    file_generated = pyqtSignal(str)    # 文件生成信号
    save_completed = pyqtSignal(str)    # 保存完成信号

# ============================================================================
# 入侵检测与评估工作线程
# ============================================================================
class IntrusionDetectionWorker(QThread):
    """入侵检测与评估工作线程 - 在后台执行评估任务"""
    
    def __init__(self, config: Dict, signals: IntrusionDetectionSignals):
        super().__init__()
        self.config = config
        self.signals = signals
        self.is_running = False
        self.should_stop = False
    
    def run(self):
        """执行评估"""
        self.is_running = True
        self.should_stop = False
        
        try:
            # 导入必要的模块
            from anomaly_detector import AnomalyDetector
            from data_integrator import DStstIntegrator
            import numpy as np
            import os
            import time
            
            # 获取配置参数
            device_name = self.config.get('device_name', 'Danmini_Doorbell')
            dstst_file = self.config.get('dstst_file', '')
            model_file = self.config.get('model_file', '')
            save_path = self.config.get('save_path', os.path.join(Config.OUTPUT_DIR, 'intrusion_detection'))
            save_data = self.config.get('save_data', True)
            save_images = self.config.get('save_images', True)
            
            # 验证文件
            if not dstst_file:
                self.signals.error.emit("请选择DStst文件")
                self.is_running = False
                return
            
            if not model_file:
                self.signals.error.emit("请选择模型文件")
                self.is_running = False
                return
            
            if not os.path.exists(dstst_file):
                # 自动生成DStst文件
                self.signals.log.emit("未检测到DStst文件，正在生成...")
                self.signals.status_update.emit("正在生成DStst文件")
                
                # 生成DStst文件
                integrator = DStstIntegrator(Config)
                generated_file = integrator.create_dstst_dataset(device_name)
                
                if not generated_file:
                    self.signals.error.emit("生成DStst文件失败")
                    self.is_running = False
                    return
                
                dstst_file = generated_file
                self.signals.file_generated.emit(f"已生成DStst文件: {dstst_file}")
                self.signals.log.emit(f"已生成DStst文件: {dstst_file}")
            
            # 加载DStst文件
            self.signals.log.emit("正在加载DStst文件...")
            self.signals.status_update.emit("正在加载DStst文件")
            
            # 这里需要根据实际的DStst文件格式进行加载
            # 假设DStst文件是numpy格式，包含数据和标签
            try:
                dstst_data = np.load(dstst_file, allow_pickle=True).item()
                X_test = dstst_data['X']
                y_test = dstst_data['y']
            except Exception as e:
                self.signals.error.emit(f"加载DStst文件失败: {str(e)}")
                self.is_running = False
                return
            
            # 加载模型
            self.signals.log.emit("正在加载模型文件...")
            self.signals.status_update.emit("正在加载模型文件")
            
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_file)
            except Exception as e:
                self.signals.error.emit(f"加载模型文件失败: {str(e)}")
                self.is_running = False
                return
            
            # 初始化异常检测器
            detector = AnomalyDetector(Config)
            
            # 计算异常阈值和滑动窗口大小
            self.signals.log.emit("正在计算异常阈值和滑动窗口大小...")
            self.signals.status_update.emit("正在计算异常阈值和滑动窗口大小")
            
            # 这里需要根据实际的实现进行计算
            # 假设我们已经有了这些值
            threshold = 0.1  # 示例值
            window_size = 5   # 示例值
            
            self.signals.log.emit(f"异常阈值: {threshold}, 滑动窗口大小: {window_size}")
            
            # 开始评估
            self.signals.log.emit("开始评估入侵检测性能...")
            self.signals.status_update.emit("正在评估入侵检测性能")
            
            total_samples = len(X_test)
            batch_size = 32
            results = []
            
            for i in range(0, total_samples, batch_size):
                if self.should_stop:
                    break
                
                end_idx = min(i + batch_size, total_samples)
                batch_X = X_test[i:end_idx]
                batch_y = y_test[i:end_idx]
                
                # 预测
                predictions = model.predict(batch_X)
                mse = np.mean(np.power(batch_X - predictions, 2), axis=1)
                
                # 应用阈值和滑动窗口
                # 这里需要根据实际的实现进行计算
                # 示例：简单的阈值判断
                batch_predictions = (mse > threshold).astype(int)
                
                # 记录结果
                for j in range(len(batch_predictions)):
                    results.append({
                        'true_label': int(batch_y[j]),
                        'predicted_label': int(batch_predictions[j]),
                        'mse': float(mse[j])
                    })
                
                # 计算进度
                progress = (end_idx / total_samples) * 100
                self.signals.progress.emit({'progress': progress})
                
                # 发送数据更新信号
                if i % (batch_size * 10) == 0:
                    # 计算当前的性能指标
                    current_results = results[-1000:] if len(results) > 1000 else results
                    if current_results:
                        tp = sum(1 for r in current_results if r['true_label'] == 1 and r['predicted_label'] == 1)
                        tn = sum(1 for r in current_results if r['true_label'] == 0 and r['predicted_label'] == 0)
                        fp = sum(1 for r in current_results if r['true_label'] == 0 and r['predicted_label'] == 1)
                        fn = sum(1 for r in current_results if r['true_label'] == 1 and r['predicted_label'] == 0)
                        
                        accuracy = (tp + tn) / len(current_results) if current_results else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        
                        self.signals.data_updated.emit({
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'fpr': fpr,
                            'progress': progress,
                            'current_sample': end_idx,
                            'total_samples': total_samples
                        })
                
                # 模拟处理时间
                time.sleep(0.1)
            
            # 计算最终的性能指标
            tp = sum(1 for r in results if r['true_label'] == 1 and r['predicted_label'] == 1)
            tn = sum(1 for r in results if r['true_label'] == 0 and r['predicted_label'] == 0)
            fp = sum(1 for r in results if r['true_label'] == 0 and r['predicted_label'] == 1)
            fn = sum(1 for r in results if r['true_label'] == 1 and r['predicted_label'] == 0)
            
            accuracy = (tp + tn) / len(results) if results else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # 保存结果
            if save_data or save_images:
                os.makedirs(save_path, exist_ok=True)
                
                # 保存数据
                if save_data:
                    results_file = os.path.join(save_path, f'{device_name}_detection_results.json')
                    import json
                    
                    # 转换numpy类型
                    def convert_numpy_types(obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        else:
                            return obj
                    
                    # 保存结果
                    with open(results_file, 'w') as f:
                        json.dump(results, f, default=convert_numpy_types)
                    
                    self.signals.log.emit(f"已保存检测结果: {results_file}")
                
                # 保存图片
                if save_images:
                    # 这里需要根据实际的实现进行保存
                    # 示例：保存性能指标图
                    self.signals.log.emit("正在保存性能指标图...")
                    # 这里需要实现图表保存逻辑
            
            # 发送完成信号
            self.signals.completed.emit({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr,
                'total_samples': total_samples,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            })
            
        except Exception as e:
            import traceback
            error_msg = f"评估错误: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        finally:
            self.is_running = False
    
    def stop(self):
        """停止评估"""
        self.should_stop = True
        self.signals.status_update.emit("正在停止...")

# ============================================================================
# 入侵检测与评估面板
# ============================================================================
class IntrusionDetectionPanel(QWidget):
    """入侵检测与评估面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        
        # 添加各个配置页面
        self.tab_widget.addTab(self.create_file_selection_tab(), "文件选择")
        self.tab_widget.addTab(self.create_evaluation_tab(), "评估过程")
        self.tab_widget.addTab(self.create_results_tab(), "评估结果")
        self.tab_widget.addTab(self.create_save_tab(), "保存选项")
        
        layout.addWidget(self.tab_widget)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.start_evaluation_btn = QPushButton("▶ 开始评估")
        self.start_evaluation_btn.setMinimumSize(120, 50)
        self.start_evaluation_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_evaluation_btn.clicked.connect(self.start_evaluation)
        control_layout.addWidget(self.start_evaluation_btn)
        
        self.stop_evaluation_btn = QPushButton("⏹ 停止")
        self.stop_evaluation_btn.setMinimumSize(100, 50)
        self.stop_evaluation_btn.setEnabled(False)
        self.stop_evaluation_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_evaluation_btn.clicked.connect(self.stop_evaluation)
        control_layout.addWidget(self.stop_evaluation_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumSize(200, 30)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("%p%")
        control_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setMinimumSize(150, 30)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 日志输出
        log_label = QLabel("📋 评估日志")
        log_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(log_label)
        
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Consolas", 10))
        self.log_widget.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.log_widget)
        
        self.setLayout(layout)
        
        self.signals = IntrusionDetectionSignals()
        self.evaluation_worker = None
    
    def create_file_selection_tab(self) -> QWidget:
        """创建文件选择页面"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 设备选择
        device_group = QGroupBox("📱 设备选择")
        device_layout = QFormLayout()
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(Config.ALL_DEVICES)
        self.device_combo.setCurrentText('Danmini_Doorbell')
        device_layout.addRow(QLabel("设备名称:"), self.device_combo)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # DStst文件选择
        dstst_group = QGroupBox("📁 DStst文件选择")
        dstst_layout = QFormLayout()
        
        self.dstst_file_edit = QLineEdit()
        self.dstst_file_edit.setPlaceholderText("选择包含标签信息的DStst文件")
        
        dstst_browse_btn = QPushButton("浏览...")
        dstst_browse_btn.clicked.connect(self.browse_dstst_file)
        
        dstst_file_layout = QHBoxLayout()
        dstst_file_layout.addWidget(self.dstst_file_edit)
        dstst_file_layout.addWidget(dstst_browse_btn)
        
        dstst_layout.addRow(QLabel("DStst文件路径:"), dstst_file_layout)
        
        self.dstst_status_label = QLabel("未选择文件")
        self.dstst_status_label.setStyleSheet("color: #666666;")
        dstst_layout.addRow(QLabel("文件状态:"), self.dstst_status_label)
        
        dstst_group.setLayout(dstst_layout)
        layout.addWidget(dstst_group)
        
        # 模型文件选择
        model_group = QGroupBox("🤖 模型文件选择")
        model_layout = QFormLayout()
        
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setPlaceholderText("选择预训练的自编码模型文件")
        
        model_browse_btn = QPushButton("浏览...")
        model_browse_btn.clicked.connect(self.browse_model_file)
        
        model_file_layout = QHBoxLayout()
        model_file_layout.addWidget(self.model_file_edit)
        model_file_layout.addWidget(model_browse_btn)
        
        model_layout.addRow(QLabel("模型文件路径:"), model_file_layout)
        
        self.model_status_label = QLabel("未选择文件")
        self.model_status_label.setStyleSheet("color: #666666;")
        model_layout.addRow(QLabel("文件状态:"), self.model_status_label)
        
        self.model_info_label = QLabel("模型信息: 无")
        self.model_info_label.setStyleSheet("color: #666666;")
        model_layout.addRow(QLabel("模型信息:"), self.model_info_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_evaluation_tab(self) -> QWidget:
        """创建评估过程页面"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 实时数据展示
        data_group = QGroupBox("📊 实时评估数据")
        data_layout = QGridLayout()
        
        self.accuracy_label = QLabel("准确率: 0.00%")
        self.accuracy_label.setStyleSheet("font-weight: bold;")
        self.precision_label = QLabel("精确率: 0.00%")
        self.precision_label.setStyleSheet("font-weight: bold;")
        self.recall_label = QLabel("召回率: 0.00%")
        self.recall_label.setStyleSheet("font-weight: bold;")
        self.f1_label = QLabel("F1分数: 0.00%")
        self.f1_label.setStyleSheet("font-weight: bold;")
        self.fpr_label = QLabel("误报率: 0.00%")
        self.fpr_label.setStyleSheet("font-weight: bold;")
        self.sample_label = QLabel("处理样本: 0/0")
        self.sample_label.setStyleSheet("font-weight: bold;")
        
        data_layout.addWidget(self.accuracy_label, 0, 0)
        data_layout.addWidget(self.precision_label, 0, 1)
        data_layout.addWidget(self.recall_label, 1, 0)
        data_layout.addWidget(self.f1_label, 1, 1)
        data_layout.addWidget(self.fpr_label, 2, 0)
        data_layout.addWidget(self.sample_label, 2, 1)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 数据可视化图表
        chart_group = QGroupBox("📈 评估过程图表")
        chart_layout = QVBoxLayout()
        
        self.evaluation_chart = RealTimeChart()
        self.evaluation_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(self.evaluation_chart)
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_results_tab(self) -> QWidget:
        """创建评估结果页面"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 评估结果
        results_group = QGroupBox("📋 评估结果")
        results_layout = QGridLayout()
        
        self.final_accuracy_label = QLabel("最终准确率: 0.00%")
        self.final_accuracy_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.final_precision_label = QLabel("最终精确率: 0.00%")
        self.final_precision_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.final_recall_label = QLabel("最终召回率: 0.00%")
        self.final_recall_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.final_f1_label = QLabel("最终F1分数: 0.00%")
        self.final_f1_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.final_fpr_label = QLabel("最终误报率: 0.00%")
        self.final_fpr_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.confusion_matrix_label = QLabel("混淆矩阵: 无")
        self.confusion_matrix_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        results_layout.addWidget(self.final_accuracy_label, 0, 0)
        results_layout.addWidget(self.final_precision_label, 0, 1)
        results_layout.addWidget(self.final_recall_label, 1, 0)
        results_layout.addWidget(self.final_f1_label, 1, 1)
        results_layout.addWidget(self.final_fpr_label, 2, 0)
        results_layout.addWidget(self.confusion_matrix_label, 2, 1)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_save_tab(self) -> QWidget:
        """创建保存选项页面"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 保存路径
        save_group = QGroupBox("💾 保存选项")
        save_layout = QFormLayout()
        
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setText(os.path.join(Config.OUTPUT_DIR, 'intrusion_detection'))
        
        save_browse_btn = QPushButton("浏览...")
        save_browse_btn.clicked.connect(self.browse_save_path)
        
        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(self.save_path_edit)
        save_path_layout.addWidget(save_browse_btn)
        
        save_layout.addRow(QLabel("保存路径:"), save_path_layout)
        
        # 保存选项
        self.save_data_check = QCheckBox("保存评估数据")
        self.save_data_check.setChecked(True)
        save_layout.addRow(QLabel("保存数据:"), self.save_data_check)
        
        self.save_images_check = QCheckBox("保存评估图表")
        self.save_images_check.setChecked(True)
        save_layout.addRow(QLabel("保存图表:"), self.save_images_check)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def browse_dstst_file(self):
        """浏览DStst文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择DStst文件",
            self.dstst_file_edit.text(),
            "Numpy文件 (*.npy);;All Files (*)"
        )
        
        if file_path:
            self.dstst_file_edit.setText(file_path)
            # 验证文件
            self.validate_dstst_file(file_path)
    
    def browse_model_file(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件",
            self.model_file_edit.text(),
            "模型文件 (*.h5 *.hdf5 *.keras);;All Files (*)"
        )
        
        if file_path:
            self.model_file_edit.setText(file_path)
            # 验证文件
            self.validate_model_file(file_path)
    
    def browse_save_path(self):
        """浏览保存路径"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择保存目录",
            self.save_path_edit.text()
        )
        
        if directory:
            self.save_path_edit.setText(directory)
    
    def validate_dstst_file(self, file_path):
        """验证DStst文件"""
        try:
            import numpy as np
            # 尝试加载文件
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, dict) and 'X' in data and 'y' in data:
                self.dstst_status_label.setText("文件有效")
                self.dstst_status_label.setStyleSheet("color: #4CAF50;")
            else:
                self.dstst_status_label.setText("文件格式不正确")
                self.dstst_status_label.setStyleSheet("color: #f44336;")
        except Exception as e:
            self.dstst_status_label.setText(f"文件无效: {str(e)}")
            self.dstst_status_label.setStyleSheet("color: #f44336;")
    
    def validate_model_file(self, file_path):
        """验证模型文件"""
        try:
            import tensorflow as tf
            # 尝试加载模型
            model = tf.keras.models.load_model(file_path)
            # 显示模型基本信息
            self.model_status_label.setText("模型有效")
            self.model_status_label.setStyleSheet("color: #4CAF50;")
            # 这里可以添加更多模型信息的提取和显示
            self.model_info_label.setText(f"模型信息: 输入维度={model.input_shape[1]}")
        except Exception as e:
            self.model_status_label.setText(f"模型无效: {str(e)}")
            self.model_status_label.setStyleSheet("color: #f44336;")
            self.model_info_label.setText("模型信息: 无")
    
    def start_evaluation(self):
        """开始评估"""
        # 获取配置
        config = {
            'device_name': self.device_combo.currentText(),
            'dstst_file': self.dstst_file_edit.text(),
            'model_file': self.model_file_edit.text(),
            'save_path': self.save_path_edit.text(),
            'save_data': self.save_data_check.isChecked(),
            'save_images': self.save_images_check.isChecked()
        }
        
        # 验证配置
        if not config['model_file']:
            QMessageBox.warning(self, "警告", "请选择模型文件！")
            return
        
        # 如果已有评估 worker，先断开所有旧的信号连接
        if self.evaluation_worker is not None:
            self._disconnect_signals()
        
        # 初始化工作线程
        self.evaluation_worker = IntrusionDetectionWorker(config, self.signals)
        
        # 连接信号
        self.signals.started.connect(self.on_evaluation_started)
        self.signals.progress.connect(self.on_progress_update)
        self.signals.data_updated.connect(self.on_data_updated)
        self.signals.completed.connect(self.on_evaluation_completed)
        self.signals.error.connect(self.on_evaluation_error)
        self.signals.log.connect(self.on_log_received)
        self.signals.status_update.connect(self.on_status_update)
        self.signals.file_generated.connect(self.on_file_generated)
        self.signals.save_completed.connect(self.on_save_completed)
        
        # 启动评估
        self.evaluation_worker.start()
        
        # 更新UI状态
        self.start_evaluation_btn.setEnabled(False)
        self.stop_evaluation_btn.setEnabled(True)
        self.status_label.setText("评估中...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        
        # 清空图表
        self.evaluation_chart.clear_chart()
    
    def stop_evaluation(self):
        """停止评估"""
        if self.evaluation_worker and self.evaluation_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认", "确定要停止评估吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.evaluation_worker.stop()
    
    def _disconnect_signals(self):
        """断开所有信号连接"""
        try:
            self.signals.started.disconnect(self.on_evaluation_started)
            self.signals.progress.disconnect(self.on_progress_update)
            self.signals.data_updated.disconnect(self.on_data_updated)
            self.signals.completed.disconnect(self.on_evaluation_completed)
            self.signals.error.disconnect(self.on_evaluation_error)
            self.signals.log.disconnect(self.on_log_received)
            self.signals.status_update.disconnect(self.on_status_update)
            self.signals.file_generated.disconnect(self.on_file_generated)
            self.signals.save_completed.disconnect(self.on_save_completed)
        except (TypeError, RuntimeError):
            # 如果某些信号没有被连接，忽略错误
            pass
    
    @pyqtSlot(str)
    def on_evaluation_started(self, device_name: str):
        self.log_widget.append(f"\n{'='*60}")
        self.log_widget.append(f"🚀 开始评估设备: {device_name}")
        self.log_widget.append(f"{'='*60}\n")
    
    @pyqtSlot(dict)
    def on_progress_update(self, progress: dict):
        self.progress_bar.setValue(int(progress.get('progress', 0)))
    
    @pyqtSlot(dict)
    def on_data_updated(self, data: dict):
        # 更新实时数据标签
        accuracy = data.get('accuracy', 0)
        precision = data.get('precision', 0)
        recall = data.get('recall', 0)
        f1 = data.get('f1', 0)
        fpr = data.get('fpr', 0)
        current_sample = data.get('current_sample', 0)
        total_samples = data.get('total_samples', 0)
        
        self.accuracy_label.setText(f"准确率: {accuracy:.2%}")
        self.precision_label.setText(f"精确率: {precision:.2%}")
        self.recall_label.setText(f"召回率: {recall:.2%}")
        self.f1_label.setText(f"F1分数: {f1:.2%}")
        self.fpr_label.setText(f"误报率: {fpr:.2%}")
        self.sample_label.setText(f"处理样本: {current_sample}/{total_samples}")
        
        # 更新图表
        # 这里可以根据需要更新图表数据
    
    @pyqtSlot(dict)
    def on_evaluation_completed(self, results: dict):
        # 更新最终结果标签
        accuracy = results.get('accuracy', 0)
        precision = results.get('precision', 0)
        recall = results.get('recall', 0)
        f1 = results.get('f1', 0)
        fpr = results.get('fpr', 0)
        total_samples = results.get('total_samples', 0)
        tp = results.get('tp', 0)
        tn = results.get('tn', 0)
        fp = results.get('fp', 0)
        fn = results.get('fn', 0)
        
        self.final_accuracy_label.setText(f"最终准确率: {accuracy:.2%}")
        self.final_precision_label.setText(f"最终精确率: {precision:.2%}")
        self.final_recall_label.setText(f"最终召回率: {recall:.2%}")
        self.final_f1_label.setText(f"最终F1分数: {f1:.2%}")
        self.final_fpr_label.setText(f"最终误报率: {fpr:.2%}")
        self.confusion_matrix_label.setText(f"混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # 更新UI状态
        self.start_evaluation_btn.setEnabled(True)
        self.stop_evaluation_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("评估完成")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        
        # 显示完成消息
        QMessageBox.information(
            self, "评估完成",
            f"评估完成！\n\n"
            f"准确率: {accuracy:.2%}\n"
            f"精确率: {precision:.2%}\n"
            f"召回率: {recall:.2%}\n"
            f"F1分数: {f1:.2%}\n"
            f"误报率: {fpr:.2%}\n"
            f"处理样本数: {total_samples}"
        )
    
    @pyqtSlot(str)
    def on_evaluation_error(self, error: str):
        self.log_widget.append(f"\n❌ 错误: {error}")
        self.log_widget.append("\n" + "="*60)
        
        # 更新UI状态
        self.start_evaluation_btn.setEnabled(True)
        self.stop_evaluation_btn.setEnabled(False)
        self.status_label.setText("错误")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        
        # 显示错误消息
        QMessageBox.critical(self, "评估错误", error)
    
    @pyqtSlot(str)
    def on_log_received(self, log: str):
        self.log_widget.append(log)
    
    @pyqtSlot(str)
    def on_status_update(self, status: str):
        self.status_label.setText(status)
    
    @pyqtSlot(str)
    def on_file_generated(self, message: str):
        self.log_widget.append(f"\n✅ {message}")
        # 更新DStst文件路径
        import os
        file_path = message.split(': ')[1].strip()
        self.dstst_file_edit.setText(file_path)
        self.validate_dstst_file(file_path)
    
    @pyqtSlot(str)
    def on_save_completed(self, message: str):
        self.log_widget.append(f"\n✅ {message}")


# ============================================================================
# 主窗口
# ============================================================================
class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("N-BaIoT 自编码器训练系统 - GUI版 (最终修复)")
        self.setMinimumSize(1400, 900)
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()
        
    def setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        
        # 添加训练选项卡
        training_tab = QWidget()
        training_layout = QVBoxLayout()
        training_layout.setContentsMargins(5, 5, 5, 5)
        training_layout.setSpacing(5)
        
        splitter = QSplitter(Qt.Vertical)
        
        top_splitter = QSplitter(Qt.Horizontal)
        
        self.config_panel = ConfigPanel(self)
        self.config_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        top_splitter.addWidget(self.config_panel)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)
        
        chart_label = QLabel("📈 Real-time Training Loss Curve (Max 200 epochs display)")
        chart_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(chart_label)
        
        self.chart = RealTimeChart()
        self.chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.chart)
        
        progress_widget = QWidget()
        progress_layout = QGridLayout()
        
        self.epoch_label = QLabel("Epoch: 0/0")
        self.loss_label = QLabel("训练损失: -")
        self.best_loss_label = QLabel("验证损失: -")
        self.phase_label = QLabel("阶段: -")
        
        for label in [self.epoch_label, self.loss_label, self.best_loss_label, self.phase_label]:
            label.setStyleSheet("""
                QLabel {
                    background-color: #f5f5f5;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)
            progress_layout.addWidget(label)
        
        progress_widget.setLayout(progress_layout)
        right_layout.addWidget(progress_widget)
        
        right_widget.setLayout(right_layout)
        top_splitter.addWidget(right_widget)
        
        top_splitter.setSizes([500, 700])
        splitter.addWidget(top_splitter)
        
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        log_widget = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(5, 5, 5, 5)
        log_layout.setSpacing(5)
        
        log_label = QLabel("📋 训练日志")
        log_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        log_layout.addWidget(log_label)
        
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Consolas", 10))
        self.log_widget.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333;
                border-radius: 5px;
            }
        """)
        log_layout.addWidget(self.log_widget)
        
        log_widget.setLayout(log_layout)
        bottom_splitter.addWidget(log_widget)
        
        self.control_panel = TrainingControlPanel(self)
        self.control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        bottom_splitter.addWidget(self.control_panel)
        
        bottom_splitter.setSizes([700, 300])
        splitter.addWidget(bottom_splitter)
        
        splitter.setSizes([600, 300])
        
        training_layout.addWidget(splitter)
        training_tab.setLayout(training_layout)
        
        # 添加入侵检测与评估选项卡
        self.intrusion_detection_panel = IntrusionDetectionPanel(self)
        
        # 添加选项卡到主窗口
        self.tab_widget.addTab(training_tab, "模型训练")
        self.tab_widget.addTab(self.intrusion_detection_panel, "入侵检测与评估")
        
        main_layout.addWidget(self.tab_widget)
        
    def setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("文件")
        
        save_config_action = QAction("保存配置", self)
        save_config_action.triggered.connect(self.save_config)
        file_menu.addAction(save_config_action)
        
        load_config_action = QAction("加载配置", self)
        load_config_action.triggered.connect(self.load_config)
        file_menu.addAction(load_config_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        tools_menu = menubar.addMenu("工具")
        
        clear_log_action = QAction("清空日志", self)
        clear_log_action.triggered.connect(self.log_widget.clear)
        tools_menu.addAction(clear_log_action)
        
        clear_chart_action = QAction("清空图表", self)
        clear_chart_action.triggered.connect(self.chart.clear_chart)
        tools_menu.addAction(clear_chart_action)
        
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_statusbar(self):
        self.statusBar().showMessage("就绪 - 请配置参数并点击开始训练")
        
    def save_config(self):
        config = self.config_panel.get_config()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存配置", "training_config.json", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                QMessageBox.information(self, "成功", f"配置已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
    
    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载配置", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.config_panel.load_config(config)
                QMessageBox.information(self, "成功", "配置已加载")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")
    
    def show_about(self):
        QMessageBox.about(
            self, "关于",
            """<h2>N-BaIoT 自编码器训练系统</h2>
            <p>版本: 2.2 (最终修复版)</p>
            <p>基于TensorFlow/Keras的深度自编码器训练系统</p>
            <hr>
            <p><b>本次修复:</b></p>
            <ul>
                <li>✅ 修复图表标题显示方框问题</li>
                <li>✅ 修复KeyError: 'training_time'错误</li>
                <li>✅ 优化曲线显示，限制200个epoch历史</li>
                <li>✅ 添加当前点高亮标记</li>
            </ul>
            """
        )
    
    def update_overall_progress(self, progress: float):
        self.statusBar().showMessage(f"训练进度: {progress:.1f}%")
    
    def closeEvent(self, event):
        if self.control_panel.training_worker and self.control_panel.training_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出", "训练正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
            
            self.control_panel.training_worker.stop()
        
        event.accept()


# ============================================================================
# 主程序入口
# ============================================================================
def main():
    """主程序入口"""
    if not GUI_AVAILABLE:
        print("❌ 错误: PyQt5未安装")
        print("请运行: pip install PyQt5")
        return
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    app.setApplicationName("N-BaIoT Autoencoder Training System")
    app.setOrganizationName("MiniMax")
    
    window = MainWindow()
    window.show()
    
    print("\n" + "="*60)
    print("N-BaIoT 自编码器训练系统 - GUI版 (最终修复)")
    print("="*60)
    print("TensorFlow 版本:", tf.__version__)
    print(" Keras 版本:", tf.keras.__version__)
    print("="*60 + "\n")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()