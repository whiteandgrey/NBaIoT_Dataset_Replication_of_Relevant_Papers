import sys
import os
import json
import time
import weakref
from typing import Dict, List, Optional
import numpy as np
from PyQt5.QtCore import QThread, QMutex

from config import Config
from data_processor import NBaIoTDataProcessor
from model import Autoencoder
from trainer import AutoencoderTrainer
from visualizer import ScientificVisualizer
from core.signals import TrainingSignals


def create_training_control_callback(tf_Callback):
    """
    动态创建TrainingControlCallback类
    
    Args:
        tf_Callback: TensorFlow回调类
        
    Returns:
        TrainingControlCallback类
    """
    class TrainingControlCallback(tf_Callback):
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
            """
            每个epoch开始时检查停止状态
            """
            worker = self.worker_ref()
            if worker is not None:
                if worker.should_stop:
                    self.model.stop_training = True
                    self.worker_signals.log.emit("🛑 停止信号已收到，正在停止训练...")
        
        def on_epoch_end(self, epoch, logs=None):
            """
            每个epoch结束时检查暂停状态并发送数据
            """
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
            """
            每个batch结束时检查状态
            """
            worker = self.worker_ref()
            if worker is None:
                return
                
            if worker.should_stop:
                self.model.stop_training = True
                return
                
            if worker.is_paused:
                while worker.is_paused and not worker.should_stop:
                    time.sleep(0.05)
    
    return TrainingControlCallback


class TrainingWorker(QThread):
    """
    训练工作线程 - 在后台执行训练任务
    """
    
    def __init__(self, config: Dict, signals: TrainingSignals):
        """
        初始化训练工作线程
        
        Args:
            config: 训练配置
            signals: 训练信号对象
        """
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
        
        # TensorFlow模块和回调类（延迟导入）
        self.tf = None
        self.tf_Callback = None
        self.tf_EarlyStopping = None
        self.tf_ReduceLROnPlateau = None
        
    def run(self):
        """
        执行训练
        """
        self.is_running = True
        self.should_stop = False
        self.is_paused = False
        
        try:
            # 首先设置环境（必须在导入TensorFlow之前）
            self._setup_environment()
            
            # 然后导入TensorFlow（确保在设置环境变量后才导入）
            import tensorflow as tf_module
            from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
            
            # 保存TensorFlow模块和回调类到实例变量
            self.tf = tf_module
            self.tf_Callback = Callback
            self.tf_EarlyStopping = EarlyStopping
            self.tf_ReduceLROnPlateau = ReduceLROnPlateau
            
            print(f"✅ TensorFlow imported successfully")
            print(f"   TensorFlow version: {tf_module.__version__}")
            print(f"   GPU available: {tf_module.config.list_physical_devices('GPU')}")
            
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
        """
        等待恢复
        """
        while self.is_paused and not self.should_stop:
            time.sleep(0.1)
    
    def _setup_environment(self):
        """
        设置环境
        """
        Config.USE_GPU = self.config.get('use_gpu', False)
        Config.GPU_DEVICES = self.config.get('gpu_devices', "0")
        Config.GPU_MEMORY_LIMIT = self.config.get('gpu_memory_limit')
        Config.DATA_ROOT = self.config.get('data_root', Config.DATA_ROOT)
        Config.OUTPUT_DIR = self.config.get('output_dir', Config.OUTPUT_DIR)
        
        # 调用Config.setup_environment()来设置环境变量（必须在导入TensorFlow之前）
        Config.setup_environment()
        
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
        """
        获取要训练的设备列表
        """
        selected_devices = self.config.get('selected_devices', [])
        
        if not selected_devices:
            return data_processor.get_available_devices()
        
        available = data_processor.get_available_devices()
        valid_devices = [d for d in selected_devices if d in available]
        
        if not valid_devices:
            return available
        
        return valid_devices
    
    def _train_device(self, device_name: str, data_processor, visualizer) -> Optional[Dict]:
        """
        训练单个设备
        
        Args:
            device_name: 设备名称
            data_processor: 数据处理器
            visualizer: 可视化器
            
        Returns:
            训练结果
        """
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
        TrainingControlCallback = create_training_control_callback(self.tf_Callback)
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
        
        # 保存模型（修复bug：添加实际的模型保存代码）
        if Config.SAVE_MODEL:
            model_path = os.path.join(trainer.device_output_dir, 'final_model.h5')
            trainer.model.save(model_path)
            self.signals.log.emit(f"✅ 模型已保存到: {model_path}")
        
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
                             control_callback = None,
                             input_dim: int = None) -> Dict:
        """
        带回调的训练
        """
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
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae']
        )
        
        # 创建回调
        callbacks = [
            self.tf_EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=trainer.config.EARLY_STOPPING_PATIENCE,
                mode='min',
                min_delta=trainer.config.MIN_DELTA,
                restore_best_weights=True,
                verbose=0
            ),
            self.tf_ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=trainer.config.REDUCE_LR_FACTOR,
                patience=trainer.config.REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                mode='min',
                verbose=0
            )
        ]
        
        if control_callback is None:
            TrainingControlCallback = create_training_control_callback(self.tf_Callback)
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
                                           control_callback = None,
                                           input_dim: int = None):
        """
        超参数调优（带图表更新）- 修复版，包含training_time
        """
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
                    optimizer=self.tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='mse',
                    metrics=['mae']
                )
                
                # 创建回调
                callbacks = [
                    self.tf_EarlyStopping(
                        monitor='val_loss',
                        patience=trainer.config.EARLY_STOPPING_PATIENCE,
                        mode='min',
                        min_delta=trainer.config.MIN_DELTA,
                        restore_best_weights=True,
                        verbose=0
                    )
                ]
                
                if control_callback is None:
                    TrainingControlCallback = create_training_control_callback(self.tf_Callback)
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
        """
        暂停训练
        """
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
        self.signals.status_update.emit("已暂停 - 点击继续恢复训练")
        
    def resume(self):
        """
        恢复训练
        """
        self.mutex.lock()
        self.is_paused = False
        self.mutex.unlock()
        self.signals.status_update.emit("正在恢复训练...")
        
    def stop(self):
        """
        停止训练
        """
        self.mutex.lock()
        self.should_stop = True
        self.is_paused = False
        self.mutex.unlock()
        self.signals.status_update.emit("正在停止训练...")
