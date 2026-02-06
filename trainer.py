"""
训练模块 - 含超参数优化和早停
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import json


class AutoencoderTrainer:
    """自编码器训练器"""

    def __init__(self, config, device_name):
        """
        初始化训练器

        Args:
            config: 配置对象
            device_name: 设备名称
        """
        self.config = config
        self.device_name = device_name
        self.model = None
        self.history = {}

        # 训练历史记录
        self.training_history = {
            'initial_train': None,
            'hyperparameter_tuning': [],
            'final_train': None,
            'best_params': None,
            'best_val_loss': float('inf')
        }

        # 创建设备特定的输出目录
        self.device_output_dir = os.path.join(config.OUTPUT_DIR, device_name)
        os.makedirs(self.device_output_dir, exist_ok=True)

        print(f"✅ Trainer initialized for device: {device_name}")
        print(f"   Output directory: {self.device_output_dir}")

    def create_callbacks(self, monitor='val_loss', mode='min',
                         patience=None, save_best_only=True):
        """
        创建训练回调函数

        Args:
            monitor: 监控指标
            mode: 监控模式（'min'或'max'）
            patience: 早停耐心值
            save_best_only: 是否只保存最佳模型

        Returns:
            回调函数列表
        """
        if patience is None:
            patience = self.config.EARLY_STOPPING_PATIENCE

        callbacks = []

        # 早停回调
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=self.config.MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # 学习率调整回调
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            mode=mode,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # 模型检查点回调（根据配置决定是否保存）
        if self.config.SAVE_MODEL:
            model_checkpoint = ModelCheckpoint(
                filepath=os.path.join(self.device_output_dir, 'best_model.h5'),
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                verbose=1
            )
            callbacks.append(model_checkpoint)

        # TensorBoard回调（根据配置决定是否保存）
        if self.config.SAVE_TENSORBOARD_LOGS:
            try:
                tensorboard_dir = os.path.join(self.device_output_dir, 'tensorboard_logs')
                os.makedirs(tensorboard_dir, exist_ok=True)

                tensorboard = TensorBoard(
                    log_dir=tensorboard_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=False,
                    update_freq='epoch'
                )
                callbacks.append(tensorboard)
            except Exception as e:
                print(f"⚠️ TensorBoard callback error: {e}")

        return callbacks

    def train(self, train_data, val_data, model=None,
              learning_rate=None, epochs=None, batch_size=None,
              phase_name="Training", verbose=1):
        """
        训练模型

        Args:
            train_data: 训练数据，可以是(X_train, y_train)或tf.data.Dataset
            val_data: 验证数据
            model: 要训练的模型，如果为None则使用self.model
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批大小（仅用于NumPy数据）
            phase_name: 训练阶段名称
            verbose: 详细程度

        Returns:
            训练历史和最佳验证损失
        """
        print(f"\n{'=' * 60}")
        print(f"PHASE: {phase_name}")
        print(f"Device: {self.device_name}")
        print(f"{'=' * 60}")

        start_time = time.time()

        # 设置参数
        if model is None:
            model = self.model
        if learning_rate is None:
            learning_rate = self.config.DEFAULT_LEARNING_RATE
        if epochs is None:
            epochs = self.config.DEFAULT_EPOCHS

        # 编译模型
        if hasattr(model, 'compile'):
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )

        # 创建回调函数
        callbacks = self.create_callbacks(monitor='val_loss', mode='min')

        # 训练模型
        print(f"🔧 Training parameters:")
        print(f"   Learning rate: {learning_rate:.6f}")
        print(f"   Epochs: {epochs}")

        # 检查数据类型
        if isinstance(train_data, tuple) and isinstance(val_data, tuple):
            # NumPy数据
            X_train, y_train = train_data
            X_val, y_val = val_data

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size or self.config.DEFAULT_BATCH_SIZE,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            # TensorFlow数据集
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose
            )

        training_time = time.time() - start_time

        # 获取历史记录字典
        history_dict = history.history if hasattr(history, 'history') else history

        # 获取最佳验证损失
        if 'val_loss' in history_dict:
            best_val_loss = min(history_dict['val_loss'])
        else:
            best_val_loss = history_dict['loss'][-1] if history_dict['loss'] else float('inf')

        print(f"\n📊 {phase_name} completed:")
        print(f"   Best val loss: {best_val_loss:.6f}")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Total epochs trained: {len(history_dict['loss'])}")

        return history_dict, best_val_loss, training_time

    def initial_training(self, train_data, val_data, learning_rate=0.001, epochs=100):
        """
        初始训练阶段

        Args:
            train_data: 训练数据
            val_data: 验证数据
            learning_rate: 学习率
            epochs: epoch数

        Returns:
            最佳验证损失
        """
        print(f"\n{'=' * 60}")
        print(f"INITIAL TRAINING PHASE")
        print(f"Device: {self.device_name}")
        print(f"{'=' * 60}")

        # 创建模型
        from model import Autoencoder
        autoencoder = Autoencoder(self.config)
        model = autoencoder.build()
        self.model = model

        history_dict, best_val_loss, training_time = self.train(
            train_data=train_data,
            val_data=val_data,
            model=model,
            learning_rate=learning_rate,
            epochs=epochs,
            phase_name="Initial Training",
            verbose=self.config.VERBOSE
        )

        # 记录训练历史
        self.training_history['initial_train'] = {
            'history': history_dict,
            'training_time': training_time,
            'best_val_loss': best_val_loss
        }

        return best_val_loss

    def hyperparameter_tuning(self, train_data, val_data):
        """
        超参数调优阶段

        Args:
            train_data: 训练数据
            val_data: 验证数据

        Returns:
            最佳超参数
        """
        print(f"\n{'=' * 60}")
        print(f"HYPERPARAMETER TUNING PHASE")
        print(f"Device: {self.device_name}")
        print(f"{'=' * 60}")

        results = []
        best_val_loss = float('inf')
        best_params = None

        # 遍历超参数组合
        for lr in self.config.LEARNING_RATES:
            for epochs in self.config.EPOCHS_OPTIONS:
                print(f"\n🧪 Testing: LR={lr:.6f}, Epochs={epochs}")

                # 重新创建模型
                from model import Autoencoder
                autoencoder = Autoencoder(self.config)
                model = autoencoder.build()

                # 训练并评估
                history_dict, val_loss, tuning_time = self.train(
                    train_data=train_data,
                    val_data=val_data,
                    model=model,
                    learning_rate=lr,
                    epochs=epochs,
                    phase_name=f"Tuning LR={lr:.4f}, Epochs={epochs}",
                    verbose=0  # 静默模式，减少输出
                )

                # 记录结果
                result = {
                    'lr': lr,
                    'epochs': epochs,
                    'val_loss': val_loss,
                    'training_time': tuning_time
                }
                results.append(result)

                print(f"   Result: Val Loss={val_loss:.6f}, Time={tuning_time:.2f}s")

                # 更新最佳参数
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {'lr': lr, 'epochs': epochs}

        # 保存到训练历史
        self.training_history['hyperparameter_tuning'] = results
        self.training_history['best_params'] = best_params
        self.training_history['best_val_loss'] = best_val_loss

        # 保存调优结果（根据配置决定是否保存）
        if self.config.SAVE_HYPERPARAMETER_TUNING_RESULTS:
            tuning_results_path = os.path.join(self.device_output_dir, 'hyperparameter_tuning.json')
            with open(tuning_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Hyperparameter tuning results saved to: {tuning_results_path}")

        print(f"\n✅ Hyperparameter tuning completed!")
        print(f"   Best parameters: LR={best_params['lr']:.6f}, Epochs={best_params['epochs']}")
        print(f"   Best validation loss: {best_val_loss:.6f}")

        return best_params

    def final_training(self, train_data, val_data=None):
        """
        最终训练阶段（使用最佳参数）

        Args:
            train_data: 训练数据
            val_data: 验证数据（可选）

        Returns:
            最终训练损失
        """
        print(f"\n{'=' * 60}")
        print(f"FINAL TRAINING PHASE")
        print(f"Device: {self.device_name}")
        print(f"{'=' * 60}")

        # 获取最佳参数
        if self.training_history['best_params'] is None:
            print("⚠️ No best parameters found, using defaults")
            best_params = {
                'lr': self.config.DEFAULT_LEARNING_RATE,
                'epochs': self.config.DEFAULT_EPOCHS
            }
        else:
            best_params = self.training_history['best_params']

        print(f"🔧 Using best parameters:")
        print(f"   Learning rate: {best_params['lr']:.6f}")
        print(f"   Epochs: {best_params['epochs']}")

        # 重新创建模型
        from model import Autoencoder
        autoencoder = Autoencoder(self.config)
        model = autoencoder.build()
        self.model = model

        # 记录开始时间
        start_time = time.time()

        if val_data is not None:
            # 使用验证数据进行训练（带早停和验证）
            history_dict, best_val_loss, training_time = self.train(
                train_data=train_data,
                val_data=val_data,
                model=model,
                learning_rate=best_params['lr'],
                epochs=best_params['epochs'],
                phase_name="Final Training",
                verbose=self.config.VERBOSE
            )
        else:
            # 如果没有验证集，只使用训练集
            print("⚠️ No validation data provided, training without validation")

            if isinstance(train_data, tuple):
                X_train, y_train = train_data
                history = model.fit(
                    X_train, y_train,
                    epochs=best_params['epochs'],
                    batch_size=self.config.DEFAULT_BATCH_SIZE,
                    verbose=self.config.VERBOSE,
                    callbacks=self.create_callbacks(monitor='loss', mode='min')
                )
            else:
                history = model.fit(
                    train_data,
                    epochs=best_params['epochs'],
                    verbose=self.config.VERBOSE,
                    callbacks=self.create_callbacks(monitor='loss', mode='min')
                )

            history_dict = history.history
            best_val_loss = history_dict['loss'][-1] if history_dict['loss'] else float('inf')
            training_time = time.time() - start_time

        # 保存最终训练历史
        self.training_history['final_train'] = {
            'history': history_dict,
            'training_time': training_time,
            'best_val_loss': best_val_loss
        }

        # 保存最终模型（根据配置决定是否保存）
        if self.config.SAVE_MODEL:
            model_save_path = os.path.join(self.device_output_dir, 'final_model.h5')
            model.save(model_save_path)
            print(f"✅ Final model saved to: {model_save_path}")

        # 保存训练历史（根据配置决定是否保存）
        if self.config.SAVE_TRAINING_HISTORY:
            self.save_training_history()

        return best_val_loss

    def save_training_history(self):
        """保存训练历史到文件"""
        history_path = os.path.join(self.device_output_dir, 'training_history.json')

        # 转换numpy数组为列表以便JSON序列化
        history_dict = {}
        for key, value in self.training_history.items():
            if key in ['initial_train', 'final_train'] and value is not None:
                if 'history' in value:
                    # 转换history中的numpy数组
                    converted_history = {}
                    for metric, values in value['history'].items():
                        if hasattr(values, 'tolist'):
                            converted_history[metric] = values.tolist()
                        else:
                            converted_history[metric] = values
                    value['history'] = converted_history
                history_dict[key] = value
            elif key == 'hyperparameter_tuning':
                history_dict[key] = value
            elif key == 'best_params':
                history_dict[key] = value
            elif key == 'best_val_loss':
                history_dict[key] = float(value)  # 转换为Python float

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2, default=str)

        print(f"✅ Training history saved to: {history_path}")

    def load_training_history(self):
        """从文件加载训练历史"""
        history_path = os.path.join(self.device_output_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            print(f"✅ Training history loaded from: {history_path}")

    def get_training_summary(self):
        """获取训练摘要"""
        summary = {
            'device_name': self.device_name,
            'best_params': self.training_history.get('best_params'),
            'best_val_loss': self.training_history.get('best_val_loss'),
        }

        if self.training_history.get('initial_train'):
            summary['initial_training_time'] = self.training_history['initial_train'].get('training_time')
            summary['initial_best_val_loss'] = self.training_history['initial_train'].get('best_val_loss')

        if self.training_history.get('final_train'):
            summary['final_training_time'] = self.training_history['final_train'].get('training_time')
            summary['final_best_val_loss'] = self.training_history['final_train'].get('best_val_loss')

        return summary

