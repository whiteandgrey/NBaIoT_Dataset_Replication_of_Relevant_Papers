from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QProgressBar, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt

from core.signals import TrainingSignals
from core.training.training_worker import TrainingWorker
from config import Config


class TrainingControlPanel(QWidget):
    """
    训练控制面板
    
    用于控制训练过程的开始、暂停、停止等操作，显示训练进度和状态。
    """
    
    def __init__(self, parent=None):
        """
        初始化训练控制面板
        
        Args:
            parent: 父组件
        """
        super().__init__(parent)
        self.main_window = parent
        self.training_worker = None
        self.init_ui()
        
    def init_ui(self):
        """
        初始化UI
        """
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
        """
        开始训练
        """
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
        """
        断开所有信号连接，防止重复连接导致的重复日志
        """
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
        """
        暂停训练
        """
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
        """
        停止训练
        """
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
    
    def on_training_started(self, device_name: str):
        """
        训练开始回调
        
        Args:
            device_name: 设备名称
        """
        self.main_window.log_widget.append(f"\n{'='*60}")
        self.main_window.log_widget.append(f"🚀 开始训练设备: {device_name}")
        self.main_window.log_widget.append(f"{'='*60}\n")
        
        if self.main_window.chart:
            self.main_window.chart.clear_chart()
    
    def on_progress_update(self, progress: dict):
        """
        进度更新回调
        
        Args:
            progress: 进度信息
        """
        self.progress_bar.setValue(int(progress.get('progress', 0)))
    
    def on_epoch_completed(self, data: dict):
        """
        Epoch完成回调
        
        Args:
            data: Epoch数据
        """
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
    
    def on_phase_completed(self, data: dict):
        """
        阶段完成回调
        
        Args:
            data: 阶段数据
        """
        phase = data.get('phase', '')
        loss = data.get('loss', 0)
        self.main_window.log_widget.append(f"✅ {phase}完成，最佳损失: {loss:.6f}")
    
    def on_device_completed(self, data: dict):
        """
        设备完成回调
        
        Args:
            data: 设备数据
        """
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
    
    def on_training_finished(self, data: dict):
        """
        训练完成回调
        
        Args:
            data: 训练结果数据
        """
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
    
    def on_training_error(self, error: str):
        """
        训练错误回调
        
        Args:
            error: 错误信息
        """
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
    
    def on_log_received(self, log: str):
        """
        日志接收回调
        
        Args:
            log: 日志信息
        """
        self.main_window.log_widget.append(log)
    
    def on_status_update(self, status: str):
        """
        状态更新回调
        
        Args:
            status: 状态信息
        """
        self.status_label.setText(status)



