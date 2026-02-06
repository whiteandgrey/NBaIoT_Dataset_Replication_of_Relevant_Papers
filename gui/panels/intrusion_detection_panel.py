import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QGroupBox, 
    QLabel, QLineEdit, QComboBox, QPushButton, QProgressBar, QTextEdit, 
    QFileDialog, QMessageBox, QSizePolicy, QGridLayout, QCheckBox
)
from PyQt5.QtCore import Qt

from core.signals import IntrusionDetectionSignals
from config import Config


class IntrusionDetectionPanel(QWidget):
    """
    入侵检测与评估面板
    
    用于管理入侵检测与评估功能，包括文件选择、评估过程、
    结果显示和保存选项等。
    """
    
    def __init__(self, parent=None):
        """
        初始化入侵检测与评估面板
        
        Args:
            parent: 父组件
        """
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
    
    def init_ui(self):
        """
        初始化UI
        """
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
        """
        创建文件选择页面
        
        Returns:
            文件选择页面组件
        """
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 设备选择
        device_group = QGroupBox("📱 设备选择")
        device_layout = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(Config.ALL_DEVICES)
        self.device_combo.setCurrentText('Danmini_Doorbell')
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        device_layout.addRow(QLabel("设备名称:"), self.device_combo)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # DStst文件选择
        dstst_group = QGroupBox("📁 DStst文件选择")
        dstst_layout = QFormLayout()

        self.dstst_data_edit = QLineEdit()
        self.dstst_data_edit.setPlaceholderText("选择dstst_data.npy文件")

        dstst_data_browse_btn = QPushButton("浏览...")
        dstst_data_browse_btn.clicked.connect(self.browse_dstst_data_file)

        dstst_data_file_layout = QHBoxLayout()
        dstst_data_file_layout.addWidget(self.dstst_data_edit)
        dstst_data_file_layout.addWidget(dstst_data_browse_btn)

        dstst_layout.addRow(QLabel("DStst数据文件:"), dstst_data_file_layout)

        self.dstst_labels_edit = QLineEdit()
        self.dstst_labels_edit.setPlaceholderText("选择dstst_labels.npy文件")

        dstst_labels_browse_btn = QPushButton("浏览...")
        dstst_labels_browse_btn.clicked.connect(self.browse_dstst_labels_file)

        dstst_labels_file_layout = QHBoxLayout()
        dstst_labels_file_layout.addWidget(self.dstst_labels_edit)
        dstst_labels_file_layout.addWidget(dstst_labels_browse_btn)

        dstst_layout.addRow(QLabel("DStst标签文件:"), dstst_labels_file_layout)

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
        
        # 滑动窗口配置
        window_group = QGroupBox("⚙️ 滑动窗口配置")
        window_layout = QFormLayout()
        
        self.min_window_size_edit = QLineEdit()
        self.min_window_size_edit.setText(str(Config.MIN_WINDOW_SIZE))
        self.min_window_size_edit.setPlaceholderText("最小窗口大小")
        window_layout.addRow(QLabel("最小窗口大小:"), self.min_window_size_edit)
        
        self.max_window_size_edit = QLineEdit()
        self.max_window_size_edit.setText(str(Config.MAX_WINDOW_SIZE))
        self.max_window_size_edit.setPlaceholderText("最大窗口大小")
        window_layout.addRow(QLabel("最大窗口大小:"), self.max_window_size_edit)
        
        self.window_size_step_edit = QLineEdit()
        self.window_size_step_edit.setText(str(Config.WINDOW_SIZE_STEP))
        self.window_size_step_edit.setPlaceholderText("窗口大小步长")
        window_layout.addRow(QLabel("窗口大小步长:"), self.window_size_step_edit)
        
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_evaluation_tab(self) -> QWidget:
        """
        创建评估过程页面
        
        Returns:
            评估过程页面组件
        """
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

        self.evaluation_chart = EvaluationChart()
        self.evaluation_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(self.evaluation_chart)
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_results_tab(self) -> QWidget:
        """
        创建评估结果页面
        
        Returns:
            评估结果页面组件
        """
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
        """
        创建保存选项页面
        
        Returns:
            保存选项页面组件
        """
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
    
    def browse_dstst_data_file(self):
        """
        浏览DStst数据文件
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择DStst数据文件",
            self.dstst_data_edit.text(),
            "Numpy文件 (*.npy);;All Files (*)"
        )

        if file_path:
            self.dstst_data_edit.setText(file_path)
            self.validate_dstst_files()

    def browse_dstst_labels_file(self):
        """
        浏览DStst标签文件
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择DStst标签文件",
            self.dstst_labels_edit.text(),
            "Numpy文件 (*.npy);;All Files (*)"
        )

        if file_path:
            self.dstst_labels_edit.setText(file_path)
            self.validate_dstst_files()
    
    def browse_model_file(self):
        """
        浏览模型文件
        """
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
        """
        浏览保存路径
        """
        directory = QFileDialog.getExistingDirectory(
            self, "选择保存目录",
            self.save_path_edit.text()
        )
        
        if directory:
            self.save_path_edit.setText(directory)
    
    def validate_dstst_files(self):
        """
        验证DStst文件（数据文件和标签文件）
        """
        data_file = self.dstst_data_edit.text()
        labels_file = self.dstst_labels_edit.text()

        if not data_file or not labels_file:
            self.dstst_status_label.setText("未选择文件")
            self.dstst_status_label.setStyleSheet("color: #666666;")
            return

        try:
            import numpy as np

            if not os.path.exists(data_file):
                self.dstst_status_label.setText("数据文件不存在")
                self.dstst_status_label.setStyleSheet("color: #f44336;")
                return

            if not os.path.exists(labels_file):
                self.dstst_status_label.setText("标签文件不存在")
                self.dstst_status_label.setStyleSheet("color: #f44336;")
                return

            data = np.load(data_file)
            labels = np.load(labels_file)

            if len(data) != len(labels):
                self.dstst_status_label.setText("数据和标签数量不匹配")
                self.dstst_status_label.setStyleSheet("color: #f44336;")
                return

            self.dstst_status_label.setText(f"文件有效 (数据: {len(data)} 样本, 标签: {len(labels)} 样本)")
            self.dstst_status_label.setStyleSheet("color: #4CAF50;")
        except Exception as e:
            self.dstst_status_label.setText(f"文件无效: {str(e)}")
            self.dstst_status_label.setStyleSheet("color: #f44336;")
    
    def validate_model_file(self, file_path):
        """
        验证模型文件
        
        Args:
            file_path: 模型文件路径
        """
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
        """
        开始评估
        """
        # 检查DStst文件是否存在
        data_file = self.dstst_data_edit.text()
        labels_file = self.dstst_labels_edit.text()

        if not data_file or not labels_file:
            reply = QMessageBox.question(
                self, "未找到DStst文件",
                f"未找到DStst文件。\n\n是否自动生成DStst文件？",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                if not self.generate_dstst_files():
                    return
            else:
                return

        # 获取配置
        try:
            min_window_size = int(self.min_window_size_edit.text())
            max_window_size = int(self.max_window_size_edit.text())
            window_size_step = int(self.window_size_step_edit.text())
        except ValueError:
            QMessageBox.warning(self, "警告", "滑动窗口配置必须是整数！")
            return
        
        config = {
            'device_name': self.device_combo.currentText(),
            'dstst_data_file': self.dstst_data_edit.text(),
            'dstst_labels_file': self.dstst_labels_edit.text(),
            'model_file': self.model_file_edit.text(),
            'save_path': self.save_path_edit.text(),
            'save_data': self.save_data_check.isChecked(),
            'save_images': self.save_images_check.isChecked(),
            'min_window_size': min_window_size,
            'max_window_size': max_window_size,
            'window_size_step': window_size_step
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
        """
        停止评估
        """
        if self.evaluation_worker and self.evaluation_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认", "确定要停止评估吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.evaluation_worker.stop()
    
    def _disconnect_signals(self):
        """
        断开所有信号连接
        """
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
    
    def on_evaluation_started(self, device_name: str):
        """
        评估开始回调
        
        Args:
            device_name: 设备名称
        """
        self.log_widget.append(f"\n{'='*60}")
        self.log_widget.append(f"🚀 开始评估设备: {device_name}")
        self.log_widget.append(f"{'='*60}\n")
    
    def on_progress_update(self, progress: dict):
        """
        进度更新回调
        
        Args:
            progress: 进度信息
        """
        self.progress_bar.setValue(int(progress.get('progress', 0)))
    
    def on_data_updated(self, data: dict):
        """
        数据更新回调

        Args:
            data: 数据信息
        """
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

        # 更新评估图表
        self.evaluation_chart.update_chart(
            sample_index=current_sample,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
            total_samples=total_samples
        )
    
    def on_evaluation_completed(self, results: dict):
        """
        评估完成回调

        Args:
            results: 评估结果
        """
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
            f"处理样本数: {total_samples}\n"
            f"混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}"
        )
    
    def on_evaluation_error(self, error: str):
        """
        评估错误回调
        
        Args:
            error: 错误信息
        """
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
    
    def on_log_received(self, log: str):
        """
        日志接收回调
        
        Args:
            log: 日志信息
        """
        self.log_widget.append(log)
    
    def on_status_update(self, status: str):
        """
        状态更新回调
        
        Args:
            status: 状态信息
        """
        self.status_label.setText(status)
    
    def on_device_changed(self, device_name: str):
        """
        设备选择变化回调

        Args:
            device_name: 设备名称
        """
        # 检查DStst文件
        self.check_dstst_files()
        
        # 自动切换模型文件路径
        self.check_model_file()
    
    def check_model_file(self):
        """
        检查默认路径下是否存在模型文件
        """
        device_name = self.device_combo.currentText()
        training_results_dir = os.path.join(Config.OUTPUT_DIR, device_name)
        
        # 检查final_model.h5
        model_file = os.path.join(training_results_dir, "final_model.h5")
        if not os.path.exists(model_file):
            # 检查best_model.h5
            model_file = os.path.join(training_results_dir, "best_model.h5")
        
        if os.path.exists(model_file):
            self.model_file_edit.setText(model_file)
            self.log_widget.append(f"✅ 找到模型文件: {model_file}")
        else:
            self.model_file_edit.clear()
            self.log_widget.append(f"⚠️ 未找到模型文件: {training_results_dir}")

    def check_dstst_files(self):
        """
        检查默认路径下是否存在DStst文件
        """
        device_name = self.device_combo.currentText()
        default_dir = os.path.join(Config.OUTPUT_DIR, device_name)

        data_file = os.path.join(default_dir, "dstst_data.npy")
        labels_file = os.path.join(default_dir, "dstst_labels.npy")

        if os.path.exists(data_file) and os.path.exists(labels_file):
            self.dstst_data_edit.setText(data_file)
            self.dstst_labels_edit.setText(labels_file)
            self.validate_dstst_files()
            self.log_widget.append(f"✅ 找到DStst文件: {default_dir}")
        else:
            self.dstst_data_edit.clear()
            self.dstst_labels_edit.clear()
            self.dstst_status_label.setText("未找到DStst文件")
            self.dstst_status_label.setStyleSheet("color: #FF9800;")

    def generate_dstst_files(self):
        """
        自动生成DStst文件
        """
        device_name = self.device_combo.currentText()
        save_dir = os.path.join(Config.OUTPUT_DIR, device_name)

        try:
            from data_integrator import DStstIntegrator

            self.log_widget.append(f"\n{'='*60}")
            self.log_widget.append(f"开始生成DStst文件...")
            self.log_widget.append(f"设备: {device_name}")
            self.log_widget.append(f"保存目录: {save_dir}")
            self.log_widget.append(f"{'='*60}")

            integrator = DStstIntegrator(Config)
            dstst_data, dstst_labels = integrator.create_dstst(device_name)
            data_path, labels_path = integrator.save_dstst(device_name, dstst_data, dstst_labels, save_dir)

            self.dstst_data_edit.setText(data_path)
            self.dstst_labels_edit.setText(labels_path)
            self.validate_dstst_files()

            self.log_widget.append(f"\n✅ DStst文件生成成功！")
            self.log_widget.append(f"   数据文件: {data_path}")
            self.log_widget.append(f"   标签文件: {labels_path}")
            self.log_widget.append(f"   数据样本数: {len(dstst_data)}")
            self.log_widget.append(f"   标签样本数: {len(dstst_labels)}")

            return True
        except Exception as e:
            self.log_widget.append(f"\n❌ 生成DStst文件失败: {str(e)}")
            QMessageBox.critical(self, "生成失败", f"生成DStst文件失败:\n{str(e)}")
            return False

    def on_file_generated(self, message: str):
        """
        文件生成回调

        Args:
            message: 消息信息
        """
        self.log_widget.append(f"\n✅ {message}")
    
    def on_save_completed(self, message: str):
        """
        保存完成回调
        
        Args:
            message: 消息信息
        """
        self.log_widget.append(f"\n✅ {message}")


# 避免循环导入
from core.detection.intrusion_detection_worker import IntrusionDetectionWorker
from gui.widgets import RealTimeChart
from gui.widgets.evaluation_chart import EvaluationChart
from PyQt5.QtGui import QFont
