import sys
import json
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QSplitter, QLabel, 
    QTextEdit, QMenuBar, QAction, QFileDialog, QMessageBox, QStatusBar, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from gui.panels import ConfigPanel, TrainingControlPanel, IntrusionDetectionPanel
from gui.widgets import RealTimeChart


class MainWindow(QMainWindow):
    """
    主窗口
    
    应用程序的主窗口，包含训练和入侵检测两个主要选项卡，
    负责协调各个面板和组件的交互。
    """
    
    def __init__(self):
        """
        初始化主窗口
        """
        super().__init__()
        self.setWindowTitle("N-BaIoT 自编码器训练系统 - GUI版 (最终修复)")
        self.setMinimumSize(1400, 900)
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()
        
    def setup_ui(self):
        """
        设置UI
        """
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
        progress_layout = QVBoxLayout()
        
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
        """
        设置菜单栏
        """
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
        """
        设置状态栏
        """
        self.statusBar().showMessage("就绪 - 请配置参数并点击开始训练")
        
    def save_config(self):
        """
        保存配置
        """
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
        """
        加载配置
        """
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
        """
        显示关于对话框
        """
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
                <li>✅ 修复模型保存功能，确保勾选后能正确保存模型文件</li>
            </ul>
            """
        )
    
    def update_overall_progress(self, progress: float):
        """
        更新整体进度
        
        Args:
            progress: 进度百分比
        """
        self.statusBar().showMessage(f"训练进度: {progress:.1f}%")
    
    def closeEvent(self, event):
        """
        关闭事件处理
        """
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
