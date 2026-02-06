from typing import Dict, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QGroupBox, 
    QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, 
    QPushButton, QFileDialog, QScrollArea, QGridLayout, QListWidget, QListWidgetItem, QFrame
)
from PyQt5.QtCore import Qt

from config import Config


class ConfigPanel(QWidget):
    """
    配置面板组件
    
    用于管理训练系统的所有配置参数，包括基础配置、模型架构、
    训练参数、设备选择、保存选项和高级选项等。
    """
    
    def __init__(self, parent=None):
        """
        初始化配置面板
        
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
        self.tab_widget.addTab(self.create_basic_config(), "基础配置")
        self.tab_widget.addTab(self.create_model_config(), "模型架构")
        self.tab_widget.addTab(self.create_training_config(), "训练参数")
        self.tab_widget.addTab(self.create_device_config(), "设备选择")
        self.tab_widget.addTab(self.create_save_config(), "保存选项")
        self.tab_widget.addTab(self.create_advanced_config(), "高级选项")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
    def create_basic_config(self) -> QWidget:
        """
        创建基础配置页面
        
        Returns:
            基础配置页面组件
        """
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
        """
        创建模型架构配置页面
        
        Returns:
            模型架构配置页面组件
        """
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
        """
        创建训练参数配置页面
        
        Returns:
            训练参数配置页面组件
        """
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
        """
        创建设备选择配置页面
        
        Returns:
            设备选择配置页面组件
        """
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
        """
        创建保存选项配置页面
        
        Returns:
            保存选项配置页面组件
        """
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
        """
        创建高级配置页面
        
        Returns:
            高级配置页面组件
        """
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
        """
        浏览数据根目录
        """
        directory = QFileDialog.getExistingDirectory(
            self, "选择N-BaIoT数据集目录",
            self.data_root_edit.text()
        )
        if directory:
            self.data_root_edit.setText(directory)
            
    def browse_output_dir(self):
        """
        浏览输出目录
        """
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
        """
        全选设备
        """
        for i in range(self.device_list.count()):
            self.device_list.item(i).setSelected(True)
            
    def deselect_all_devices(self):
        """
        全不选设备
        """
        for i in range(self.device_list.count()):
            self.device_list.item(i).setSelected(False)
    
    def get_config(self) -> Dict:
        """
        获取配置
        
        Returns:
            配置字典
        """
        def parse_list(text: str, default: List):
            try:
                return eval(text)
            except:
                return default
        
        selected_devices = []
        for i in range(self.device_list.count()):
            if self.device_list.item(i).isSelected():
                selected_devices.append(self.device_list.item(i).text())
        
        config_dict = {
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
        """
        加载配置
        
        Args:
            config: 配置字典
        """
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
