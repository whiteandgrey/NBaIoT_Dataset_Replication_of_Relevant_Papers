from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EvaluationChart(QWidget):
    """
    评估过程图表组件
    
    用于显示评估过程中的实时性能指标变化曲线，支持动态更新数据点，
    限制最大数据点数量以优化性能，提供清空图表功能。
    """

    def __init__(self, parent=None):
        """
        初始化评估图表组件

        Args:
            parent: 父组件
        """
        super().__init__(parent)
        self.setMinimumHeight(400)
        self.max_data_points = 200  # 限制最大数据点数量，避免曲线过多
        self.setup_plot()

    def setup_plot(self):
        """
        设置图表
        """
        if not MATPLOTLIB_AVAILABLE:
            self.setStyleSheet("background-color: #2d2d2d;")
            return

        # 创建图表
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.figure.patch.set_facecolor('#1e1e1e')

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")

        # 设置布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 初始化数据
        self.sample_indices = []
        self.accuracy_values = []
        self.precision_values = []
        self.recall_values = []
        self.f1_values = []
        self.fpr_values = []

        # 设置图表样式
        self.setup_chart_style()

    def setup_chart_style(self):
        """
        设置图表样式
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        # 配置中文字体
        try:
            import matplotlib.font_manager as fm
            # 尝试使用常见的中文字体
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            # 找到第一个可用的中文字体
            font_name = None
            for font in chinese_fonts:
                if font in available_fonts:
                    font_name = font
                    break
            
            if font_name:
                plt.rcParams['font.sans-serif'] = [font_name]
                print(f"Using Chinese font: {font_name}")
            else:
                print("Warning: No Chinese font found, using default font")
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        except Exception as e:
            print(f"Warning: Failed to configure Chinese font: {e}")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False

        # 使用dark_background样式
        plt.style.use('dark_background')

        # 创建2x2的子图布局
        self.ax1 = self.figure.add_subplot(2, 2, 1)
        self.ax2 = self.figure.add_subplot(2, 2, 2)
        self.ax3 = self.figure.add_subplot(2, 2, 3)
        self.ax4 = self.figure.add_subplot(2, 2, 4)

        # 设置坐标轴颜色
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#1e1e1e')
            ax.spines['bottom'].set_color('#ffffff')
            ax.spines['top'].set_color('#ffffff')
            ax.spines['left'].set_color('#ffffff')
            ax.spines['right'].set_color('#ffffff')
            ax.tick_params(axis='x', colors='#ffffff')
            ax.tick_params(axis='y', colors='#ffffff')
            ax.yaxis.label.set_color('#ffffff')
            ax.xaxis.label.set_color('#ffffff')
            ax.title.set_color('#ffffff')

        # 初始化线条
        self.accuracy_line, = self.ax1.plot([], [], color='#4CAF50', linewidth=2,
                                                 label='Accuracy', marker='o', markersize=4)
        self.precision_line, = self.ax1.plot([], [], color='#2196F3', linewidth=2,
                                                  label='Precision', marker='s', markersize=4)
        self.recall_line, = self.ax2.plot([], [], color='#FF9800', linewidth=2,
                                               label='Recall', marker='^', markersize=4)
        self.f1_line, = self.ax2.plot([], [], color='#9C27B0', linewidth=2,
                                            label='F1 Score', marker='v', markersize=4)
        self.fpr_line, = self.ax3.plot([], [], color='#FF5722', linewidth=2,
                                              label='FPR', marker='d', markersize=4)

        # 设置标签
        self.ax1.set_ylabel('Score', fontsize=10, color='white')
        self.ax1.set_xlabel('Sample Index', fontsize=10, color='white')
        self.ax1.set_title('Accuracy & Precision', fontsize=12, color='white', fontweight='bold')
        self.ax1.legend(loc='lower left', facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=8)
        self.ax1.grid(True, alpha=0.3, color='gray')

        self.ax2.set_ylabel('Score', fontsize=10, color='white')
        self.ax2.set_xlabel('Sample Index', fontsize=10, color='white')
        self.ax2.set_title('Recall & F1 Score', fontsize=12, color='white', fontweight='bold')
        self.ax2.legend(loc='lower left', facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=8)
        self.ax2.grid(True, alpha=0.3, color='gray')

        self.ax3.set_ylabel('FPR', fontsize=10, color='white')
        self.ax3.set_xlabel('Sample Index', fontsize=10, color='white')
        self.ax3.set_title('False Positive Rate', fontsize=12, color='white', fontweight='bold')
        self.ax3.legend(loc='upper right', facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=8)
        self.ax3.grid(True, alpha=0.3, color='gray')

        # 创建性能指标汇总文本框
        self.ax4.axis('off')
        self.ax4.set_facecolor('#1e1e1e')
        self.summary_text = self.ax4.text(0.1, 0.9, '', transform=self.ax4.transAxes,
                                       fontsize=10, color='white', verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8))

        self.figure.tight_layout()

    def update_chart(self, sample_index: int, accuracy: float, precision: float,
                 recall: float, f1: float, fpr: float, total_samples: int):
        """
        更新图表数据

        Args:
            sample_index: 当前样本索引
            accuracy: 准确率
            precision: 精确率
            recall: 召回率
            f1: F1分数
            fpr: 误报率
            total_samples: 总样本数
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        # 添加数据点
        self.sample_indices.append(sample_index)
        self.accuracy_values.append(accuracy)
        self.precision_values.append(precision)
        self.recall_values.append(recall)
        self.f1_values.append(f1)
        self.fpr_values.append(fpr)

        # 限制数据点数量：如果超过最大限制，移除最旧的数据
        if len(self.sample_indices) > self.max_data_points:
            self.sample_indices = self.sample_indices[-self.max_data_points:]
            self.accuracy_values = self.accuracy_values[-self.max_data_points:]
            self.precision_values = self.precision_values[-self.max_data_points:]
            self.recall_values = self.recall_values[-self.max_data_points:]
            self.f1_values = self.f1_values[-self.max_data_points:]
            self.fpr_values = self.fpr_values[-self.max_data_points:]

        # 更新线条数据
        self.accuracy_line.set_data(self.sample_indices, self.accuracy_values)
        self.precision_line.set_data(self.sample_indices, self.precision_values)
        self.recall_line.set_data(self.sample_indices, self.recall_values)
        self.f1_line.set_data(self.sample_indices, self.f1_values)
        self.fpr_line.set_data(self.sample_indices, self.fpr_values)

        # 更新汇总文本
        progress = (sample_index / total_samples) * 100
        summary_text = f"""
        评估进度: {progress:.1f}%
        当前样本: {sample_index}/{total_samples}
        
        实时性能指标:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        准确率: {accuracy:.4f}
        精确率: {precision:.4f}
        召回率: {recall:.4f}
        F1分数: {f1:.4f}
        误报率: {fpr:.4f}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        self.summary_text.set_text(summary_text)

        # 动态调整坐标轴
        if len(self.sample_indices) > 1:
            x_max = max(self.sample_indices) * 1.05
            self.ax1.set_xlim(0, max(100, x_max))
            self.ax2.set_xlim(0, max(100, x_max))
            self.ax3.set_xlim(0, max(100, x_max))

            # Y轴范围
            all_values = self.accuracy_values + self.precision_values + self.recall_values + self.f1_values + self.fpr_values
            if all_values:
                y_max = max(all_values) * 1.1
                y_min = min(all_values) * 0.9
                self.ax1.set_ylim(max(0, y_min), min(1.0, y_max))
                self.ax2.set_ylim(max(0, y_min), min(1.0, y_max))
                self.ax3.set_ylim(max(0, y_min), min(1.0, y_max))

        # 刷新图表
        self.canvas.draw_idle()

    def clear_chart(self):
        """
        清空图表
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        self.sample_indices = []
        self.accuracy_values = []
        self.precision_values = []
        self.recall_values = []
        self.f1_values = []
        self.fpr_values = []

        self.accuracy_line.set_data([], [])
        self.precision_line.set_data([], [])
        self.recall_line.set_data([], [])
        self.f1_line.set_data([], [])
        self.fpr_line.set_data([], [])

        self.ax1.set_xlim(0, 100)
        self.ax2.set_xlim(0, 100)
        self.ax3.set_xlim(0, 100)
        self.ax1.set_ylim(0, 1.0)
        self.ax2.set_ylim(0, 1.0)
        self.ax3.set_ylim(0, 1.0)

        self.summary_text.set_text('等待评估开始...')

        self.canvas.draw_idle()

    def resizeEvent(self, event):
        """
        响应窗口大小变化
        """
        super().resizeEvent(event)
        if MATPLOTLIB_AVAILABLE:
            self.figure.tight_layout()
