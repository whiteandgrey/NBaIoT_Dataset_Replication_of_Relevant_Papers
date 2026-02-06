from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 检查matplotlib是否可用
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class RealTimeChart(QWidget):
    """
    实时训练曲线图表组件

    用于显示训练过程中的实时损失曲线，支持动态更新数据点，
    限制最大数据点数量以优化性能，提供清空图表功能。
    """

    def __init__(self, parent=None):
        """
        初始化实时图表组件

        Args:
            parent: 父组件
        """
        super().__init__(parent)
        self.setMinimumHeight(300)
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
        """
        设置图表样式
        """
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
        """
        更新图表数据

        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            val_loss: 验证损失
            phase: 当前训练阶段
            total_epochs: 总epoch数
        """
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
        """
        清空图表
        """
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
        """
        响应窗口大小变化
        """
        super().resizeEvent(event)
        if MATPLOTLIB_AVAILABLE:
            self.figure.tight_layout()
