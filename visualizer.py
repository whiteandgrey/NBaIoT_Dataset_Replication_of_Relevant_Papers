"""
科研级可视化模块 - 支持多种图表类型和灵活的保存/显示控制
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, LogLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats


class ScientificVisualizer:
    """科研级训练可视化器"""

    def __init__(self, config):
        """
        初始化可视化器

        Args:
            config: 配置对象
        """
        self.config = config
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

        # 不使用seaborn样式，直接设置白色背景
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.unicode_minus': False,
            'figure.autolayout': True,
            'figure.figsize': (10, 6),
            'figure.facecolor': 'white',
            'figure.edgecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#000000',
            'axes.titlecolor': '#000000',
            'axes.linewidth': 1.0,
            'xtick.color': '#000000',
            'xtick.labelcolor': '#000000',
            'ytick.color': '#000000',
            'ytick.labelcolor': '#000000',
            'text.color': '#000000',
            'legend.facecolor': 'white',
            'legend.edgecolor': '#333333',
            'legend.framealpha': 1.0,
            'legend.fontsize': 9,
            'grid.color': '#cccccc',
            'grid.alpha': 0.5,
            'grid.linewidth': 0.5
        })

        # 设置图表保存参数
        plt.rcParams['savefig.dpi'] = config.PLOT_DPI
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1

        # 创建输出目录结构
        self.setup_plot_directories()

        print(f"✅ Scientific visualizer initialized")
        print(f"   Plot save: {config.PLOT_SAVE}")
        print(f"   Output directory: {config.OUTPUT_DIR}")

    def setup_plot_directories(self):
        """设置图表输出目录结构"""
        # 只创建主输出目录，设备特定的目录会在保存时自动创建
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def save_or_show_plot(self, fig, filename, plot_type="training", device_name=None):
        """
        保存或显示图表

        Args:
            fig: matplotlib图形对象
            filename: 文件名（不含路径）
            plot_type: 图表类型（training/comparison/debug/metrics/data）
            device_name: 设备名称（用于设备特定目录）
        """
        # 确定保存目录
        if device_name:
            # 使用设备特定的目录
            save_dir = os.path.join(self.config.OUTPUT_DIR, device_name, plot_type)
        else:
            # 使用全局目录
            save_dir = os.path.join(self.config.OUTPUT_DIR, plot_type)

        # 创建目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)

        # 保存图表
        if self.config.PLOT_SAVE:
            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path, dpi=self.config.PLOT_DPI, bbox_inches='tight', 
                      facecolor='white', edgecolor='white')
            print(f"📊 Plot saved to: {save_path}")

        # 显示图表
        if self.config.PLOT_SHOW:
            plt.show()
        else:
            plt.close(fig)

    def generate_plot_filename(self, device_name, plot_name, timestamp=None):
        """
        生成图表文件名

        Args:
            device_name: 设备名称
            plot_name: 图表名称
            timestamp: 时间戳，如果为None则使用当前时间

        Returns:
            格式化的文件名
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = self.config.PLOT_FILENAME_PATTERN.format(
            device=device_name,
            plot_type=plot_name,
            timestamp=timestamp,
            format=self.config.PLOT_FORMAT
        )

        return filename

    # ========================================================================
    # 训练曲线图表
    # ========================================================================

    def plot_training_loss_curve(self, trainer, device_name):
        """
        绘制训练损失曲线

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_TRAINING_LOSS_CURVE:
            return

        print(f"📈 Plotting training loss curve for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制初始训练损失
        if trainer.training_history.get('initial_train'):
            history = trainer.training_history['initial_train']['history']
            print(f"   Initial train history type: {type(history)}")
            print(f"   Initial train history keys: {list(history.keys()) if hasattr(history, 'keys') else 'N/A'}")
            
            if isinstance(history, dict):
                if 'loss' in history:
                    loss_data = history['loss']
                    print(f"   Loss data type: {type(loss_data)}, length: {len(loss_data) if hasattr(loss_data, '__len__') else 'N/A'}")
                    
                    if hasattr(loss_data, '__len__') and len(loss_data) > 0:
                        epochs = range(1, len(loss_data) + 1)
                        ax.plot(epochs, loss_data, label='Initial Training Loss', 
                               color=self.colors[0], linewidth=2, alpha=0.8, marker='o', markersize=4)
                        print(f"   ✓ Plotted {len(loss_data)} points for Initial Training Loss")
                        
                        if 'val_loss' in history:
                            val_loss_data = history['val_loss']
                            if hasattr(val_loss_data, '__len__') and len(val_loss_data) > 0:
                                ax.plot(epochs, val_loss_data, label='Initial Validation Loss', 
                                       color=self.colors[1], linewidth=2, alpha=0.8, linestyle='--', marker='s', markersize=4)
                                print(f"   ✓ Plotted {len(val_loss_data)} points for Initial Validation Loss")
                    else:
                        print(f"   ✗ Loss data is empty or not iterable")
                else:
                    print(f"   ✗ 'loss' key not found in history")
            else:
                print(f"   ✗ History is not a dict: {type(history)}")

        # 绘制最终训练损失
        if trainer.training_history.get('final_train'):
            history = trainer.training_history['final_train']['history']
            print(f"   Final train history type: {type(history)}")
            print(f"   Final train history keys: {list(history.keys()) if hasattr(history, 'keys') else 'N/A'}")
            
            if isinstance(history, dict):
                if 'loss' in history:
                    loss_data = history['loss']
                    print(f"   Loss data type: {type(loss_data)}, length: {len(loss_data) if hasattr(loss_data, '__len__') else 'N/A'}")
                    
                    if hasattr(loss_data, '__len__') and len(loss_data) > 0:
                        epochs = range(1, len(loss_data) + 1)
                        ax.plot(epochs, loss_data, label='Final Training Loss', 
                               color=self.colors[2], linewidth=2, alpha=0.8, marker='o', markersize=4)
                        print(f"   ✓ Plotted {len(loss_data)} points for Final Training Loss")
                        
                        if 'val_loss' in history:
                            val_loss_data = history['val_loss']
                            if hasattr(val_loss_data, '__len__') and len(val_loss_data) > 0:
                                ax.plot(epochs, val_loss_data, label='Final Validation Loss', 
                                       color=self.colors[3], linewidth=2, alpha=0.8, linestyle='--', marker='s', markersize=4)
                                print(f"   ✓ Plotted {len(val_loss_data)} points for Final Validation Loss")
                    else:
                        print(f"   ✗ Loss data is empty or not iterable")
                else:
                    print(f"   ✗ 'loss' key not found in history")
            else:
                print(f"   ✗ History is not a dict: {type(history)}")

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'Training Loss Curve - {device_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = self.generate_plot_filename(device_name, "training_loss_curve")
        self.save_or_show_plot(fig, filename, "training", device_name)

    def plot_training_mae_curve(self, trainer, device_name):
        """
        绘制训练MAE曲线

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_TRAINING_MAE_CURVE:
            return

        print(f"📈 Plotting training MAE curve for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制初始训练MAE
        if trainer.training_history.get('initial_train'):
            history = trainer.training_history['initial_train']['history']
            print(f"   Initial train history type: {type(history)}")
            print(f"   Initial train history keys: {list(history.keys()) if hasattr(history, 'keys') else 'N/A'}")
            
            if isinstance(history, dict):
                if 'mae' in history:
                    mae_data = history['mae']
                    print(f"   MAE data type: {type(mae_data)}, length: {len(mae_data) if hasattr(mae_data, '__len__') else 'N/A'}")
                    
                    if hasattr(mae_data, '__len__') and len(mae_data) > 0:
                        epochs = range(1, len(mae_data) + 1)
                        ax.plot(epochs, mae_data, label='Initial Training MAE', 
                               color=self.colors[0], linewidth=2, alpha=0.8, marker='o', markersize=4)
                        print(f"   ✓ Plotted {len(mae_data)} points for Initial Training MAE")
                        
                        if 'val_mae' in history:
                            val_mae_data = history['val_mae']
                            if hasattr(val_mae_data, '__len__') and len(val_mae_data) > 0:
                                ax.plot(epochs, val_mae_data, label='Initial Validation MAE', 
                                       color=self.colors[1], linewidth=2, alpha=0.8, linestyle='--', marker='s', markersize=4)
                                print(f"   ✓ Plotted {len(val_mae_data)} points for Initial Validation MAE")
                    else:
                        print(f"   ✗ MAE data is empty or not iterable")
                else:
                    print(f"   ✗ 'mae' key not found in history")
            else:
                print(f"   ✗ History is not a dict: {type(history)}")

        # 绘制最终训练MAE
        if trainer.training_history.get('final_train'):
            history = trainer.training_history['final_train']['history']
            print(f"   Final train history type: {type(history)}")
            print(f"   Final train history keys: {list(history.keys()) if hasattr(history, 'keys') else 'N/A'}")
            
            if isinstance(history, dict):
                if 'mae' in history:
                    mae_data = history['mae']
                    print(f"   MAE data type: {type(mae_data)}, length: {len(mae_data) if hasattr(mae_data, '__len__') else 'N/A'}")
                    
                    if hasattr(mae_data, '__len__') and len(mae_data) > 0:
                        epochs = range(1, len(mae_data) + 1)
                        ax.plot(epochs, mae_data, label='Final Training MAE', 
                               color=self.colors[2], linewidth=2, alpha=0.8, marker='o', markersize=4)
                        print(f"   ✓ Plotted {len(mae_data)} points for Final Training MAE")
                        
                        if 'val_mae' in history:
                            val_mae_data = history['val_mae']
                            if hasattr(val_mae_data, '__len__') and len(val_mae_data) > 0:
                                ax.plot(epochs, val_mae_data, label='Final Validation MAE', 
                                       color=self.colors[3], linewidth=2, alpha=0.8, linestyle='--', marker='s', markersize=4)
                                print(f"   ✓ Plotted {len(val_mae_data)} points for Final Validation MAE")
                    else:
                        print(f"   ✗ MAE data is empty or not iterable")
                else:
                    print(f"   ✗ 'mae' key not found in history")
            else:
                print(f"   ✗ History is not a dict: {type(history)}")

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax.set_title(f'Training MAE Curve - {device_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = self.generate_plot_filename(device_name, "training_mae_curve")
        self.save_or_show_plot(fig, filename, "training", device_name)

    def plot_learning_rate_curve(self, trainer, device_name):
        """
        绘制学习率变化曲线

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_TRAINING_LR_CURVE:
            return

        print(f"📈 Plotting learning rate curve for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 从超参数调优结果中提取学习率信息
        if trainer.training_history.get('hyperparameter_tuning'):
            tuning_results = trainer.training_history['hyperparameter_tuning']
            lrs = [r['lr'] for r in tuning_results]
            epochs = [r['epochs'] for r in tuning_results]
            losses = [r['val_loss'] for r in tuning_results]

            # 按学习率排序
            sorted_indices = np.argsort(lrs)
            lrs_sorted = [lrs[i] for i in sorted_indices]
            losses_sorted = [losses[i] for i in sorted_indices]

            ax.scatter(lrs_sorted, losses_sorted, s=100, alpha=0.7, 
                      c=range(len(lrs_sorted)), cmap='viridis')
            ax.plot(lrs_sorted, losses_sorted, alpha=0.3, linewidth=1)

            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
            ax.set_title(f'Learning Rate vs Loss - {device_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 标记最佳学习率
            if trainer.training_history.get('best_params'):
                best_lr = trainer.training_history['best_params']['lr']
                best_loss = trainer.training_history['best_val_loss']
                ax.scatter([best_lr], [best_loss], s=200, marker='*', 
                          color='red', edgecolors='black', linewidth=2, zorder=5,
                          label=f'Best LR: {best_lr:.6f}')
                ax.legend(fontsize=10)

        filename = self.generate_plot_filename(device_name, "learning_rate_curve")
        self.save_or_show_plot(fig, filename, "training", device_name)

    # ========================================================================
    # 超参数调优图表
    # ========================================================================

    def plot_hyperparameter_heatmap(self, trainer, device_name):
        """
        绘制超参数热图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_HYPERPARAM_HEATMAP:
            return

        if not trainer.training_history.get('hyperparameter_tuning'):
            return

        print(f"📊 Plotting hyperparameter heatmap for {device_name}...")

        tuning_results = trainer.training_history['hyperparameter_tuning']

        # 提取数据
        lrs = [r.get('lr', 0) for r in tuning_results]
        epochs = [r.get('epochs', 0) for r in tuning_results]
        losses = [r.get('val_loss', 0) for r in tuning_results]

        # 创建数据矩阵
        unique_lrs = sorted(set(lrs))
        unique_epochs = sorted(set(epochs))

        # 创建损失矩阵
        loss_matrix = np.full((len(unique_epochs), len(unique_lrs)), np.nan)

        for r in tuning_results:
            if r.get('epochs') in unique_epochs and r.get('lr') in unique_lrs:
                i = unique_epochs.index(r.get('epochs'))
                j = unique_lrs.index(r.get('lr'))
                loss_matrix[i, j] = r.get('val_loss', 0)

        # 绘制热图
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(loss_matrix, cmap='YlOrRd', aspect='auto')

        # 设置刻度
        ax.set_xticks(range(len(unique_lrs)))
        ax.set_xticklabels([f'{lr:.1e}' for lr in unique_lrs], rotation=45)
        ax.set_yticks(range(len(unique_epochs)))
        ax.set_yticklabels(unique_epochs)

        ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Epochs', fontsize=12, fontweight='bold')
        ax.set_title(f'Hyperparameter Optimization Heatmap - {device_name}', 
                    fontsize=14, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Validation Loss', fontsize=11, fontweight='bold')

        # 标记最佳点
        min_loss_idx = np.unravel_index(np.nanargmin(loss_matrix), loss_matrix.shape)
        ax.scatter(min_loss_idx[1], min_loss_idx[0], color='blue', s=200,
                  marker='*', edgecolors='white', linewidth=2, zorder=5)

        filename = self.generate_plot_filename(device_name, "hyperparam_heatmap")
        self.save_or_show_plot(fig, filename, "training", device_name)

    def plot_hyperparameter_contour(self, trainer, device_name):
        """
        绘制超参数等高线图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_HYPERPARAM_CONTOUR:
            return

        if not trainer.training_history.get('hyperparameter_tuning'):
            return

        print(f"📊 Plotting hyperparameter contour for {device_name}...")

        tuning_results = trainer.training_history['hyperparameter_tuning']

        # 提取数据
        lrs = [r.get('lr', 0) for r in tuning_results]
        epochs = [r.get('epochs', 0) for r in tuning_results]
        losses = [r.get('val_loss', 0) for r in tuning_results]

        # 创建数据矩阵
        unique_lrs = sorted(set(lrs))
        unique_epochs = sorted(set(epochs))

        # 创建损失矩阵
        loss_matrix = np.full((len(unique_epochs), len(unique_lrs)), np.nan)

        for r in tuning_results:
            if r.get('epochs') in unique_epochs and r.get('lr') in unique_lrs:
                i = unique_epochs.index(r.get('epochs'))
                j = unique_lrs.index(r.get('lr'))
                loss_matrix[i, j] = r.get('val_loss', 0)

        # 绘制等高线图
        fig, ax = plt.subplots(figsize=(10, 8))

        # 创建网格
        X, Y = np.meshgrid(np.arange(len(unique_lrs)), np.arange(len(unique_epochs)))

        # 绘制等高线
        contour = ax.contour(X, Y, loss_matrix, levels=10, colors='black', alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=8)

        # 填充等高线
        contourf = ax.contourf(X, Y, loss_matrix, levels=20, cmap='YlOrRd', alpha=0.7)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Validation Loss', fontsize=11, fontweight='bold')

        # 设置刻度
        ax.set_xticks(range(len(unique_lrs)))
        ax.set_xticklabels([f'{lr:.1e}' for lr in unique_lrs], rotation=45)
        ax.set_yticks(range(len(unique_epochs)))
        ax.set_yticklabels(unique_epochs)

        ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Epochs', fontsize=12, fontweight='bold')
        ax.set_title(f'Hyperparameter Optimization Contour - {device_name}', 
                    fontsize=14, fontweight='bold')

        # 标记最佳点
        min_loss_idx = np.unravel_index(np.nanargmin(loss_matrix), loss_matrix.shape)
        ax.scatter(min_loss_idx[1], min_loss_idx[0], color='blue', s=200,
                  marker='*', edgecolors='white', linewidth=2, zorder=5)

        filename = self.generate_plot_filename(device_name, "hyperparam_contour")
        self.save_or_show_plot(fig, filename, "training", device_name)

    def plot_hyperparameter_3d(self, trainer, device_name):
        """
        绘制超参数3D图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_HYPERPARAM_3D:
            return

        if not trainer.training_history.get('hyperparameter_tuning'):
            return

        print(f"📊 Plotting hyperparameter 3D for {device_name}...")

        tuning_results = trainer.training_history['hyperparameter_tuning']

        # 提取数据
        lrs = [r.get('lr', 0) for r in tuning_results]
        epochs = [r.get('epochs', 0) for r in tuning_results]
        losses = [r.get('val_loss', 0) for r in tuning_results]

        # 创建数据矩阵
        unique_lrs = sorted(set(lrs))
        unique_epochs = sorted(set(epochs))

        # 创建损失矩阵
        loss_matrix = np.full((len(unique_epochs), len(unique_lrs)), np.nan)

        for r in tuning_results:
            if r.get('epochs') in unique_epochs and r.get('lr') in unique_lrs:
                i = unique_epochs.index(r.get('epochs'))
                j = unique_lrs.index(r.get('lr'))
                loss_matrix[i, j] = r.get('val_loss', 0)

        # 绘制3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 创建网格
        X, Y = np.meshgrid(np.arange(len(unique_lrs)), np.arange(len(unique_epochs)))

        # 绘制曲面
        surf = ax.plot_surface(X, Y, loss_matrix, cmap='YlOrRd', alpha=0.8)

        # 设置刻度
        ax.set_xticks(range(len(unique_lrs)))
        ax.set_xticklabels([f'{lr:.1e}' for lr in unique_lrs])
        ax.set_yticks(range(len(unique_epochs)))
        ax.set_yticklabels(unique_epochs)

        ax.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('Epochs', fontsize=11, fontweight='bold')
        ax.set_zlabel('Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title(f'Hyperparameter Optimization 3D - {device_name}', 
                    fontsize=14, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Validation Loss', fontsize=10, fontweight='bold')

        filename = self.generate_plot_filename(device_name, "hyperparam_3d")
        self.save_or_show_plot(fig, filename, "training", device_name)

    # ========================================================================
    # 损失分析图表
    # ========================================================================

    def plot_loss_distribution(self, trainer, device_name):
        """
        绘制损失分布图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_LOSS_DISTRIBUTION:
            return

        print(f"📊 Plotting loss distribution for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 收集损失值
        all_losses = []
        labels = []

        if trainer.training_history.get('initial_train'):
            history = trainer.training_history['initial_train']['history']
            if 'loss' in history:
                all_losses.extend(history['loss'])
                labels.extend(['Initial Train'] * len(history['loss']))
            if 'val_loss' in history:
                all_losses.extend(history['val_loss'])
                labels.extend(['Initial Val'] * len(history['val_loss']))

        if trainer.training_history.get('final_train'):
            history = trainer.training_history['final_train']['history']
            if 'loss' in history:
                all_losses.extend(history['loss'])
                labels.extend(['Final Train'] * len(history['loss']))
            if 'val_loss' in history:
                all_losses.extend(history['val_loss'])
                labels.extend(['Final Val'] * len(history['val_loss']))

        if not all_losses:
            return

        # 绘制分布图
        for i, label in enumerate(set(labels)):
            losses = [l for l, lbl in zip(all_losses, labels) if lbl == label]
            ax.hist(losses, bins=30, alpha=0.5, label=label, 
                   color=self.colors[i % len(self.colors)], density=True)

        ax.set_xlabel('Loss Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Loss Distribution - {device_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        filename = self.generate_plot_filename(device_name, "loss_distribution")
        self.save_or_show_plot(fig, filename, "metrics", device_name)

    def plot_loss_histogram(self, trainer, device_name):
        """
        绘制损失直方图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_LOSS_HISTOGRAM:
            return

        print(f"📊 Plotting loss histogram for {device_name}...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 收集损失值
        phases = ['initial_train', 'final_train']
        phase_names = ['Initial Training', 'Final Training']

        for idx, (phase, phase_name) in enumerate(zip(phases, phase_names)):
            if not trainer.training_history.get(phase):
                continue

            history = trainer.training_history[phase]['history']

            # 训练损失直方图
            ax = axes[idx, 0]
            if 'loss' in history:
                ax.hist(history['loss'], bins=30, color=self.colors[idx * 2], 
                       alpha=0.7, edgecolor='black')
                ax.set_xlabel('Training Loss', fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax.set_title(f'{phase_name} - Training Loss', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

            # 验证损失直方图
            ax = axes[idx, 1]
            if 'val_loss' in history:
                ax.hist(history['val_loss'], bins=30, color=self.colors[idx * 2 + 1], 
                       alpha=0.7, edgecolor='black')
                ax.set_xlabel('Validation Loss', fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax.set_title(f'{phase_name} - Validation Loss', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        filename = self.generate_plot_filename(device_name, "loss_histogram")
        self.save_or_show_plot(fig, filename, "metrics", device_name)

    def plot_loss_boxplot(self, trainer, device_name):
        """
        绘制损失箱线图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_LOSS_BOX_PLOT:
            return

        print(f"📊 Plotting loss boxplot for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 收集损失值
        data = []
        labels = []

        if trainer.training_history.get('initial_train'):
            history = trainer.training_history['initial_train']['history']
            if 'loss' in history:
                data.append(history['loss'])
                labels.append('Initial Train')
            if 'val_loss' in history:
                data.append(history['val_loss'])
                labels.append('Initial Val')

        if trainer.training_history.get('final_train'):
            history = trainer.training_history['final_train']['history']
            if 'loss' in history:
                data.append(history['loss'])
                labels.append('Final Train')
            if 'val_loss' in history:
                data.append(history['val_loss'])
                labels.append('Final Val')

        if not data:
            return

        # 绘制箱线图
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # 设置颜色
        for patch, color in zip(bp['boxes'], self.colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Loss Box Plot - {device_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        filename = self.generate_plot_filename(device_name, "loss_boxplot")
        self.save_or_show_plot(fig, filename, "metrics", device_name)

    def plot_loss_violin(self, trainer, device_name):
        """
        绘制损失小提琴图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_LOSS_VIOLIN_PLOT:
            return

        print(f"📊 Plotting loss violin for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 收集损失值
        data = []
        labels = []

        if trainer.training_history.get('initial_train'):
            history = trainer.training_history['initial_train']['history']
            if 'loss' in history:
                data.append(history['loss'])
                labels.append('Initial Train')
            if 'val_loss' in history:
                data.append(history['val_loss'])
                labels.append('Initial Val')

        if trainer.training_history.get('final_train'):
            history = trainer.training_history['final_train']['history']
            if 'loss' in history:
                data.append(history['loss'])
                labels.append('Final Train')
            if 'val_loss' in history:
                data.append(history['val_loss'])
                labels.append('Final Val')

        if not data:
            return

        # 绘制小提琴图
        parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, 
                              showmedians=True, showextrema=True)

        # 设置颜色
        for pc, color in zip(parts['bodies'], self.colors[:len(data)]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Loss Violin Plot - {device_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        filename = self.generate_plot_filename(device_name, "loss_violin")
        self.save_or_show_plot(fig, filename, "metrics", device_name)

    # ========================================================================
    # 模型性能图表
    # ========================================================================

    def plot_performance_metrics(self, trainer, device_name):
        """
        绘制性能指标图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_PERFORMANCE_METRICS:
            return

        print(f"📊 Plotting performance metrics for {device_name}...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 提取数据
        phases = ['initial_train', 'final_train']
        phase_names = ['Initial', 'Final']

        for idx, (phase, phase_name) in enumerate(zip(phases, phase_names)):
            if not trainer.training_history.get(phase):
                continue

            history = trainer.training_history[phase]['history']

            # 损失收敛图
            ax = axes[idx, 0]
            if 'loss' in history:
                epochs = range(1, len(history['loss']) + 1)
                ax.plot(epochs, history['loss'], label='Training Loss', 
                       color=self.colors[idx * 2], linewidth=2)
                if 'val_loss' in history:
                    ax.plot(epochs, history['val_loss'], label='Validation Loss', 
                           color=self.colors[idx * 2 + 1], linewidth=2, linestyle='--')
                ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
                ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
                ax.set_title(f'{phase_name} - Loss Convergence', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            # MAE收敛图
            ax = axes[idx, 1]
            if 'mae' in history:
                epochs = range(1, len(history['mae']) + 1)
                ax.plot(epochs, history['mae'], label='Training MAE', 
                       color=self.colors[idx * 2], linewidth=2)
                if 'val_mae' in history:
                    ax.plot(epochs, history['val_mae'], label='Validation MAE', 
                           color=self.colors[idx * 2 + 1], linewidth=2, linestyle='--')
                ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
                ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
                ax.set_title(f'{phase_name} - MAE Convergence', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = self.generate_plot_filename(device_name, "performance_metrics")
        self.save_or_show_plot(fig, filename, "metrics", device_name)

    def plot_learning_rate_schedule(self, trainer, device_name):
        """
        绘制学习率调度图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_LEARNING_RATE_SCHEDULE:
            return

        print(f"📊 Plotting learning rate schedule for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 从超参数调优结果中提取学习率信息
        if trainer.training_history.get('hyperparameter_tuning'):
            tuning_results = trainer.training_history['hyperparameter_tuning']

            # 按顺序绘制学习率变化
            for i, result in enumerate(tuning_results):
                ax.scatter(i, result['lr'], s=100, alpha=0.7, 
                          color=self.colors[i % len(self.colors)])
                ax.annotate(f"LR={result['lr']:.1e}", 
                          (i, result['lr']), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)

            # 标记最佳学习率
            if trainer.training_history.get('best_params'):
                best_idx = next(i for i, r in enumerate(tuning_results) 
                              if r['lr'] == trainer.training_history['best_params']['lr'])
                ax.scatter(best_idx, tuning_results[best_idx]['lr'], 
                          s=200, marker='*', color='red', 
                          edgecolors='black', linewidth=2, zorder=5,
                          label='Best LR')

            ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.set_title(f'Learning Rate Schedule - {device_name}', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        filename = self.generate_plot_filename(device_name, "lr_schedule")
        self.save_or_show_plot(fig, filename, "metrics", device_name)

    # ========================================================================
    # 数据分析图表
    # ========================================================================

    def plot_data_distribution(self, data_info, device_name):
        """
        绘制数据分布图

        Args:
            data_info: 数据信息字典
            device_name: 设备名称
        """
        if not self.config.PLOT_DATA_DISTRIBUTION:
            return

        print(f"📊 Plotting data distribution for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 提取数据集大小
        train_samples = data_info.get('train_samples', 0)
        val_samples = data_info.get('val_samples', 0)
        test_samples = data_info.get('test_samples', 0)

        # 创建饼图
        sizes = [train_samples, val_samples, test_samples]
        labels = ['Training', 'Validation', 'Test']
        colors = [self.colors[0], self.colors[1], self.colors[2]]
        explode = (0.05, 0.05, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            explode=explode, colors=colors, startangle=90
        )

        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(f'Data Distribution - {device_name}', 
                    fontsize=14, fontweight='bold')

        # 添加数据标签
        info_text = f"Total Samples: {data_info.get('n_samples', 0)}\n"
        info_text += f"Features: {data_info.get('n_features', 0)}\n"
        info_text += f"Train: {train_samples}\n"
        info_text += f"Val: {val_samples}\n"
        info_text += f"Test: {test_samples}"

        ax.text(1.3, 0.5, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        filename = self.generate_plot_filename(device_name, "data_distribution")
        self.save_or_show_plot(fig, filename, "data", device_name)

    # ========================================================================
    # 时间分析图表
    # ========================================================================

    def plot_training_time_analysis(self, trainer, device_name):
        """
        绘制训练时间分析

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_TRAINING_TIME_ANALYSIS:
            return

        print(f"📊 Plotting training time analysis for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 收集时间数据
        times = []
        labels = []

        if trainer.training_history.get('initial_train'):
            times.append(trainer.training_history['initial_train'].get('training_time', 0))
            labels.append('Initial Training')

        if trainer.training_history.get('hyperparameter_tuning'):
            tuning_times = [r.get('training_time', 0) for r in 
                          trainer.training_history['hyperparameter_tuning']]
            if tuning_times:
                times.append(sum(tuning_times))
                labels.append('Hyperparameter Tuning')

        if trainer.training_history.get('final_train'):
            times.append(trainer.training_history['final_train'].get('training_time', 0))
            labels.append('Final Training')

        if not times:
            return

        # 绘制条形图
        bars = ax.bar(labels, times, color=self.colors[:len(times)], 
                     alpha=0.7, edgecolor='black')

        # 添加数值标签
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + height * 0.01,
                   f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Training Time Analysis - {device_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        filename = self.generate_plot_filename(device_name, "training_time_analysis")
        self.save_or_show_plot(fig, filename, "metrics", device_name)

    def plot_epoch_time_distribution(self, trainer, device_name):
        """
        绘制Epoch时间分布

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_EPOCH_TIME_DISTRIBUTION:
            return

        print(f"📊 Plotting epoch time distribution for {device_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 收集时间数据
        times = []
        labels = []

        if trainer.training_history.get('initial_train'):
            train_time = trainer.training_history['initial_train'].get('training_time', 0)
            history = trainer.training_history['initial_train']['history']
            num_epochs = len(history.get('loss', []))
            if num_epochs > 0:
                times.append(train_time / num_epochs)
                labels.append('Initial Training')

        if trainer.training_history.get('final_train'):
            train_time = trainer.training_history['final_train'].get('training_time', 0)
            history = trainer.training_history['final_train']['history']
            num_epochs = len(history.get('loss', []))
            if num_epochs > 0:
                times.append(train_time / num_epochs)
                labels.append('Final Training')

        if not times:
            return

        # 绘制条形图
        bars = ax.bar(labels, times, color=self.colors[:len(times)], 
                     alpha=0.7, edgecolor='black')

        # 添加数值标签
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + height * 0.01,
                   f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Average Time per Epoch (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Epoch Time Distribution - {device_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        filename = self.generate_plot_filename(device_name, "epoch_time_distribution")
        self.save_or_show_plot(fig, filename, "metrics", device_name)

    # ========================================================================
    # 比较图表
    # ========================================================================

    def plot_device_comparison(self, all_results, device_name="comparison"):
        """
        绘制设备比较图

        Args:
            all_results: 所有设备的训练结果
            device_name: 设备名称（用于文件名）
        """
        if not self.config.PLOT_DEVICE_COMPARISON or len(all_results) <= 1:
            return

        print(f"📊 Plotting device comparison for {len(all_results)} devices...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 提取数据
        device_names = [r['device_name'] for r in all_results]
        losses = [r.get('best_val_loss', 0) for r in all_results]
        times = [r.get('training_time', 0) for r in all_results]

        # 损失比较
        ax = axes[0, 0]
        bars = ax.bar(range(len(device_names)), losses, 
                     color=self.colors[:len(device_names)], alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(device_names)))
        ax.set_xticklabels(device_names, rotation=45, ha='right')
        ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Device Loss Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 时间比较
        ax = axes[0, 1]
        bars = ax.bar(range(len(device_names)), times, 
                     color=self.colors[:len(device_names)], alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(device_names)))
        ax.set_xticklabels(device_names, rotation=45, ha='right')
        ax.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
        ax.set_title('Device Time Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 散点图
        ax = axes[1, 0]
        scatter = ax.scatter(times, losses, c=self.colors[:len(device_names)], 
                          s=100, alpha=0.7, edgecolors='black')
        for i, name in enumerate(device_names):
            ax.annotate(name, (times[i], losses[i]), 
                      xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.set_xlabel('Training Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Time vs Loss Scatter', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 效率指标
        ax = axes[1, 1]
        efficiency = [1.0 / (l + 0.0001) for l in losses]
        bars = ax.bar(range(len(device_names)), efficiency, 
                     color=self.colors[:len(device_names)], alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(device_names)))
        ax.set_xticklabels(device_names, rotation=45, ha='right')
        ax.set_ylabel('Efficiency (1/Loss)', fontsize=11, fontweight='bold')
        ax.set_title('Device Efficiency', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        filename = self.generate_plot_filename(device_name, "device_comparison")
        self.save_or_show_plot(fig, filename, "comparison", device_name=None)

    def plot_phase_comparison(self, trainer, device_name):
        """
        绘制训练阶段比较

        Args:
            trainer: 训练器对象
            device_name: 设备名称
        """
        if not self.config.PLOT_PHASE_COMPARISON:
            return

        print(f"📊 Plotting phase comparison for {device_name}...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 收集数据
        phases = []
        train_losses = []
        val_losses = []

        if trainer.training_history.get('initial_train'):
            history = trainer.training_history['initial_train']['history']
            if 'loss' in history:
                phases.append('Initial')
                train_losses.append(history['loss'][-1])
                if 'val_loss' in history:
                    val_losses.append(min(history['val_loss']))
                else:
                    val_losses.append(history['loss'][-1])

        if trainer.training_history.get('final_train'):
            history = trainer.training_history['final_train']['history']
            if 'loss' in history:
                phases.append('Final')
                train_losses.append(history['loss'][-1])
                if 'val_loss' in history:
                    val_losses.append(min(history['val_loss']))
                else:
                    val_losses.append(history['loss'][-1])

        if not phases:
            return

        # 条形图
        ax = axes[0]
        x = np.arange(len(phases))
        width = 0.35

        bars1 = ax.bar(x - width / 2, train_losses, width,
                      label='Training Loss', color=self.colors[0], alpha=0.7)
        bars2 = ax.bar(x + width / 2, val_losses, width,
                      label='Validation Loss', color=self.colors[1], alpha=0.7)

        ax.set_xlabel('Training Phase', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax.set_title('Phase Loss Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + height * 0.01,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        # 改进百分比
        ax = axes[1]
        if len(train_losses) >= 2:
            improvement = ((train_losses[0] - train_losses[1]) / train_losses[0]) * 100
            val_improvement = ((val_losses[0] - val_losses[1]) / val_losses[0]) * 100

            categories = ['Training Loss', 'Validation Loss']
            improvements = [improvement, val_improvement]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]

            bars = ax.bar(categories, improvements, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
            ax.set_title('Loss Improvement', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + (1 if height > 0 else -1),
                       f'{imp:.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

        plt.tight_layout()

        filename = self.generate_plot_filename(device_name, "phase_comparison")
        self.save_or_show_plot(fig, filename, "training", device_name)

    def plot_performance_ranking(self, all_results, device_name="ranking"):
        """
        绘制性能排名图

        Args:
            all_results: 所有设备的训练结果
            device_name: 设备名称（用于文件名）
        """
        if not self.config.PLOT_PERFORMANCE_RANKING or len(all_results) <= 1:
            return

        print(f"📊 Plotting performance ranking for {len(all_results)} devices...")

        # 按损失值排序（越小越好）
        sorted_results = sorted(all_results, key=lambda x: x.get('best_val_loss', float('inf')))
        sorted_losses = [r.get('best_val_loss', 0) for r in sorted_results]
        sorted_names = [r['device_name'] for r in sorted_results]

        fig, ax = plt.subplots(figsize=(10, 6))

        # 创建水平条形图
        y_pos = np.arange(len(sorted_names))

        bars = ax.barh(y_pos, sorted_losses, color=self.colors[:len(sorted_names)], alpha=0.7)

        # 添加数值标签
        for i, (bar, loss) in enumerate(zip(bars, sorted_losses)):
            width = bar.get_width()
            ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                   f'{loss:.4f}', ha='left', va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Validation Loss', fontsize=12, fontweight='bold')
        ax.set_title('Performance Ranking', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 添加排名数字
        for i in range(len(sorted_names)):
            ax.text(-0.1, i, f'#{i + 1}', ha='right', va='center',
                   fontsize=10, fontweight='bold', transform=ax.get_yaxis_transform())

        filename = self.generate_plot_filename(device_name, "performance_ranking")
        self.save_or_show_plot(fig, filename, "comparison", device_name=None)

    # ========================================================================
    # 综合报告图表
    # ========================================================================

    def plot_comprehensive_summary(self, trainer, device_name, data_info):
        """
        绘制综合总结图

        Args:
            trainer: 训练器对象
            device_name: 设备名称
            data_info: 数据信息
        """
        if not self.config.PLOT_COMPREHENSIVE_SUMMARY:
            return

        print(f"📊 Plotting comprehensive summary for {device_name}...")

        rows, cols = self.config.PLOT_SUMMARY_GRID
        fig = plt.figure(figsize=(cols * 5, rows * 4))
        fig.suptitle(f'Autoencoder Training Summary - {device_name}',
                     fontsize=18, fontweight='bold', y=0.98)

        gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.3)

        plot_count = 0

        # 训练损失曲线
        if trainer.training_history.get('initial_train'):
            ax = fig.add_subplot(gs[plot_count // cols, plot_count % cols])
            history = trainer.training_history['initial_train']['history']
            epochs = range(1, len(history['loss']) + 1)
            ax.plot(epochs, history['loss'], label='Training Loss', 
                   color=self.colors[0], linewidth=2)
            if 'val_loss' in history:
                ax.plot(epochs, history['val_loss'], label='Validation Loss', 
                       color=self.colors[1], linewidth=2, linestyle='--')
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax.set_title('Initial Training Loss', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plot_count += 1

        # 超参数热图
        if trainer.training_history.get('hyperparameter_tuning') and plot_count < rows * cols:
            ax = fig.add_subplot(gs[plot_count // cols, plot_count % cols])
            tuning_results = trainer.training_history['hyperparameter_tuning']
            lrs = sorted(set([r['lr'] for r in tuning_results]))
            epochs = sorted(set([r['epochs'] for r in tuning_results]))
            loss_matrix = np.full((len(epochs), len(lrs)), np.nan)
            for r in tuning_results:
                i = epochs.index(r['epochs'])
                j = lrs.index(r['lr'])
                loss_matrix[i, j] = r['val_loss']
            im = ax.imshow(loss_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(lrs)))
            ax.set_xticklabels([f'{lr:.1e}' for lr in lrs], rotation=45)
            ax.set_yticks(range(len(epochs)))
            ax.set_yticklabels(epochs)
            ax.set_title('Hyperparameter Heatmap', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Loss')
            plot_count += 1

        # 训练时间分析
        if plot_count < rows * cols:
            ax = fig.add_subplot(gs[plot_count // cols, plot_count % cols])
            times = []
            labels = []
            if trainer.training_history.get('initial_train'):
                times.append(trainer.training_history['initial_train'].get('training_time', 0))
                labels.append('Initial')
            if trainer.training_history.get('final_train'):
                times.append(trainer.training_history['final_train'].get('training_time', 0))
                labels.append('Final')
            if times:
                bars = ax.bar(labels, times, color=self.colors[:len(times)], alpha=0.7)
                for bar, t in zip(bars, times):
                    ax.text(bar.get_x() + bar.get_width() / 2, t + t * 0.01,
                           f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
                ax.set_ylabel('Time (s)', fontsize=11, fontweight='bold')
                ax.set_title('Training Time', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                plot_count += 1

        # 数据信息
        if plot_count < rows * cols:
            ax = fig.add_subplot(gs[plot_count // cols, plot_count % cols])
            ax.axis('off')
            info_text = f"Device: {device_name}\n"
            info_text += f"Samples: {data_info.get('n_samples', 'N/A')}\n"
            info_text += f"Features: {data_info.get('n_features', 'N/A')}\n"
            info_text += f"Train: {data_info.get('train_samples', 'N/A')}\n"
            info_text += f"Val: {data_info.get('val_samples', 'N/A')}\n"
            info_text += f"Test: {data_info.get('test_samples', 'N/A')}"
            ax.text(0.1, 0.5, info_text, fontsize=10, va='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax.set_title('Data Information', fontsize=12, fontweight='bold')
            plot_count += 1

        # 最佳参数
        if trainer.training_history.get('best_params') and plot_count < rows * cols:
            ax = fig.add_subplot(gs[plot_count // cols, plot_count % cols])
            ax.axis('off')
            best_params = trainer.training_history['best_params']
            param_text = f"Best LR: {best_params['lr']:.6f}\n"
            param_text += f"Best Epochs: {best_params['epochs']}\n"
            param_text += f"Best Loss: {trainer.training_history['best_val_loss']:.6f}"
            ax.text(0.1, 0.5, param_text, fontsize=10, va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            ax.set_title('Best Parameters', fontsize=12, fontweight='bold')
            plot_count += 1

        plt.tight_layout()

        filename = self.generate_plot_filename(device_name, "comprehensive_summary")
        self.save_or_show_plot(fig, filename, "training", device_name)

    def generate_all_plots(self, trainer, device_name, data_info):
        """
        生成所有图表

        Args:
            trainer: 训练器对象
            device_name: 设备名称
            data_info: 数据信息
        """
        print(f"📊 Generating all plots for {device_name}...")
        
        # 调试：打印训练历史结构
        print(f"📊 Training history keys: {list(trainer.training_history.keys())}")
        for key, value in trainer.training_history.items():
            if value is not None and isinstance(value, dict):
                print(f"   {key}: {list(value.keys())}")
            elif value is not None and isinstance(value, list):
                print(f"   {key}: list with {len(value)} items")
            else:
                print(f"   {key}: {type(value).__name__}")

        # 训练曲线图表
        self.plot_training_loss_curve(trainer, device_name)
        self.plot_training_mae_curve(trainer, device_name)
        self.plot_learning_rate_curve(trainer, device_name)

        # 超参数调优图表
        self.plot_hyperparameter_heatmap(trainer, device_name)
        self.plot_hyperparameter_contour(trainer, device_name)
        self.plot_hyperparameter_3d(trainer, device_name)

        # 损失分析图表
        self.plot_loss_distribution(trainer, device_name)
        self.plot_loss_histogram(trainer, device_name)
        self.plot_loss_boxplot(trainer, device_name)
        self.plot_loss_violin(trainer, device_name)

        # 模型性能图表
        self.plot_performance_metrics(trainer, device_name)
        self.plot_learning_rate_schedule(trainer, device_name)

        # 数据分析图表
        self.plot_data_distribution(data_info, device_name)

        # 时间分析图表
        self.plot_training_time_analysis(trainer, device_name)
        self.plot_epoch_time_distribution(trainer, device_name)

        # 比较图表
        self.plot_phase_comparison(trainer, device_name)

        # 综合报告图表
        self.plot_comprehensive_summary(trainer, device_name, data_info)

        print(f"✅ All plots generated for {device_name}")
