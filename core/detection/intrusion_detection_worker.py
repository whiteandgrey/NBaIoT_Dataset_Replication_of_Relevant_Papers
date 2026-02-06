import os
import time
import json
import numpy as np
from datetime import datetime
from PyQt5.QtCore import QThread

from config import Config
from anomaly_detector import AnomalyDetector
from data_integrator import DStstIntegrator
from core.signals import IntrusionDetectionSignals


class IntrusionDetectionWorker(QThread):
    """
    入侵检测与评估工作线程 - 在后台执行评估任务
    """
    
    def __init__(self, config: dict, signals: IntrusionDetectionSignals):
        """
        初始化入侵检测工作线程
        
        Args:
            config: 评估配置
            signals: 入侵检测信号对象
        """
        super().__init__()
        self.config = config
        self.signals = signals
        self.is_running = False
        self.should_stop = False
    
    def run(self):
        """
        执行评估
        """
        self.is_running = True
        self.should_stop = False
        
        try:
            # 获取配置参数
            device_name = self.config.get('device_name', 'Danmini_Doorbell')
            dstst_data_file = self.config.get('dstst_data_file', '')
            dstst_labels_file = self.config.get('dstst_labels_file', '')
            model_file = self.config.get('model_file', '')
            save_path = self.config.get('save_path', os.path.join(Config.OUTPUT_DIR, 'intrusion_detection'))
            save_data = self.config.get('save_data', True)
            save_images = self.config.get('save_images', True)
            
            # 获取滑动窗口配置
            min_window_size = self.config.get('min_window_size', Config.MIN_WINDOW_SIZE)
            max_window_size = self.config.get('max_window_size', Config.MAX_WINDOW_SIZE)
            window_size_step = self.config.get('window_size_step', Config.WINDOW_SIZE_STEP)
            
            # 更新Config中的滑动窗口配置
            Config.MIN_WINDOW_SIZE = min_window_size
            Config.MAX_WINDOW_SIZE = max_window_size
            Config.WINDOW_SIZE_STEP = window_size_step

            # 验证文件
            if not dstst_data_file or not dstst_labels_file:
                self.signals.error.emit("请选择DStst文件（数据文件和标签文件）")
                self.is_running = False
                return

            if not model_file:
                self.signals.error.emit("请选择模型文件")
                self.is_running = False
                return

            if not os.path.exists(dstst_data_file):
                self.signals.error.emit(f"DStst数据文件不存在: {dstst_data_file}")
                self.is_running = False
                return

            if not os.path.exists(dstst_labels_file):
                self.signals.error.emit(f"DStst标签文件不存在: {dstst_labels_file}")
                self.is_running = False
                return

            # 加载DStst文件
            self.signals.log.emit("正在加载DStst文件...")
            self.signals.status_update.emit("正在加载DStst文件")

            try:
                X_test = np.load(dstst_data_file)
                y_test = np.load(dstst_labels_file)

                if len(X_test) != len(y_test):
                    self.signals.error.emit(f"数据和标签数量不匹配: 数据={len(X_test)}, 标签={len(y_test)}")
                    self.is_running = False
                    return

                self.signals.log.emit(f"✅ DStst文件加载成功: {len(X_test)} 个样本")
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

            # 加载训练好的模型和scaler
            self.signals.log.emit("正在加载训练好的模型和scaler...")
            self.signals.status_update.emit("正在加载训练好的模型和scaler")

            model_path = os.path.join(os.path.dirname(dstst_data_file), "final_model.h5")
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(dstst_data_file), "best_model.h5")

            scaler_path = os.path.join(os.path.dirname(dstst_data_file), "scaler.pkl")

            detector.load_model(model_path)
            detector.load_scaler(scaler_path)

            # 加载DSopt数据用于计算阈值和窗口大小
            self.signals.log.emit("正在加载DSopt数据用于计算阈值和窗口大小...")
            self.signals.status_update.emit("正在加载DSopt数据用于计算阈值和窗口大小")

            from data_processor import NBaIoTDataProcessor
            data_processor = NBaIoTDataProcessor(Config)

            benign_data = data_processor.load_device_data(device_name)
            if benign_data is None:
                raise ValueError(f"Failed to load benign data for device: {device_name}")

            DStrn, DSopt, DStst_benign = data_processor.split_data_chronologically(benign_data)

            # 计算异常阈值 tr*
            self.signals.log.emit("正在计算异常阈值 tr*...")
            self.signals.status_update.emit("正在计算异常阈值 tr*")

            tr_threshold = detector.calculate_anomaly_threshold(DSopt)

            # 优化滑动窗口大小 ws*
            self.signals.log.emit("正在优化滑动窗口大小 ws*...")
            self.signals.status_update.emit("正在优化滑动窗口大小 ws*")

            ws_threshold = detector.optimize_window_size(DSopt, tr_threshold)

            self.signals.log.emit(f"✅ 阈值计算完成: tr*={tr_threshold:.6f}, ws*={ws_threshold}")
            
            # 开始评估
            self.signals.log.emit("开始评估入侵检测性能...")
            self.signals.status_update.emit("正在评估入侵检测性能")

            # 使用AnomalyDetector进行评估
            self.signals.log.emit("开始评估入侵检测性能...")
            self.signals.status_update.emit("正在评估入侵检测性能")

            # 分批处理数据以实现实时更新
            total_samples = len(X_test)
            batch_size = 32
            all_predictions = []
            all_mse_values = []

            # 分批处理数据
            for i in range(0, total_samples, batch_size):
                if self.should_stop:
                    break

                end_idx = min(i + batch_size, total_samples)
                batch_X = X_test[i:end_idx]
                batch_y = y_test[i:end_idx]

                # 计算MSE
                mse_values = detector.calculate_reconstruction_error(batch_X)

                # 应用阈值和滑动窗口
                initial_decisions = (mse_values > tr_threshold).astype(int)

                # 应用滑动窗口多数投票
                final_decisions = []
                n_batch_samples = len(initial_decisions)

                for j in range(n_batch_samples):
                    start = max(0, j - ws_threshold + 1)
                    window = initial_decisions[start:j+1]
                    if len(window) >= ws_threshold // 2:
                        window_decision = 1 if sum(window) > len(window) / 2 else 0
                    else:
                        window_decision = initial_decisions[j]
                    final_decisions.append(window_decision)

                final_decisions = np.array(final_decisions)

                # 记录结果
                for j in range(len(final_decisions)):
                    all_predictions.append(int(final_decisions[j]))
                    all_mse_values.append(float(mse_values[j]))

                # 计算当前批次的性能指标
                current_y = y_test[i:end_idx]
                current_preds = final_decisions

                from sklearn.metrics import confusion_matrix

                # 计算混淆矩阵，处理特殊情况
                try:
                    cm = confusion_matrix(current_y, current_preds)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                    elif cm.shape == (1, 1):
                        # 只有一类样本的情况
                        if len(np.unique(current_y)) == 1:
                            # 只有良性样本
                            tn = int(len(current_preds) - np.sum(current_preds))
                            fp = int(np.sum(current_preds))
                            fn = 0
                            tp = 0
                        else:
                            # 只有攻击样本
                            tn = 0
                            fp = 0
                            fn = int(len(current_preds) - np.sum(current_preds))
                            tp = int(np.sum(current_preds))
                    else:
                        # 其他情况，使用默认值
                        tn, fp, fn, tp = 0, 0, 0, 0
                except Exception as e:
                    # 混淆矩阵计算失败，使用默认值
                    print(f"Warning: Confusion matrix calculation failed: {e}")
                    tn, fp, fn, tp = 0, 0, 0, 0

                accuracy = float((tp + tn) / len(current_preds)) if len(current_preds) > 0 else 0.0
                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
                fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

                # 计算进度
                progress = (end_idx / total_samples) * 100

                # 发送数据更新信号
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

                # 发送进度更新信号
                self.signals.progress.emit({'progress': progress})

            # 计算最终的性能指标
            all_predictions = np.array(all_predictions)
            all_mse_values = np.array(all_mse_values)

            from sklearn.metrics import confusion_matrix

            # 计算混淆矩阵，处理特殊情况
            try:
                cm = confusion_matrix(y_test, all_predictions)

                # 确保混淆矩阵是2x2的形状
                if cm.shape == (2, 2):
                    # 使用tolist()确保返回Python列表
                    cm_list = cm.tolist()
                    tn, fp, fn, tp = cm_list[0][0], cm_list[0][1], cm_list[1][0], cm_list[1][1]
                    # 确保所有值都是Python标量
                    tn = int(tn)
                    fp = int(fp)
                    fn = int(fn)
                    tp = int(tp)
                else:
                    # 处理特殊情况：只有一类样本
                    # 计算实际的混淆矩阵元素
                    tp = int(np.sum((y_test == 1) & (all_predictions == 1)))
                    tn = int(np.sum((y_test == 0) & (all_predictions == 0)))
                    fp = int(np.sum((y_test == 0) & (all_predictions == 1)))
                    fn = int(np.sum((y_test == 1) & (all_predictions == 0)))

                    print(f"Confusion matrix shape: {cm.shape}, calculated: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            except Exception as e:
                # 混淆矩阵计算失败，使用默认值
                print(f"Warning: Final confusion matrix calculation failed: {e}")
                tp, tn, fp, fn = 0, 0, 0, 0

            # 计算性能指标
            accuracy = float((tp + tn) / len(all_predictions)) if len(all_predictions) > 0 else 0.0
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            self.signals.log.emit(f"✅ 评估完成:")
            self.signals.log.emit(f"   准确率: {accuracy:.4f}")
            self.signals.log.emit(f"   精确率: {precision:.4f}")
            self.signals.log.emit(f"   召回率: {recall:.4f}")
            self.signals.log.emit(f"   F1分数: {f1:.4f}")
            self.signals.log.emit(f"   误报率: {fpr:.4f}")
            self.signals.log.emit(f"   混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

            # 保存性能结果用于图表生成
            performance = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'fpr': float(fpr),
                'roc_auc': 0.0,
                'predictions': all_predictions,
                'mse_values': all_mse_values
            }
            
            # 保存结果
            if save_data or save_images:
                os.makedirs(save_path, exist_ok=True)

                # 保存数据
                if save_data:
                    results_file = os.path.join(save_path, f'{device_name}_detection_results.json')

                    # 转换numpy类型
                    def convert_numpy_types(obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(item) for item in obj]
                        else:
                            return obj

                    # 保存结果
                    evaluation_results = {
                        'device_name': device_name,
                        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'tr_threshold': tr_threshold,
                        'ws_threshold': ws_threshold,
                        'performance': {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'fpr': fpr,
                            'roc_auc': performance.get('roc_auc', 0.0)
                        },
                        'confusion_matrix': {
                            'tp': int(tp),
                            'tn': int(tn),
                            'fp': int(fp),
                            'fn': int(fn)
                        },
                        'dataset_statistics': {
                            'total_samples': len(X_test),
                            'benign_samples': int(np.sum(y_test == 0)),
                            'attack_samples': int(np.sum(y_test == 1))
                        }
                    }

                    evaluation_results = convert_numpy_types(evaluation_results)

                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

                    self.signals.log.emit(f"✅ 已保存检测结果: {results_file}")

                # 保存图片
                if save_images:
                    self.signals.log.emit("正在生成性能指标图...")

                    import matplotlib.pyplot as plt

                    # 创建图表保存目录
                    charts_dir = os.path.join(save_path, 'charts')
                    os.makedirs(charts_dir, exist_ok=True)

                    # 1. 性能指标条形图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    values = [accuracy, precision, recall, f1]
                    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
                    bars = ax.bar(metrics, values, color=colors)
                    ax.set_ylabel('Score')
                    ax.set_title('Performance Metrics')
                    ax.set_ylim(0, 1.0)
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}',
                                ha='center', va='bottom')
                    plt.tight_layout()
                    metrics_chart_path = os.path.join(charts_dir, f'{device_name}_performance_metrics.png')
                    plt.savefig(metrics_chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.signals.log.emit(f"✅ 已保存性能指标图: {metrics_chart_path}")

                    # 2. 混淆矩阵热图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = np.array([[tn, fp], [fn, tp]])
                    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    ax.set(xticks=[0, 1], yticks=[0, 1],
                           xticklabels=['Predicted Normal', 'Predicted Attack'],
                           yticklabels=['Actual Normal', 'Actual Attack'])
                    ax.set_xlabel('Predicted Label')
                    ax.set_ylabel('True Label')
                    ax.set_title('Confusion Matrix')

                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, format(cm[i, j], 'd'),
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > thresh else "black")

                    plt.tight_layout()
                    cm_chart_path = os.path.join(charts_dir, f'{device_name}_confusion_matrix.png')
                    plt.savefig(cm_chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.signals.log.emit(f"✅ 已保存混淆矩阵图: {cm_chart_path}")

                    # 3. ROC曲线
                    from sklearn.metrics import roc_curve, auc

                    mse_values = performance.get('mse_values', [])
                    if len(mse_values) > 0:
                        roc_fpr, roc_tpr, thresholds = roc_curve(y_test, mse_values)
                        roc_auc = auc(roc_fpr, roc_tpr)

                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.plot(roc_fpr, roc_tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                        ax.legend(loc="lower right")
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        roc_chart_path = os.path.join(charts_dir, f'{device_name}_roc_curve.png')
                        plt.savefig(roc_chart_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        self.signals.log.emit(f"✅ 已保存ROC曲线图: {roc_chart_path}")

                    self.signals.log.emit(f"✅ 所有图表已保存到: {charts_dir}")
            
            # 发送完成信号
            self.signals.completed.emit({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'fpr': float(fpr),
                'total_samples': int(total_samples),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            })
            
        except Exception as e:
            import traceback
            error_msg = f"评估错误: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        finally:
            self.is_running = False
    
    def stop(self):
        """
        停止评估
        """
        self.should_stop = True
        self.signals.status_update.emit("正在停止...")
