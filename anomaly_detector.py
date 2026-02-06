"""
异常检测模块 - 阈值计算、滑动窗口优化、性能评估
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
import matplotlib.pyplot as plt


class AnomalyDetector:
    """异常检测器类"""

    def __init__(self, config):
        """
        初始化异常检测器

        Args:
            config: 配置对象
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.tr_threshold = None
        self.ws_threshold = None

    def load_model(self, model_path):
        """
        加载训练好的自编码器模型

        Args:
            model_path: 模型文件路径

        Returns:
            加载的Keras模型
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
        return self.model

    def load_scaler(self, scaler_path):
        """
        加载训练好的scaler

        Args:
            scaler_path: scaler文件路径

        Returns:
            加载的scaler对象
        """
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        import joblib
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from: {scaler_path}")
        return self.scaler

    def calculate_reconstruction_error(self, data):
        """
        计算重建误差（MSE）

        Args:
            data: 输入数据

        Returns:
            重建误差数组
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.scaler is not None:
            data = self.scaler.transform(data)

        # 预测重建数据
        reconstructed = self.model.predict(data, verbose=0)
        
        # 计算每个样本的MSE
        mse = np.mean(np.power(data - reconstructed, 2), axis=1)
        
        print(f"📊 Reconstruction error calculated: {len(mse)} samples")
        print(f"   MSE stats: min={mse.min():.6f}, max={mse.max():.6f}, mean={mse.mean():.6f}, std={mse.std():.6f}")
        
        return mse

    def calculate_anomaly_threshold(self, dsopt_data):
        """
        计算异常阈值 tr*

        Args:
            dsopt_data: DSopt数据集

        Returns:
            异常阈值 tr*
        """
        print(f"\n{'=' * 60}")
        print("CALCULATING ANOMALY THRESHOLD (tr*)")
        print(f"{'=' * 60}")

        # 计算DSopt上的MSE
        mse_values = self.calculate_reconstruction_error(dsopt_data)

        # 计算均值和标准差
        mean_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)

        # 计算阈值
        self.tr_threshold = mean_mse + std_mse

        print(f"📊 DSopt MSE statistics:")
        print(f"   Mean: {mean_mse:.6f}")
        print(f"   Std: {std_mse:.6f}")
        print(f"   Calculated threshold tr*: {self.tr_threshold:.6f}")

        return self.tr_threshold

    def calculate_fpr_with_window(self, anomaly_decisions, true_labels, window_size):
        """
        使用滑动窗口计算误报率（FPR）

        Args:
            anomaly_decisions: 初始异常决策（0/1）
            true_labels: 真实标签（0=良性，1=恶意）
            window_size: 滑动窗口大小

        Returns:
            误报率（FPR）
        """
        if len(anomaly_decisions) != len(true_labels):
            raise ValueError("anomaly_decisions and true_labels must have the same length")

        windowed_decisions = []
        n_samples = len(anomaly_decisions)

        # 应用滑动窗口多数投票
        for i in range(n_samples):
            start = max(0, i - window_size + 1)
            window = anomaly_decisions[start:i+1]
            if len(window) >= window_size // 2:
                # 多数投票
                window_decision = 1 if sum(window) > len(window) / 2 else 0
            else:
                # 窗口太小，直接使用当前决策
                window_decision = anomaly_decisions[i]
            windowed_decisions.append(window_decision)

        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, windowed_decisions)
        
        # 处理特殊情况：如果混淆矩阵不是2x2，手动计算
        if cm.shape == (1, 1):
            # 所有预测都是0（良性）
            tn = cm[0, 0]
            fp = 0
            fn = 0
            tp = 0
        else:
            # 正常情况：2x2混淆矩阵
            tn, fp, fn, tp = cm.ravel()

        # 计算误报率
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return fpr

    def optimize_window_size(self, dsopt_data, tr_threshold):
        """
        寻找最小的窗口大小 ws*，使得在DSopt上实现0%的误报率

        Args:
            dsopt_data: DSopt数据集（全部为良性数据）
            tr_threshold: 异常阈值

        Returns:
            最优窗口大小 ws*
        """
        print(f"\n{'=' * 60}")
        print("OPTIMIZING WINDOW SIZE (ws*)")
        print(f"{'=' * 60}")

        # 计算DSopt上的MSE
        mse_values = self.calculate_reconstruction_error(dsopt_data)

        # 生成初始异常决策（>tr*=1，否则=0）
        anomaly_decisions = (mse_values > tr_threshold).astype(int)

        # DSopt全部是良性数据，真实标签全为0
        true_labels = np.zeros(len(dsopt_data), dtype=int)

        print(f"📊 Initial anomaly detection on DSopt:")
        print(f"   Total samples: {len(dsopt_data)}")
        print(f"   Initial anomaly candidates: {sum(anomaly_decisions)} ({sum(anomaly_decisions)/len(dsopt_data)*100:.2f}%)")

        # 寻找最小的窗口大小
        max_window_size = min(self.config.MAX_WINDOW_SIZE, len(dsopt_data))
        best_window_size = max_window_size

        for window_size in range(self.config.MIN_WINDOW_SIZE, max_window_size + 1, self.config.WINDOW_SIZE_STEP):
            fpr = self.calculate_fpr_with_window(anomaly_decisions, true_labels, window_size)
            
            if window_size % 10 == 0:
                print(f"   Window size {window_size}: FPR = {fpr:.4f}")

            if fpr == 0.0:
                best_window_size = window_size
                print(f"✅ Found optimal window size ws* = {best_window_size}")
                break

        self.ws_threshold = best_window_size
        print(f"📊 Final optimal window size: {self.ws_threshold}")

        return self.ws_threshold

    def detect_anomalies(self, data, tr_threshold=None, ws_threshold=None):
        """
        使用训练好的模型和阈值检测异常

        Args:
            data: 输入数据
            tr_threshold: 异常阈值（可选，使用已计算的阈值）
            ws_threshold: 滑动窗口大小（可选，使用已计算的阈值）

        Returns:
            异常检测结果（0=良性，1=恶意）
        """
        if tr_threshold is None:
            tr_threshold = self.tr_threshold
        if ws_threshold is None:
            ws_threshold = self.ws_threshold

        if tr_threshold is None:
            raise ValueError("tr_threshold not set. Call calculate_anomaly_threshold() first.")
        if ws_threshold is None:
            raise ValueError("ws_threshold not set. Call optimize_window_size() first.")

        # 计算MSE
        mse_values = self.calculate_reconstruction_error(data)

        # 生成初始异常决策
        initial_decisions = (mse_values > tr_threshold).astype(int)

        # 应用滑动窗口多数投票
        final_decisions = []
        n_samples = len(initial_decisions)

        for i in range(n_samples):
            start = max(0, i - ws_threshold + 1)
            window = initial_decisions[start:i+1]
            if len(window) >= ws_threshold // 2:
                # 多数投票
                window_decision = 1 if sum(window) > len(window) / 2 else 0
            else:
                # 窗口太小，直接使用当前决策
                window_decision = initial_decisions[i]
            final_decisions.append(window_decision)

        final_decisions = np.array(final_decisions)
        print(f"📊 Anomaly detection results:")
        print(f"   Total samples: {n_samples}")
        print(f"   Detected anomalies: {sum(final_decisions)} ({sum(final_decisions)/n_samples*100:.2f}%)")

        return final_decisions, mse_values

    def evaluate_performance(self, data, true_labels, tr_threshold=None, ws_threshold=None):
        """
        评估异常检测性能

        Args:
            data: 输入数据
            true_labels: 真实标签（0=良性，1=恶意）
            tr_threshold: 异常阈值
            ws_threshold: 滑动窗口大小

        Returns:
            性能指标字典
        """
        print(f"\n{'=' * 60}")
        print("EVALUATING DETECTION PERFORMANCE")
        print(f"{'=' * 60}")

        # 检测异常
        predictions, mse_values = self.detect_anomalies(data, tr_threshold, ws_threshold)

        # 计算性能指标
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

        # 准确率
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # 精确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # 召回率（TPR）
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # 误报率（FPR）
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # F1分数
        f1 = f1_score(true_labels, predictions) if (tp + fp + fn) > 0 else 0.0

        # 计算ROC AUC
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(true_labels, mse_values)

        # 计算PR AUC
        precision_vals, recall_vals, _ = precision_recall_curve(true_labels, mse_values)
        from sklearn.metrics import auc
        pr_auc = auc(recall_vals, precision_vals)

        performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': {
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            }
        }

        print(f"📊 Performance metrics:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall (TPR): {recall:.4f}")
        print(f"   FPR: {fpr:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   PR AUC: {pr_auc:.4f}")
        print(f"   Confusion Matrix:")
        print(f"      TN: {tn}, FP: {fp}")
        print(f"      FN: {fn}, TP: {tp}")

        return performance

    def plot_performance_metrics(self, performance, save_dir=None):
        """
        绘制性能指标图表

        Args:
            performance: 性能指标字典
            save_dir: 保存目录（可选）
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 绘制混淆矩阵
        cm = np.array([[performance['confusion_matrix']['tn'], performance['confusion_matrix']['fp']],
                       [performance['confusion_matrix']['fn'], performance['confusion_matrix']['tp']]])

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Benign', 'Malicious'])
        plt.yticks(tick_marks, ['Benign', 'Malicious'])

        # 添加数值标签
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save_dir:
            save_path = os.path.join(save_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to: {save_path}")
        else:
            plt.show()

        # 绘制性能指标条形图
        metrics = ['Accuracy', 'Precision', 'Recall (TPR)', 'F1 Score', 'ROC AUC', 'PR AUC']
        values = [performance['accuracy'], performance['precision'], performance['recall'],
                  performance['f1'], performance['roc_auc'], performance['pr_auc']]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color='skyblue')
        plt.title('Performance Metrics')
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.4f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'performance_metrics.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Performance metrics plot saved to: {save_path}")
        else:
            plt.show()

    def plot_reconstruction_error(self, mse_values, true_labels, save_dir=None):
        """
        绘制重建误差分布

        Args:
            mse_values: 重建误差数组
            true_labels: 真实标签
            save_dir: 保存目录（可选）
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))

        # 分离良性和恶意数据的MSE
        benign_mse = mse_values[true_labels == 0]
        malicious_mse = mse_values[true_labels == 1]

        # 绘制直方图
        plt.hist(benign_mse, bins=50, alpha=0.5, label='Benign', color='green')
        plt.hist(malicious_mse, bins=50, alpha=0.5, label='Malicious', color='red')

        # 绘制阈值线
        if self.tr_threshold:
            plt.axvline(x=self.tr_threshold, color='blue', linestyle='--', label=f'Threshold tr* = {self.tr_threshold:.6f}')

        plt.title('Reconstruction Error Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)

        if save_dir:
            save_path = os.path.join(save_dir, 'reconstruction_error_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Reconstruction error distribution saved to: {save_path}")
        else:
            plt.show()
