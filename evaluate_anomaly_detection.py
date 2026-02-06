"""
异常检测性能评估驱动脚本
使用训练好的自编码器模型评估异常检测性能
"""
import os
import sys
import numpy as np
import json
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_processor import NBaIoTDataProcessor
from data_integrator import DStstIntegrator
from anomaly_detector import AnomalyDetector


def evaluate_device_anomaly_detection(device_name="Danmini_Doorbell"):
    """
    评估指定设备的异常检测性能

    Args:
        device_name: 设备名称（默认为Danmini_Doorbell）
    """
    print(f"\n{'=' * 80}")
    print(f"ANOMALY DETECTION PERFORMANCE EVALUATION")
    print(f"Device: {device_name}")
    print(f"{'=' * 80}")

    # 1. 初始化配置和模块
    print(f"\n1. INITIALIZING MODULES")
    print(f"{'-' * 40}")

    config = Config()
    data_processor = NBaIoTDataProcessor(config)
    data_integrator = DStstIntegrator(config)
    anomaly_detector = AnomalyDetector(config)

    # 2. 加载训练好的模型和scaler
    print(f"\n2. LOADING TRAINED MODEL AND SCALER")
    print(f"{'-' * 40}")

    device_output_dir = os.path.join(config.OUTPUT_DIR, device_name)
    model_path = os.path.join(device_output_dir, "final_model.h5")
    scaler_path = os.path.join(device_output_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        model_path = os.path.join(device_output_dir, "best_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found for device: {device_name}\n"
                f"Check if the device has been trained successfully"
            )

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No scaler found for device: {device_name}\n"
            f"Check if the device has been trained successfully"
        )

    # 加载模型和scaler
    anomaly_detector.load_model(model_path)
    anomaly_detector.load_scaler(scaler_path)

    # 3. 加载DSopt数据
    print(f"\n3. LOADING DSopt DATA")
    print(f"{'-' * 40}")

    # 加载良性数据
    benign_data = data_processor.load_device_data(device_name)
    if benign_data is None:
        raise ValueError(f"Failed to load benign data for device: {device_name}")

    # 按时间顺序划分数据
    DStrn, DSopt, DStst_benign = data_processor.split_data_chronologically(benign_data)

    # 4. 计算异常阈值 tr*
    print(f"\n4. CALCULATING ANOMALY THRESHOLD (tr*)")
    print(f"{'-' * 40}")

    tr_threshold = anomaly_detector.calculate_anomaly_threshold(DSopt)

    # 5. 优化滑动窗口大小 ws*
    print(f"\n5. OPTIMIZING WINDOW SIZE (ws*)")
    print(f"{'-' * 40}")

    ws_threshold = anomaly_detector.optimize_window_size(DSopt, tr_threshold)

    # 6. 创建或加载DStst数据集
    print(f"\n6. PREPARING DStst DATASET")
    print(f"{'-' * 40}")

    # 检查是否已存在保存的DStst数据集
    dstst_data_path = os.path.join(device_output_dir, "dstst_data.npy")
    dstst_labels_path = os.path.join(device_output_dir, "dstst_labels.npy")

    if os.path.exists(dstst_data_path) and os.path.exists(dstst_labels_path):
        print(f"📥 Loading existing DStst dataset")
        dstst_data = np.load(dstst_data_path)
        dstst_labels = np.load(dstst_labels_path)
        print(f"✅ Loaded {len(dstst_data)} samples")
    else:
        print(f"📋 Creating new DStst dataset")
        dstst_data, dstst_labels = data_integrator.create_dstst(device_name)
        data_integrator.save_dstst(device_name, dstst_data, dstst_labels)

    # 7. 评估异常检测性能
    print(f"\n7. EVALUATING ANOMALY DETECTION PERFORMANCE")
    print(f"{'-' * 40}")

    performance = anomaly_detector.evaluate_performance(
        dstst_data, dstst_labels, tr_threshold, ws_threshold
    )

    # 8. 保存评估结果
    print(f"\n8. SAVING EVALUATION RESULTS")
    print(f"{'-' * 40}")

    # 创建性能评估输出目录
    eval_output_dir = os.path.join(device_output_dir, "anomaly_detection_evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)

    # 辅助函数：将numpy类型转换为Python原生类型
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

    # 保存性能指标
    evaluation_results = {
        'device_name': device_name,
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tr_threshold': tr_threshold,
        'ws_threshold': ws_threshold,
        'performance': performance,
        'dataset_statistics': {
            'total_samples': len(dstst_data),
            'benign_samples': int(np.sum(dstst_labels == 0)),
            'attack_samples': int(np.sum(dstst_labels == 1))
        }
    }

    # 转换numpy类型
    evaluation_results = convert_numpy_types(evaluation_results)

    results_path = os.path.join(eval_output_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Evaluation results saved to: {results_path}")

    # 9. 生成性能图表
    print(f"\n9. GENERATING PERFORMANCE CHARTS")
    print(f"{'-' * 40}")

    # 计算重建误差
    _, mse_values = anomaly_detector.detect_anomalies(
        dstst_data, tr_threshold, ws_threshold
    )

    # 绘制性能图表
    anomaly_detector.plot_performance_metrics(performance, eval_output_dir)
    anomaly_detector.plot_reconstruction_error(mse_values, dstst_labels, eval_output_dir)

    # 10. 生成评估摘要
    print(f"\n{'=' * 80}")
    print(f"EVALUATION SUMMARY")
    print(f"Device: {device_name}")
    print(f"{'=' * 80}")

    print(f"\n📊 Key Metrics:")
    print(f"   Anomaly threshold (tr*): {tr_threshold:.6f}")
    print(f"   Window size (ws*): {ws_threshold}")
    print(f"   Accuracy: {performance['accuracy']:.4f}")
    print(f"   Recall (TPR): {performance['recall']:.4f}")
    print(f"   FPR: {performance['fpr']:.4f}")
    print(f"   F1 Score: {performance['f1']:.4f}")
    print(f"   ROC AUC: {performance['roc_auc']:.4f}")

    print(f"\n📥 Dataset:")
    print(f"   Total samples: {len(dstst_data)}")
    print(f"   Benign samples: {np.sum(dstst_labels == 0)}")
    print(f"   Attack samples: {np.sum(dstst_labels == 1)}")

    print(f"\n💾 Output:")
    print(f"   Evaluation results: {results_path}")
    print(f"   Performance charts: {eval_output_dir}")

    print(f"\n{'=' * 80}")
    print(f"EVALUATION COMPLETED")
    print(f"{'=' * 80}")

    return evaluation_results


def main():
    """
    主函数
    """
    # 解析命令行参数
    device_name = "Danmini_Doorbell"
    if len(sys.argv) > 1:
        device_name = sys.argv[1]

    print(f"🚀 Starting anomaly detection evaluation for device: {device_name}")

    try:
        # 运行评估
        evaluate_device_anomaly_detection(device_name)
        print(f"\n🎉 Evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
