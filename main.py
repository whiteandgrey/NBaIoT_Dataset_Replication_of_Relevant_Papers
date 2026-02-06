"""
增强的主程序 - 完全控制GPU使用
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

# 注意：必须在导入TensorFlow之前设置环境变量
# 导入配置并设置环境
from config import Config

# 设置环境变量（必须在导入TensorFlow之前）
Config.setup_environment()

# 现在导入TensorFlow
import tensorflow as tf

# 导入其他自定义模块
from data_processor import NBaIoTDataProcessor
from model import Autoencoder
from trainer import AutoencoderTrainer
from visualizer import ScientificVisualizer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='N-BaIoT Autoencoder Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU Control Examples:
  # 使用GPU（默认，如果可用）
  python main.py --gpu

  # 强制使用CPU
  python main.py --cpu

  # 使用特定GPU
  python main.py --gpu-device 0

  # 使用多个GPU
  python main.py --gpu-device 0,1

  # 限制GPU内存
  python main.py --gpu-memory 4096  # 限制为4GB

Device Selection Examples:
  # 训练所有设备
  python main.py

  # 训练单个设备
  python main.py --device Danmini_Doorbell

  # 训练多个设备
  python main.py --device Danmini_Doorbell --device Ecobee_Thermostat

  # 从文件读取设备列表
  python main.py --device-list devices.txt

  # 列出所有可用设备
  python main.py --list-devices

  # 交互式选择设备
  python main.py --interactive
        """
    )

    # GPU相关参数
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage (disable GPU)')
    parser.add_argument('--gpu', action='store_true',
                        help='Force GPU usage')
    parser.add_argument('--gpu-device', type=str, default=None,
                        help='Specific GPU device to use (e.g., "0" or "0,1")')
    parser.add_argument('--gpu-memory', type=int, default=None,
                        help='Limit GPU memory in MB')
    parser.add_argument('--no-memory-growth', action='store_true',
                        help='Disable GPU memory growth')

    # 设备选择参数
    parser.add_argument('--device', '-d', action='append',
                        help='Device name to train (can be used multiple times)')
    parser.add_argument('--device-list', '-dl', type=str,
                        help='File containing list of devices to train (one per line)')
    parser.add_argument('--list-devices', '-ld', action='store_true',
                        help='List all available devices and exit')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive device selection')
    parser.add_argument('--skip-existing', '-s', action='store_true',
                        help='Skip devices that already have trained models')
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Custom output directory')

    return parser.parse_args()


def setup_environment_with_args(args):
    """根据命令行参数设置环境"""
    print("=" * 80)
    print("N-BAIOT AUTOENCODER TRAINING SYSTEM")
    print("TensorFlow/Keras Implementation with Enhanced GPU Control")
    print("=" * 80)

    # 根据命令行参数更新配置
    if args.cpu:
        Config.USE_GPU = False
        Config.GPU_DEVICES = "-1"
        print("🔧 Command line: Forcing CPU usage")

    if args.gpu:
        Config.USE_GPU = True
        print("🔧 Command line: Forcing GPU usage")

    if args.gpu_device is not None:
        Config.USE_GPU = True
        Config.GPU_DEVICES = args.gpu_device
        print(f"🔧 Command line: Using GPU device(s) {args.gpu_device}")

    if args.gpu_memory is not None:
        Config.GPU_MEMORY_LIMIT = args.gpu_memory
        print(f"🔧 Command line: Limiting GPU memory to {args.gpu_memory}MB")

    if args.no_memory_growth:
        Config.GPU_MEMORY_GROWTH = False
        print("🔧 Command line: Disabling GPU memory growth")

    # 如果指定了输出目录，更新配置
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
        Config.MODEL_SAVE_DIR = os.path.join(args.output_dir, "saved_models")

    # 显示配置
    Config.display_config()

    # 设置TensorFlow配置
    Config.setup_tensorflow()

    # 创建输出目录
    Config.setup_directories()

    return True


def list_available_devices(data_processor):
    """列出所有可用设备"""
    print("\n📋 Available IoT Devices in Dataset:")
    print("=" * 50)

    # 获取实际存在的设备文件夹
    if os.path.exists(Config.DATA_ROOT):
        actual_devices = data_processor.get_available_devices()
        print(f"Found {len(actual_devices)} device folders:")

        for i, device in enumerate(actual_devices, 1):
            device_path = os.path.join(Config.DATA_ROOT, device)
            csv_path = os.path.join(device_path, "benign_traffic.csv")

            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, nrows=1)
                    print(f"  {i:2d}. {device:40} - {df.shape[0]:,} samples, {df.shape[1]} features")
                except:
                    print(f"  {i:2d}. {device:40} - (error reading file)")
            else:
                print(f"  {i:2d}. {device:40} - (file not found)")
    else:
        print(f"Data directory not found: {Config.DATA_ROOT}")
        print("Please update DATA_ROOT in config.py")

    print("=" * 50)


def interactive_device_selection(data_processor):
    """交互式设备选择"""
    print("\n🎯 Interactive Device Selection")
    print("=" * 50)

    # 获取可用设备
    available_devices = data_processor.get_available_devices()

    if not available_devices:
        print("❌ No devices found in data directory.")
        return []

    # 显示设备列表
    print(f"Available devices ({len(available_devices)} total):")
    for i, device in enumerate(available_devices, 1):
        print(f"  {i:2d}. {device}")

    print("\nOptions:")
    print("  [a]ll - Train all devices")
    print("  [n]one - Cancel training")
    print("  [1,2,3...] - Select device numbers (comma-separated)")
    print("  [1-5] - Select device range")

    while True:
        try:
            selection = input("\nEnter your selection: ").strip().lower()

            if selection == 'a' or selection == 'all':
                print("✅ Selected all devices")
                return available_devices

            elif selection == 'n' or selection == 'none':
                print("❌ Training cancelled")
                return []

            elif selection:
                selected_devices = []

                # 处理逗号分隔的列表和范围
                parts = selection.split(',')
                for part in parts:
                    part = part.strip()

                    if '-' in part:
                        # 处理范围
                        start_str, end_str = part.split('-')
                        start = int(start_str.strip()) - 1
                        end = int(end_str.strip())

                        if 0 <= start < len(available_devices) and 0 < end <= len(available_devices):
                            selected_devices.extend(available_devices[start:end])
                        else:
                            print(f"⚠️ Invalid range: {part}")
                    else:
                        # 处理单个编号
                        try:
                            idx = int(part) - 1
                            if 0 <= idx < len(available_devices):
                                selected_devices.append(available_devices[idx])
                            else:
                                print(f"⚠️ Invalid device number: {part}")
                        except ValueError:
                            # 尝试按名称匹配
                            matching_devices = [d for d in available_devices
                                                if part.lower() in d.lower()]
                            if matching_devices:
                                selected_devices.extend(matching_devices)
                            else:
                                print(f"⚠️ No device matching: {part}")

                if selected_devices:
                    # 去重
                    selected_devices = list(set(selected_devices))
                    print(f"✅ Selected {len(selected_devices)} device(s):")
                    for device in selected_devices:
                        print(f"  • {device}")
                    return selected_devices
                else:
                    print("⚠️ No valid devices selected. Please try again.")

        except KeyboardInterrupt:
            print("\n\n⚠️ Selection cancelled.")
            return []
        except Exception as e:
            print(f"⚠️ Error: {e}. Please try again.")


def load_device_list_from_file(filename):
    """从文件加载设备列表"""
    try:
        with open(filename, 'r') as f:
            devices = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"✅ Loaded {len(devices)} device(s) from {filename}")
        return devices
    except Exception as e:
        print(f"❌ Error loading device list from {filename}: {e}")
        return []


def get_devices_to_train(args, data_processor):
    """
    获取要训练的设备列表

    优先级：
    1. 命令行参数 --device 或 --device-list
    2. 配置文件中的 SELECTED_DEVICES
    3. 交互式选择（如果启用）
    4. 所有可用设备（默认）
    """
    selected_devices = []

    # 1. 检查命令行参数
    if args.device:
        # 命令行指定的设备
        selected_devices = args.device
        print(f"📋 Devices specified via command line: {selected_devices}")

    elif args.device_list:
        # 从文件加载设备列表
        selected_devices = load_device_list_from_file(args.device_list)

    elif args.interactive:
        # 交互式选择
        selected_devices = interactive_device_selection(data_processor)

    elif Config.SELECTED_DEVICES:
        # 配置文件中的设备
        selected_devices = Config.get_selected_devices()
        print(f"📋 Devices from config file: {selected_devices}")

    # 2. 如果还没有选择设备，使用所有可用设备
    if not selected_devices:
        selected_devices = data_processor.get_available_devices()
        print(f"📋 Training all available devices: {selected_devices}")

    # 3. 验证设备是否存在
    available_devices = data_processor.get_available_devices()
    valid_devices = []
    invalid_devices = []

    for device in selected_devices:
        if device in available_devices:
            valid_devices.append(device)
        else:
            invalid_devices.append(device)

    if invalid_devices:
        print(f"⚠️ Warning: {len(invalid_devices)} device(s) not found:")
        for device in invalid_devices:
            print(f"  • {device}")
        print(f"Available devices: {available_devices}")

    # 4. 检查是否跳过已训练的模型
    if args.skip_existing and valid_devices:
        filtered_devices = []
        for device in valid_devices:
            model_dir = os.path.join(Config.OUTPUT_DIR, device)
            model_file = os.path.join(model_dir, 'final_model.h5')

            if os.path.exists(model_file):
                print(f"⏭️  Skipping {device} (model already exists)")
            else:
                filtered_devices.append(device)

        valid_devices = filtered_devices

    return valid_devices


def train_single_device(device_name, data_processor, visualizer):
    """
    训练单个设备的自编码器

    Args:
        device_name: 设备名称
        data_processor: 数据处理器
        visualizer: 可视化器

    Returns:
        训练结果字典
    """
    print(f"\n{'#' * 80}")
    print(f"TRAINING DEVICE: {device_name}")
    print(f"{'#' * 80}")

    device_start_time = time.time()

    try:
        # 1. 加载数据
        print(f"\n📥 Loading data for {device_name}...")
        data = data_processor.load_device_data(device_name)
        if data is None:
            print(f"❌ Failed to load data for {device_name}")
            return None

        # 2. 划分数据
        print(f"\n📊 Splitting data...")
        if Config.TIME_ORDERED:
            DStrn, DSopt, DStst = data_processor.split_data_chronologically(data)
        else:
            DStrn, DSopt, DStst = data_processor.split_data_randomly(data)

        # 3. 预处理数据
        print(f"\n🔧 Preprocessing data...")
        DStrn_processed = data_processor.preprocess_data(DStrn, fit_scaler=True)
        DSopt_processed = data_processor.preprocess_data(DSopt, fit_scaler=False)

        # 获取数据信息
        data_info = {
            'device_name': device_name,
            'n_features': data.shape[1],
            'n_samples': len(data),
            'train_samples': len(DStrn),
            'val_samples': len(DSopt),
            'test_samples': len(DStst),
            'train_ratio': len(DStrn) / len(data) if len(data) > 0 else 0,
            'val_ratio': len(DSopt) / len(data) if len(data) > 0 else 0,
            'test_ratio': len(DStst) / len(data) if len(data) > 0 else 0
        }

        # 4. 创建训练数据
        print(f"\n📈 Creating training datasets...")
        (X_train, y_train), (X_val, y_val) = data_processor.create_numpy_datasets(
            DStrn_processed, DSopt_processed
        )

        # 5. 创建训练器
        print(f"\n🏋️ Creating trainer...")
        trainer = AutoencoderTrainer(Config, device_name)

        # 6. 初始训练
        print(f"\n🚀 Starting initial training...")
        initial_val_loss = trainer.initial_training(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val)
        )

        # 7. 超参数调优
        print(f"\n🔍 Starting hyperparameter tuning...")
        best_params = trainer.hyperparameter_tuning(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val)
        )

        # 8. 最终训练（合并训练和验证数据）
        print(f"\n🎯 Starting final training...")
        # 合并训练和验证数据
        X_combined = np.concatenate([X_train, X_val], axis=0)
        y_combined = np.concatenate([y_train, y_val], axis=0)

        final_loss = trainer.final_training(
            train_data=(X_combined, y_combined)
        )

        # 9. 计算设备训练时间
        device_total_time = time.time() - device_start_time

        # 10. 生成可视化图表
        print(f"\n📊 Generating visualizations...")
        visualizer.generate_all_plots(trainer, device_name, data_info)

        # 11. 保存scaler
        if Config.SAVE_SCALER:
            scaler_path = os.path.join(trainer.device_output_dir, 'scaler.pkl')
            data_processor.save_scaler(scaler_path)

        # 12. 返回训练结果
        result = {
            'device_name': device_name,
            'best_params': trainer.training_history.get('best_params'),
            'best_val_loss': trainer.training_history['best_val_loss'],
            'final_train_loss': final_loss,
            'training_time': device_total_time,
            'data_info': data_info,
            'model_path': os.path.join(trainer.device_output_dir, 'final_model.h5')
        }

        print(f"\n✅ Device {device_name} training completed!")
        print(f"   Training time: {device_total_time:.2f} seconds")
        print(f"   Best validation loss: {result['best_val_loss']:.6f}")
        print(f"   Final training loss: {result['final_train_loss']:.6f}")

        return result

    except Exception as e:
        print(f"\n❌ Error training {device_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    # 1. 解析命令行参数
    args = parse_arguments()

    # 2. 如果只是列出设备，则列出后退出
    if args.list_devices:
        data_processor = NBaIoTDataProcessor(Config)
        list_available_devices(data_processor)
        return

    # 3. 根据命令行参数设置环境
    print("Setting up environment...")
    if not setup_environment_with_args(args):
        return

    # 打印TensorFlow版本和设备信息
    print(f"\n🔍 TensorFlow Version: {tf.__version__}")
    print(f"🔍 Keras Version: {tf.keras.__version__}")

    # 4. 初始化模块
    print("\nInitializing modules...")
    data_processor = NBaIoTDataProcessor(Config)
    visualizer = ScientificVisualizer(Config)

    # 5. 获取要训练的设备列表
    print("\nDetermining devices to train...")
    devices_to_train = get_devices_to_train(args, data_processor)

    if not devices_to_train:
        print("❌ No devices selected for training. Exiting.")
        return

    print(f"\n🎯 Will train {len(devices_to_train)} device(s):")
    for i, device in enumerate(devices_to_train, 1):
        print(f"  {i:2d}. {device}")

    # 6. 开始训练
    print(f"\n{'=' * 80}")
    print("STARTING TRAINING PROCESS")
    print(f"{'=' * 80}")
    print(f"Start time: {Config.get_timestamp()}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Number of devices: {len(devices_to_train)}")
    print(f"Using GPU: {Config.USE_GPU}")
    print(f"Plot save: {Config.PLOT_SAVE}")
    print(f"Plot show: {Config.PLOT_SHOW}")
    print(f"{'=' * 80}\n")

    total_start_time = time.time()
    all_results = []

    # 7. 遍历选择的设备进行训练
    for device_name in devices_to_train:
        print(f"\n{'#' * 80}")
        print(f"PROCESSING: {device_name}")
        print(f"{'#' * 80}")

        result = train_single_device(device_name, data_processor, visualizer)

        if result:
            all_results.append(result)

        print(f"\n{'#' * 80}")
        print(f"COMPLETED: {device_name}")
        print(f"{'#' * 80}")

    # 8. 计算总训练时间
    total_training_time = time.time() - total_start_time

    # 9. 生成总结报告
    print(f"\n{'=' * 80}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total devices trained: {len(all_results)}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per device: {total_training_time / len(all_results) if all_results else 0:.2f} seconds")
    print(f"Completion time: {Config.get_timestamp()}")

    # 10. 保存详细结果到CSV
    if all_results:
        # 创建结果DataFrame
        results_data = []
        for result in all_results:
            row = {
                'device': result['device_name'],
                'best_lr': result['best_params']['lr'],
                'best_epochs': result['best_params']['epochs'],
                'best_val_loss': result['best_val_loss'],
                'final_train_loss': result['final_train_loss'],
                'training_time_seconds': result['training_time'],
                'train_samples': result['data_info']['train_samples'],
                'val_samples': result['data_info']['val_samples'],
                'test_samples': result['data_info']['test_samples'],
                'total_samples': result['data_info']['n_samples'],
                'model_path': result.get('model_path', '')
            }
            results_data.append(row)

        results_df = pd.DataFrame(results_data)

        # 保存到CSV
        csv_path = os.path.join(Config.OUTPUT_DIR, "training_results_summary.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n📈 Detailed results saved to: {csv_path}")

        # 11. 生成所有设备比较图（如果有多于一个设备）
        if Config.PLOT_DEVICE_COMPARISON and len(all_results) > 1:
            print(f"\n📊 Generating device comparison charts...")
            visualizer.plot_device_comparison(all_results)

        # 12. 显示统计摘要
        if len(all_results) > 0:
            print(f"\n📊 FINAL STATISTICAL SUMMARY:")
            print(f"{'-' * 50}")

            val_losses = [r['best_val_loss'] for r in all_results]
            train_times = [r['training_time'] for r in all_results]

            # 找出最佳和最差设备
            best_idx = np.argmin(val_losses)
            worst_idx = np.argmax(val_losses)
            fastest_idx = np.argmin(train_times)
            slowest_idx = np.argmax(train_times)

            print(f"🏆 Best Performing Device: {all_results[best_idx]['device_name']}")
            print(f"   Validation Loss: {val_losses[best_idx]:.6f}")
            print(f"   Training Time: {train_times[best_idx]:.2f}s")
            print()

            if len(all_results) > 1:
                print(f"📉 Worst Performing Device: {all_results[worst_idx]['device_name']}")
                print(f"   Validation Loss: {val_losses[worst_idx]:.6f}")
                print(f"   Training Time: {train_times[worst_idx]:.2f}s")
                print()

                print(f"⚡ Fastest Training Device: {all_results[fastest_idx]['device_name']}")
                print(f"   Validation Loss: {val_losses[fastest_idx]:.6f}")
                print(f"   Training Time: {train_times[fastest_idx]:.2f}s")
                print()

                print(f"🐌 Slowest Training Device: {all_results[slowest_idx]['device_name']}")
                print(f"   Validation Loss: {val_losses[slowest_idx]:.6f}")
                print(f"   Training Time: {train_times[slowest_idx]:.2f}s")
                print()

                print(f"📈 Performance Statistics:")
                print(f"   Average Loss: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
                print(f"   Loss Range: [{min(val_losses):.6f}, {max(val_losses):.6f}]")
                print(f"   Average Time: {np.mean(train_times):.2f}s ± {np.std(train_times):.2f}s")
                print(f"   Time Range: [{min(train_times):.2f}s, {max(train_times):.2f}s]")
            else:
                print(f"📈 Performance:")
                print(f"   Validation Loss: {val_losses[0]:.6f}")
                print(f"   Training Time: {train_times[0]:.2f}s")

            print(f"{'-' * 50}")

    # 13. 最终输出
    print(f"\n{'=' * 80}")
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 80}")
    print(f"Total devices processed: {len(all_results)}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {os.path.abspath(Config.OUTPUT_DIR)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    # 运行主函数
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()