"""
增强的配置 - 完全控制GPU使用
"""
import os
import sys
import numpy as np
from datetime import datetime


class Config:
    """训练配置参数"""

    # ============ GPU控制配置 ============
    # 完全控制GPU使用
    USE_GPU = False  # 设置为False将完全禁用GPU

    # GPU内存配置
    GPU_MEMORY_LIMIT = None  # 设置GPU内存限制，单位MB，None表示不限制
    GPU_MEMORY_GROWTH = True  # 允许GPU内存动态增长

    # GPU设备选择
    GPU_DEVICES = "0"  # 使用哪个GPU，例如："0"或"0,1"或"-1"表示禁用GPU

    # ============ 路径配置 ============
    DATA_ROOT = r"C:\Users\WWWWG\Desktop\NBaIoT"  # 用户需要修改为实际路径
    OUTPUT_DIR = "./training_results"
    MODEL_SAVE_DIR = "./saved_models"

    # ============ 设备选择配置 ============
    # 设备列表 - 所有可用的N-BaIoT设备
    ALL_DEVICES = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera"
    ]

    # 选择要训练的设备
    SELECTED_DEVICES = ["Danmini_Doorbell"]  # 空列表表示训练所有设备

    # ============ 数据配置 ============
    FEATURE_DIM = 115  # N-BaIoT数据集特征维度
    TRAIN_RATIO = 1 / 3
    VAL_RATIO = 1 / 3
    TEST_RATIO = 1 / 3

    # 假设数据按时间顺序排列
    TIME_ORDERED = True
    RANDOM_SEED = 42

    # ============ 模型架构配置 ============
    # 对称编码器-解码器结构
    ENCODER_RATIOS = [0.75, 0.50, 0.33, 0.25]  # 编码器每层相对于输入的比例
    DECODER_RATIOS = [0.33, 0.50, 0.75, 1.0]  # 解码器每层相对于输入的比例

    # 激活函数和正则化
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = None  # 输出层激活函数，None表示线性
    USE_BATCH_NORM = False
    DROPOUT_RATE = 0.0
    L2_REGULARIZATION = 0.001

    # ============ 超参数搜索空间 ============
    LEARNING_RATES = [1e-4, 1e-3, 5e-3]
    EPOCHS_OPTIONS = [50, 100]
    BATCH_SIZES = [32, 64, 128]

    # ============ 训练配置 ============
    # 默认训练参数
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_EPOCHS = 100

    # 早停和回调
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.5
    MIN_DELTA = 1e-6

    # 优化器
    OPTIMIZER = 'adam'
    BETA_1 = 0.9  # Adam参数
    BETA_2 = 0.999  # Adam参数
    EPSILON = 1e-7  # Adam参数

    # ============ 实验配置 ============
    VERBOSE = 1  # 0=静默, 1=进度条, 2=每个epoch一行

    # ============ 入侵检测配置 ============
    # 滑动窗口优化配置
    MAX_WINDOW_SIZE = 500  # 最大滑动窗口大小
    MIN_WINDOW_SIZE = 1  # 最小滑动窗口大小
    WINDOW_SIZE_STEP = 1  # 窗口大小步长

    # ============ 可视化配置 ============
    PLOT_SAVE = True
    PLOT_SHOW = False
    PLOT_FORMAT = 'png'
    PLOT_DPI = 300
    PLOT_STYLE = 'seaborn-darkgrid'
    
    # 比较图表总开关
    PLOT_COMPARISON = True

    # ============ 文件保存配置 ============
    # 日志文件保存
    SAVE_LOG_FILE = True
    LOG_FILE = "./training_log.txt"

    # 模型文件保存
    SAVE_MODEL = True
    SAVE_BEST_MODEL_ONLY = True
    MODEL_SAVE_DIR = "./saved_models"

    # 训练历史保存
    SAVE_TRAINING_HISTORY = True
    SAVE_HYPERPARAMETER_TUNING_RESULTS = True

    # Scaler保存
    SAVE_SCALER = True

    # TensorBoard日志
    SAVE_TENSORBOARD_LOGS = False
    TENSORBOARD_LOG_DIR = "./tensorboard_logs"

    # ============ 图表类型配置 ============
    # 训练曲线图表
    PLOT_TRAINING_LOSS_CURVE = True
    PLOT_TRAINING_MAE_CURVE = True
    PLOT_TRAINING_LR_CURVE = True

    # 超参数调优图表
    PLOT_HYPERPARAM_HEATMAP = True
    PLOT_HYPERPARAM_CONTOUR = True
    PLOT_HYPERPARAM_3D = False

    # 损失分析图表
    PLOT_LOSS_DISTRIBUTION = True
    PLOT_LOSS_HISTOGRAM = True
    PLOT_LOSS_BOX_PLOT = True
    PLOT_LOSS_VIOLIN_PLOT = True

    # 模型性能图表
    PLOT_PERFORMANCE_METRICS = True
    PLOT_LEARNING_RATE_SCHEDULE = True
    PLOT_GRADIENT_FLOW = False

    # 数据分析图表
    PLOT_DATA_DISTRIBUTION = True
    PLOT_FEATURE_CORRELATION = False
    PLOT_PCA_VISUALIZATION = False

    # 时间分析图表
    PLOT_TRAINING_TIME_ANALYSIS = True
    PLOT_EPOCH_TIME_DISTRIBUTION = True

    # 比较图表
    PLOT_DEVICE_COMPARISON = True
    PLOT_PHASE_COMPARISON = True
    PLOT_PERFORMANCE_RANKING = True

    # 综合报告图表
    PLOT_COMPREHENSIVE_SUMMARY = True
    PLOT_TRAINING_REPORT = True

    # 图表布局选项
    PLOT_SUMMARY_GRID = (3, 3)
    PLOT_COMPARISON_GRID = (2, 3)

    # 图表输出目录结构
    PLOT_SUBDIR_TRAINING = "training_plots"
    PLOT_SUBDIR_COMPARISON = "comparison_plots"
    PLOT_SUBDIR_DEBUG = "debug_plots"
    PLOT_SUBDIR_METRICS = "metrics_plots"
    PLOT_SUBDIR_DATA = "data_plots"

    # 图表文件名模式
    PLOT_FILENAME_PATTERN = "{device}_{plot_type}_{timestamp}.{format}"

    @classmethod
    def setup_environment(cls):
        """设置环境变量，必须在导入TensorFlow之前调用"""
        import warnings
        warnings.filterwarnings('ignore')

        # 设置随机种子
        np.random.seed(cls.RANDOM_SEED)

        # 设置环境变量控制TensorFlow行为
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息
        os.environ['PYTHONWARNINGS'] = 'ignore'  # 忽略Python警告

        # 根据USE_GPU设置CUDA_VISIBLE_DEVICES
        if not cls.USE_GPU:
            # 禁用GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("🔧 GPU disabled: Using CPU only")
        else:
            # 使用指定的GPU设备
            os.environ['CUDA_VISIBLE_DEVICES'] = cls.GPU_DEVICES
            if cls.GPU_DEVICES == "-1":
                print("🔧 GPU disabled via device specification")
            else:
                print(f"🔧 GPU enabled: Using device(s) {cls.GPU_DEVICES}")
                print(f"⚠️ Note: If GPU is not detected, please check:")
                print(f"   1. GPU drivers are installed")
                print(f"   2. CUDA 11.2 and cuDNN 8.1 are installed (for TF 2.10.0)")
                print(f"   3. CUDA and cuDNN paths are in system PATH")
                print(f"   4. TensorFlow GPU version is installed")

        # 设置TensorFlow日志级别
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # 创建必要的目录
        cls.setup_directories()

        return True

    @classmethod
    def setup_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)



    @classmethod
    def get_selected_devices(cls):
        """
        获取选择的设备列表

        Returns:
            选择的设备名称列表
        """
        if not cls.SELECTED_DEVICES:
            # 如果没有指定设备，返回所有设备
            return cls.ALL_DEVICES
        else:
            # 过滤出有效设备
            valid_devices = []
            for device in cls.SELECTED_DEVICES:
                if device in cls.ALL_DEVICES:
                    valid_devices.append(device)
                else:
                    print(f"⚠️ Warning: Device '{device}' is not in the known device list.")
            return valid_devices

    @classmethod
    def display_config(cls):
        """显示当前配置"""
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)

        config_items = [
            ("DATA_ROOT", cls.DATA_ROOT),
            ("FEATURE_DIM", cls.FEATURE_DIM),
            ("OUTPUT_DIR", cls.OUTPUT_DIR),
            ("USE_GPU", cls.USE_GPU),
            ("GPU_DEVICES", cls.GPU_DEVICES),
            ("SELECTED_DEVICES", cls.get_selected_devices()),
            ("NUMBER_OF_DEVICES", len(cls.get_selected_devices())),
            ("RANDOM_SEED", cls.RANDOM_SEED),
            ("TIME_ORDERED", cls.TIME_ORDERED),
            ("ENCODER_RATIOS", cls.ENCODER_RATIOS),
            ("DECODER_RATIOS", cls.DECODER_RATIOS),
            ("LEARNING_RATES", cls.LEARNING_RATES),
            ("EPOCHS_OPTIONS", cls.EPOCHS_OPTIONS),
            ("BATCH_SIZES", cls.BATCH_SIZES),
            ("EARLY_STOPPING_PATIENCE", cls.EARLY_STOPPING_PATIENCE),
            ("ACTIVATION", cls.ACTIVATION),
            ("USE_BATCH_NORM", cls.USE_BATCH_NORM),
            ("DROPOUT_RATE", cls.DROPOUT_RATE),
        ]

        for key, value in config_items:
            print(f"{key:30}: {value}")

        print("=" * 70)

    @classmethod
    def get_current_time_str(cls):
        """获取当前时间字符串"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @classmethod
    def get_timestamp(cls):
        """获取时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def setup_tensorflow(cls):
        """
        设置TensorFlow配置
        注意：这个函数必须在环境变量设置之后调用
        """
        import tensorflow as tf

        # 设置随机种子
        tf.random.set_seed(cls.RANDOM_SEED)

        # 检查GPU可用性
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')

        print("\n🔍 TensorFlow Device Information:")
        print("-" * 40)
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"Physical CPUs: {len(cpus)}")
        print(f"Physical GPUs: {len(gpus)}")

        if gpus and cls.USE_GPU:
            try:
                # 设置GPU内存增长
                if cls.GPU_MEMORY_GROWTH:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                # 设置GPU内存限制
                if cls.GPU_MEMORY_LIMIT and gpus:
                    try:
                        tf.config.set_logical_device_configuration(
                            gpus[0],
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=cls.GPU_MEMORY_LIMIT
                            )]
                        )
                        print(f"✅ GPU memory limited to {cls.GPU_MEMORY_LIMIT}MB")
                    except Exception as e:
                        print(f"⚠️ GPU memory limit error: {e}")

                print(f"✅ GPU available: {len(gpus)} device(s)")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")

                # 验证TensorFlow是否真的在使用GPU
                print("\n🔍 Testing GPU access...")
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    print(f"   Test tensor device: {test_tensor.device}")
                    print("   GPU access: ✓ Available")

                return True

            except RuntimeError as e:
                print(f"⚠️ GPU setup error: {e}")
                print("   Falling back to CPU")
                return False
        else:
            if not cls.USE_GPU:
                print("ℹ️ GPU disabled by configuration")
            elif not gpus:
                print("⚠️ No GPU devices found")
                print("\n🔧 GPU Troubleshooting Guide:")
                print("-" * 40)
                print("If you have a GPU but TensorFlow cannot detect it, please check:")
                print()
                print("1. Verify GPU is installed and visible:")
                print("   - Open NVIDIA Control Panel or Device Manager")
                print("   - Check if your GPU is listed")
                print()
                print("2. Install NVIDIA GPU Drivers:")
                print("   - Download from: https://www.nvidia.com/Download/index.aspx")
                print("   - Install the latest driver for your GPU")
                print()
                print("3. Install CUDA Toolkit (11.2 for TF 2.10.0):")
                print("   - Download from: https://developer.nvidia.com/cuda-11-2-0-download-archive")
                print("   - Install and add to PATH")
                print()
                print("4. Install cuDNN (8.1 for TF 2.10.0):")
                print("   - Download from: https://developer.nvidia.com/cudnn")
                print("   - Extract and copy files to CUDA directories")
                print("   - Add bin folder to PATH")
                print()
                print("5. Install TensorFlow GPU version:")
                print("   - pip install tensorflow-gpu==2.10.0")
                print("   - Or: pip install tensorflow==2.10.0 (includes GPU support)")
                print()
                print("6. Verify installation:")
                print("   - Run: python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"")
                print("   - Should show your GPU devices")
                print()
                print("7. Check environment variables:")
                print("   - CUDA_PATH should point to CUDA installation")
                print("   - PATH should include CUDA\\bin and cuDNN\\bin")
                print()
                print("Common issues:")
                print("   - TensorFlow 2.10.0 only supports CUDA 11.2 and cuDNN 8.1")
                print("   - Newer TensorFlow versions (2.11+) don't support GPU on Windows")
                print("   - Make sure you're using the correct TensorFlow version")
                print()
            else:
                print("ℹ️ Using CPU (GPU disabled via CUDA_VISIBLE_DEVICES)")

            # 验证TensorFlow是否在使用CPU
            print("\n🔍 Testing CPU access...")
            with tf.device('/CPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                print(f"   Test tensor device: {test_tensor.device}")
                print("   CPU access: ✓ Available")

            return False

        print("-" * 40)
