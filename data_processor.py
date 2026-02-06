"""
数据处理模块 - 加载、划分、归一化
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class NBaIoTDataProcessor:
    """N-BaIoT数据集处理器"""

    def __init__(self, config):
        """
        初始化数据处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.scaler = StandardScaler()
        self.data_info = {}
        self._setup_random_seed()

    def _setup_random_seed(self):
        """设置随机种子"""
        np.random.seed(self.config.RANDOM_SEED)

    def validate_data_root(self):
        """验证数据根目录是否存在"""
        if not os.path.exists(self.config.DATA_ROOT):
            raise FileNotFoundError(
                f"Data directory not found: {self.config.DATA_ROOT}\n"
                f"Please update DATA_ROOT in config.py"
            )

        # 检查是否有设备文件夹
        device_folders = self.get_available_devices()
        if not device_folders:
            raise FileNotFoundError(
                f"No device folders found in {self.config.DATA_ROOT}"
            )

        return device_folders

    def get_available_devices(self):
        """获取可用的设备列表（基于实际存在的文件夹）"""
        if not os.path.exists(self.config.DATA_ROOT):
            print(f"⚠️ Data directory not found: {self.config.DATA_ROOT}")
            return []

        # 获取所有文件夹
        all_folders = [f for f in os.listdir(self.config.DATA_ROOT)
                       if os.path.isdir(os.path.join(self.config.DATA_ROOT, f))]

        # 检查文件夹是否包含benign_traffic.csv文件
        valid_devices = []
        for folder in all_folders:
            csv_path = os.path.join(self.config.DATA_ROOT, folder, "benign_traffic.csv")
            if os.path.exists(csv_path):
                valid_devices.append(folder)
            else:
                print(f"⚠️ Folder {folder} does not contain benign_traffic.csv")

        return valid_devices

    def load_device_data(self, device_name):
        """
        加载单个设备的数据

        Args:
            device_name: 设备名称

        Returns:
            numpy数组或None（如果加载失败）
        """
        device_path = os.path.join(self.config.DATA_ROOT, device_name)
        csv_path = os.path.join(device_path, "benign_traffic.csv")

        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"File not found: {csv_path}")

            # 加载CSV文件
            print(f"Loading {csv_path}...")
            df = pd.read_csv(csv_path)

            # 记录数据信息
            self.data_info[device_name] = {
                'original_shape': df.shape,
                'features': df.columns.tolist(),
                'file_path': csv_path,
                'data_type': df.dtypes.to_dict()
            }

            # 检查特征维度
            n_features = df.shape[1]
            if n_features != self.config.FEATURE_DIM:
                print(f"⚠️ Warning: {device_name} has {n_features} features, "
                      f"expected {self.config.FEATURE_DIM}.")

                # 如果特征多于预期，使用前FEATURE_DIM个特征
                if n_features > self.config.FEATURE_DIM:
                    print(f"   Using first {self.config.FEATURE_DIM} features.")
                    df = df.iloc[:, :self.config.FEATURE_DIM]
                # 如果特征少于预期，填充零值
                else:
                    print(f"   Padding with zeros to {self.config.FEATURE_DIM} features.")
                    padding_cols = self.config.FEATURE_DIM - n_features
                    padding_data = np.zeros((len(df), padding_cols))
                    padding_df = pd.DataFrame(
                        padding_data,
                        columns=[f'pad_{i}' for i in range(padding_cols)]
                    )
                    df = pd.concat([df, padding_df], axis=1)

            # 转换为numpy数组
            data = df.values.astype(np.float32)

            # 检查NaN或Inf值
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(f"⚠️ Warning: {device_name} data contains NaN or Inf values.")
                # 用0填充NaN，用最大/最小值填充Inf
                data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

            print(f"✅ Loaded {device_name}: {len(data)} samples, {data.shape[1]} features")
            return data

        except Exception as e:
            print(f"❌ Error loading data for {device_name}: {str(e)}")
            return None

    def split_data_chronologically(self, data):
        """
        按时间顺序将数据三等分

        Args:
            data: 输入数据

        Returns:
            DStrn, DSopt, DStst
        """
        n_samples = len(data)

        if n_samples < 3:
            raise ValueError(f"Data too small ({n_samples} samples) for three-way split")

        # 计算划分点
        split1 = int(n_samples * self.config.TRAIN_RATIO)
        split2 = int(n_samples * (self.config.TRAIN_RATIO + self.config.VAL_RATIO))

        # 划分数据
        DStrn = data[:split1]
        DSopt = data[split1:split2]
        DStst = data[split2:]

        print(f"📊 Chronological split:")
        print(f"   DStrn (train): {len(DStrn)} samples ({len(DStrn) / n_samples * 100:.1f}%)")
        print(f"   DSopt (val):   {len(DSopt)} samples ({len(DSopt) / n_samples * 100:.1f}%)")
        print(f"   DStst (test):  {len(DStst)} samples ({len(DStst) / n_samples * 100:.1f}%)")

        return DStrn, DSopt, DStst

    def split_data_randomly(self, data, random_state=None):
        """
        随机划分数据（备用方法）

        Args:
            data: 输入数据
            random_state: 随机种子

        Returns:
            DStrn, DSopt, DStst
        """
        if random_state is None:
            random_state = self.config.RANDOM_SEED

        # 先划分训练集
        DStrn, temp = train_test_split(
            data,
            train_size=self.config.TRAIN_RATIO,
            random_state=random_state,
            shuffle=True
        )

        # 剩余数据再划分验证集和测试集
        val_ratio = self.config.VAL_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO)
        DSopt, DStst = train_test_split(
            temp,
            train_size=val_ratio,
            random_state=random_state,
            shuffle=True
        )

        print(f"📊 Random split (seed={random_state}):")
        print(f"   DStrn (train): {len(DStrn)} samples")
        print(f"   DSopt (val):   {len(DSopt)} samples")
        print(f"   DStst (test):  {len(DStst)} samples")

        return DStrn, DSopt, DStst

    def preprocess_data(self, data, fit_scaler=False):
        """
        数据预处理：标准化

        Args:
            data: 输入数据
            fit_scaler: 是否拟合新的scaler

        Returns:
            预处理后的数据
        """
        if fit_scaler:
            processed_data = self.scaler.fit_transform(data)
            print(f"✅ Fitted scaler on {len(data)} samples")
        else:
            processed_data = self.scaler.transform(data)
            print(f"✅ Transformed {len(data)} samples using fitted scaler")

        return processed_data.astype(np.float32)

    def create_tf_datasets(self, DStrn, DSopt, batch_size=None):
        """
        创建TensorFlow数据集

        Args:
            DStrn: 训练数据
            DSopt: 验证数据
            batch_size: 批大小

        Returns:
            train_dataset, val_dataset
        """
        import tensorflow as tf

        if batch_size is None:
            batch_size = self.config.DEFAULT_BATCH_SIZE

        # 创建训练数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((DStrn, DStrn))
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # 创建验证数据集
        val_dataset = tf.data.Dataset.from_tensor_slices((DSopt, DSopt))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        print(f"✅ Created TensorFlow datasets:")
        print(f"   Train batches: {len(train_dataset)}")
        print(f"   Val batches: {len(val_dataset)}")
        print(f"   Batch size: {batch_size}")

        return train_dataset, val_dataset

    def create_numpy_datasets(self, DStrn, DSopt):
        """
        创建NumPy数据集（用于简单训练）

        Args:
            DStrn: 训练数据
            DSopt: 验证数据

        Returns:
            (X_train, y_train), (X_val, y_val)
        """
        # 对于自编码器，输入和输出相同
        X_train, y_train = DStrn, DStrn
        X_val, y_val = DSopt, DSopt

        print(f"✅ Created NumPy datasets:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_val shape: {X_val.shape}")

        return (X_train, y_train), (X_val, y_val)

    def get_data_info(self, device_name):
        """获取设备数据信息"""
        return self.data_info.get(device_name, {})

    def save_scaler(self, filepath):
        """保存scaler到文件"""
        import joblib
        joblib.dump(self.scaler, filepath)
        print(f"✅ Scaler saved to: {filepath}")

    def load_scaler(self, filepath):
        """从文件加载scaler"""
        import joblib
        self.scaler = joblib.load(filepath)
        print(f"✅ Scaler loaded from: {filepath}")