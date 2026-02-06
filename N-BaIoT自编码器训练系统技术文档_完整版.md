# N-BaIoT自编码器训练系统 - 完整技术文档

## 文档信息
- **版本**: v3.2 (GUI完整版 - 入侵检测优化版)
- **最后更新**: 2026-02-05
- **修改人**: AI Assistant
- **文档类型**: 系统架构与用户指南
- **适用范围**: N-BaIoT数据集IoT设备异常检测系统
- **更新内容**:
  - 修复入侵检测面板的DStst文件选择功能bug
  - 支持分别选择dstst_data.npy和dstst_labels.npy两个文件
  - 添加默认路径检查和自动生成DStst文件功能
  - 修复评估逻辑，使其与evaluate_anomaly_detection.py脚本的实现一致
  - 创建EvaluationChart组件，实现评估过程的动态性能指标变化曲线
  - 修复保存选项面板功能，确保能正确生成并保存所有相关文件
  - 优化评估过程子面板的动态图表功能，实现时间序列数据点和关键指标变化曲线
  - 修正评估结果数据合理性问题
  - 更新相关模块的实现细节、参数配置及算法逻辑
  - 修正文档中与实际修复结果不一致的描述
  - 补充必要的代码示例和配置说明

---

## 目录

### 一、系统代码架构与实现详解
1. [神经网络模型技术说明](#11-神经网络模型技术说明)
2. [可配置参数说明](#12-可配置参数说明)
3. [系统模块功能解析](#13-系统模块功能解析)
4. [模块间交互关系](#14-模块间交互关系)

### 二、用户界面使用指南
1. [界面布局结构](#21-界面布局结构)
2. [功能区域说明](#22-功能区域说明)
3. [操作流程指南](#23-操作流程指南)
4. [交互逻辑与注意事项](#24-交互逻辑与注意事项)

---

## 一、系统代码架构与实现详解

### 1.1 神经网络模型技术说明

#### 1.1.1 模型架构概述

本系统采用**深度对称自编码器（Deep Symmetric Autoencoder）**作为核心神经网络模型，专门用于IoT设备网络流量异常检测。该模型通过学习正常流量模式的重构能力，建立安全基线，从而识别异常流量。

**核心设计思想**：
- **无监督学习**：仅使用正常流量数据进行训练，无需标注数据
- **特征压缩**：通过编码器将高维特征压缩到低维潜在空间
- **重构学习**：通过解码器重构原始特征，最小化重构误差
- **异常检测**：基于重构误差判断流量是否异常

#### 1.1.2 模型架构详解

**对称编码器-解码器结构**：

```
输入层: 115个神经元 (对应N-BaIoT数据集的115个特征)
    ↓
编码器 (4层):
    Layer 1: 115 → 86 (75%压缩)
    Layer 2: 86 → 58 (50%压缩)
    Layer 3: 58 → 38 (33%压缩)
    Layer 4: 38 → 29 (25%压缩，潜在空间)
    ↓
潜在表示: 29维 (压缩率约25%)
    ↓
解码器 (4层，对称结构):
    Layer 1: 29 → 38 (33%扩展)
    Layer 2: 38 → 58 (50%扩展)
    Layer 3: 58 → 86 (75%扩展)
    Layer 4: 86 → 115 (100%扩展，输出层)
    ↓
输出层: 115个神经元 (重构原始特征)
```

**架构特点**：
1. **对称设计**：编码器和解码器层数和维度比例对称，便于学习可逆映射
2. **渐进压缩**：逐层压缩特征，保留关键信息的同时去除噪声
3. **灵活配置**：支持自定义压缩比例和层数
4. **正则化支持**：集成L2正则化、Dropout、批量归一化等技术

#### 1.1.3 核心算法原理

**自编码器损失函数**：
```python
L(x, x̂) = 1/n ∑(x_i - x̂_i)²
```
其中：
- x：原始输入特征向量
- x̂：重构输出特征向量
- n：特征维度（115）

**训练目标**：最小化重构误差，使模型能够准确重构正常流量特征。

**异常检测原理**：
```python
anomaly_score = MSE(x, x̂) = 1/n ∑(x_i - x̂_i)²
if anomaly_score > threshold:
    判定为异常流量
else:
    判定为正常流量
```

**阈值确定方法**：
- 使用验证集的MSE分布
- 计算均值μ和标准差σ
- 设置阈值 = μ + k·σ（k通常取2-3）

#### 1.1.4 关键技术创新点

1. **三阶段训练策略**：
   - **阶段1 - 初始训练**：建立性能基准，评估模型基础能力
   - **阶段2 - 超参数优化**：网格搜索最优学习率和训练轮数
   - **阶段3 - 最终训练**：使用最优参数训练最终模型

2. **动态学习率调整**：
   - 使用ReduceLROnPlateau回调
   - 当验证损失停滞时自动降低学习率
   - 避免陷入局部最优

3. **早停机制**：
   - 监控验证损失变化
   - 连续N个epoch无改善则停止训练
   - 恢复最佳权重，避免过拟合

4. **GPU内存优化**：
   - 支持GPU内存动态增长
   - 可设置GPU内存上限
   - 完全禁用GPU以节省资源

#### 1.1.5 模型训练与优化方法

**训练优化技术**：

1. **Adam优化器**：
   ```python
   optimizer = Adam(
       learning_rate=0.001,
       beta_1=0.9,
       beta_2=0.999,
       epsilon=1e-7
   )
   ```
   - 自适应学习率
   - 结合动量和RMSProp优点
   - 适合稀疏梯度

2. **L2正则化**：
   ```python
   kernel_regularizer = l2(0.001)
   ```
   - 防止权重过大
   - 提高模型泛化能力

3. **批量归一化**（可选）：
   ```python
   BatchNormalization()
   ```
   - 加速训练收敛
   - 减少内部协变量偏移
   - 提高训练稳定性

4. **Dropout正则化**（可选）：
   ```python
   Dropout(rate=0.0)
   ```
   - 随机丢弃神经元
   - 防止过拟合
   - 提高模型鲁棒性

**超参数搜索空间**：
```python
LEARNING_RATES = [1e-4, 1e-3, 5e-3]  # 学习率搜索
EPOCHS_OPTIONS = [50, 100]            # 训练轮数搜索
BATCH_SIZES = [32, 64, 128]           # 批大小搜索
```

---

### 1.2 可配置参数说明

#### 1.2.1 GPU控制配置

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| USE_GPU | bool | False | True/False | 全局GPU开关 | 有GPU时设为True，无GPU时设为False |
| GPU_MEMORY_LIMIT | int/None | None | 1024-32768或None | GPU内存限制(MB) | 显存不足时设置，如4096(4GB) |
| GPU_MEMORY_GROWTH | bool | True | True/False | GPU内存动态增长 | 建议保持True，避免显存浪费 |
| GPU_DEVICES | str | "0" | "-1","0","0,1"等 | GPU设备选择 | 单GPU用"0"，多GPU用"0,1"，禁用用"-1" |

**使用示例**：
```python
# 完全禁用GPU
USE_GPU = False
GPU_DEVICES = "-1"

# 使用第一个GPU
USE_GPU = True
GPU_DEVICES = "0"

# 限制GPU内存为4GB
USE_GPU = True
GPU_MEMORY_LIMIT = 4096
```

#### 1.2.2 路径配置

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| DATA_ROOT | str | "C:\Users\WWWWG\Desktop\NBaIoT" | 有效路径 | N-BaIoT数据集根目录 | 修改为实际数据集路径 |
| OUTPUT_DIR | str | "./training_results" | 有效路径 | 训练结果输出目录 | 可自定义输出位置 |
| MODEL_SAVE_DIR | str | "./saved_models" | 有效路径 | 模型保存目录 | 与OUTPUT_DIR保持一致 |

#### 1.2.3 设备选择配置

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| ALL_DEVICES | list | [9个设备] | 设备名称列表 | 所有支持的IoT设备 | 添加新设备时更新此列表 |
| SELECTED_DEVICES | list | ["Danmini_Doorbell"] | 设备名称子集或空列表 | 要训练的设备列表 | 空列表=训练所有设备 |

**支持的设备列表**：
```python
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
```

#### 1.2.4 数据配置

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| FEATURE_DIM | int | 115 | 正整数 | N-BaIoT特征维度 | 通常保持115不变 |
| TRAIN_RATIO | float | 1/3 | 0-1 | 训练集比例 | 三等分时为1/3 |
| VAL_RATIO | float | 1/3 | 0-1 | 验证集比例 | 三等分时为1/3 |
| TEST_RATIO | float | 1/3 | 0-1 | 测试集比例 | 三等分时为1/3 |
| TIME_ORDERED | bool | True | True/False | 数据按时间顺序排列 | 假设数据有序时设为True |
| RANDOM_SEED | int | 42 | 任意正整数 | 随机种子 | 保持一致以确保可复现 |

#### 1.2.5 模型架构配置

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| ENCODER_RATIOS | list | [0.75,0.50,0.33,0.25] | 0-1浮点数列表 | 编码器维度比例 | 增加层数时添加新比例 |
| DECODER_RATIOS | list | [0.33,0.50,0.75,1.0] | 0-1浮点数列表 | 解码器维度比例 | 与ENCODER_RATIOS对称 |
| ACTIVATION | str | 'relu' | 'relu','leaky_relu','tanh','sigmoid' | 激活函数 | ReLU适合大多数情况 |
| OUTPUT_ACTIVATION | str/None | None | None,'sigmoid','tanh','relu' | 输出层激活函数 | 通常为None（线性） |
| USE_BATCH_NORM | bool | False | True/False | 批量归一化 | 训练不稳定时可启用 |
| DROPOUT_RATE | float | 0.0 | 0.0-1.0 | Dropout比例 | 过拟合时增加，如0.2-0.5 |
| L2_REGULARIZATION | float | 0.001 | 0.0-0.1 | L2正则化系数 | 过拟合时增加 |

**架构配置示例**：
```python
# 浅层架构（快速训练）
ENCODER_RATIOS = [0.5, 0.25]
DECODER_RATIOS = [0.25, 0.5]

# 深层架构（更强表达能力）
ENCODER_RATIOS = [0.8, 0.6, 0.4, 0.2, 0.1]
DECODER_RATIOS = [0.1, 0.2, 0.4, 0.6, 0.8]

# 强正则化（防止过拟合）
DROPOUT_RATE = 0.3
L2_REGULARIZATION = 0.01
USE_BATCH_NORM = True
```

#### 1.2.6 超参数搜索空间

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| LEARNING_RATES | list | [1e-4,1e-3,5e-3] | 1e-6到1e-2 | 学习率搜索空间 | 快速实验时减少选项 |
| EPOCHS_OPTIONS | list | [50, 100] | 10-1000 | 训练轮数搜索空间 | 根据数据量调整 |
| BATCH_SIZES | list | [32, 64, 128] | 16-512 | 批大小搜索空间 | 显存不足时减小 |

**搜索空间优化建议**：
```python
# 快速实验（减少搜索空间）
LEARNING_RATES = [1e-3, 5e-3]
EPOCHS_OPTIONS = [50]
BATCH_SIZES = [64]

# 精细调优（增加搜索空间）
LEARNING_RATES = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
EPOCHS_OPTIONS = [50, 100, 150, 200]
BATCH_SIZES = [32, 64, 128, 256]
```

#### 1.2.7 训练配置

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| DEFAULT_LEARNING_RATE | float | 0.001 | 1e-6到1e-2 | 默认学习率 | 根据收敛情况调整 |
| DEFAULT_BATCH_SIZE | int | 64 | 16-512 | 默认批大小 | 显存不足时减小 |
| DEFAULT_EPOCHS | int | 100 | 10-1000 | 默认训练轮数 | 根据数据量调整 |
| EARLY_STOPPING_PATIENCE | int | 15 | 5-50 | 早停耐心值 | 训练慢时减小 |
| REDUCE_LR_PATIENCE | int | 10 | 3-20 | 学习率调整耐心值 | 通常保持默认 |
| REDUCE_LR_FACTOR | float | 0.5 | 0.1-0.9 | 学习率衰减因子 | 通常保持0.5 |
| MIN_DELTA | float | 1e-6 | 1e-8到1e-4 | 最小改善阈值 | 通常保持默认 |
| OPTIMIZER | str | 'adam' | 'adam','rmsprop','sgd' | 优化器类型 | Adam适合大多数情况 |
| BETA_1 | float | 0.9 | 0.8-0.999 | Adam动量参数1 | 通常保持默认 |
| BETA_2 | float | 0.999 | 0.9-0.9999 | Adam动量参数2 | 通常保持默认 |
| EPSILON | float | 1e-7 | 1e-9到1e-5 | Adam数值稳定参数 | 通常保持默认 |

#### 1.2.8 文件保存配置

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| SAVE_LOG_FILE | bool | True | True/False | 保存训练日志 | 建议保持True |
| SAVE_MODEL | bool | True | True/False | 保存模型文件 | 需要部署时设为True |
| SAVE_BEST_MODEL_ONLY | bool | True | True/False | 仅保存最佳模型 | 节省磁盘空间 |
| SAVE_TRAINING_HISTORY | bool | True | True/False | 保存训练历史 | 用于后续分析 |
| SAVE_HYPERPARAMETER_TUNING_RESULTS | bool | True | True/False | 保存调优结果 | 用于参数分析 |
| SAVE_SCALER | bool | True | True/False | 保存标准化器 | 推理时必需 |
| SAVE_TENSORBOARD_LOGS | bool | False | True/False | 保存TensorBoard日志 | 需要可视化时启用 |

#### 1.2.9 可视化配置

| 参数名称 | 数据类型 | 默认值 | 取值范围 | 功能描述 | 修改建议 |
|---------|---------|--------|---------|---------|---------|
| PLOT_SAVE | bool | True | True/False | 保存图表到文件 | 建议保持True |
| PLOT_SHOW | bool | False | True/False | 显示图表窗口 | GUI模式下通常为False |
| PLOT_FORMAT | str | 'png' | 'png','jpg','pdf','svg' | 图表格式 | 根据需求选择 |
| PLOT_DPI | int | 300 | 72-600 | 图表分辨率 | 高质量报告用300+ |
| PLOT_STYLE | str | 'seaborn-darkgrid' | matplotlib样式 | 图表样式 | 根据喜好选择 |

**图表类型开关**：
```python
# 训练曲线图表
PLOT_TRAINING_LOSS_CURVE = True      # 训练损失曲线
PLOT_TRAINING_MAE_CURVE = True        # 训练MAE曲线
PLOT_TRAINING_LR_CURVE = True         # 学习率变化曲线

# 超参数调优图表
PLOT_HYPERPARAM_HEATMAP = True      # 超参数热图
PLOT_HYPERPARAM_CONTOUR = True     # 超参数等高线图
PLOT_HYPERPARAM_3D = False         # 超参数3D图

# 损失分析图表
PLOT_LOSS_DISTRIBUTION = True        # 损失分布图
PLOT_LOSS_HISTOGRAM = True          # 损失直方图
PLOT_LOSS_BOX_PLOT = True            # 损失箱线图
PLOT_LOSS_VIOLIN_PLOT = True        # 损失小提琴图

# 模型性能图表
PLOT_PERFORMANCE_METRICS = True      # 性能指标图
PLOT_LEARNING_RATE_SCHEDULE = True     # 学习率调度图
PLOT_GRADIENT_FLOW = False           # 梯度流图

# 数据分析图表
PLOT_DATA_DISTRIBUTION = True        # 数据分布图
PLOT_FEATURE_CORRELATION = False     # 特征相关性图
PLOT_PCA_VISUALIZATION = False       # PCA可视化

# 时间分析图表
PLOT_TRAINING_TIME_ANALYSIS = True  # 训练时间分析
PLOT_EPOCH_TIME_DISTRIBUTION = True  # Epoch时间分布

# 比较图表
PLOT_DEVICE_COMPARISON = True       # 设备比较图
PLOT_PHASE_COMPARISON = True        # 训练阶段比较
PLOT_PERFORMANCE_RANKING = True      # 性能排名图

# 综合报告图表
PLOT_COMPREHENSIVE_SUMMARY = True   # 综合总结图
PLOT_TRAINING_REPORT = True         # 训练报告
```

---

### 1.3 系统模块功能解析

#### 1.3.1 核心模块架构

系统采用**模块化设计**，将功能划分为独立的模块，每个模块负责特定的功能，通过明确的接口进行交互。

**模块层次结构**：
```
user_input_files/
├── app.py                          # 主程序入口
├── config.py                       # 配置管理模块
├── model.py                        # 模型定义模块
├── trainer.py                      # 训练管理模块
├── data_processor.py               # 数据处理模块
├── visualizer.py                  # 可视化模块
├── anomaly_detector.py            # 异常检测模块
├── data_integrator.py            # 数据集成模块
├── evaluate_anomaly_detection.py  # 评估模块
├── core/                          # 核心功能模块
│   ├── signals/                  # 信号定义模块
│   │   ├── __init__.py
│   │   ├── training_signals.py    # 训练信号类
│   │   └── detection_signals.py  # 检测信号类
│   ├── training/                 # 训练功能模块
│   │   └── training_worker.py   # 训练工作线程
│   └── detection/               # 检测功能模块
│       └── intrusion_detection_worker.py  # 检测工作线程
└── gui/                           # GUI界面模块
    ├── components/               # GUI组件模块
    │   └── main_window.py      # 主窗口组件
    ├── panels/                  # GUI面板模块
    │   ├── config_panel.py           # 配置面板
    │   ├── training_control_panel.py  # 训练控制面板
    │   └── intrusion_detection_panel.py  # 入侵检测面板
    └── widgets/                 # GUI组件模块
        └── real_time_chart.py   # 实时图表组件
```

#### 1.3.2 配置管理模块 (config.py)

**功能描述**：
- 集中管理所有系统配置参数
- 提供配置验证和默认值
- 支持环境变量和命令行参数覆盖
- 提供配置显示和导出功能

**关键类和方法**：
```python
class Config:
    """训练配置参数类"""
    
    @classmethod
    def setup_environment(cls):
        """设置环境变量，必须在导入TensorFlow之前调用"""
        # 设置随机种子
        # 配置CUDA_VISIBLE_DEVICES
        # 设置TensorFlow日志级别
        # 创建必要的目录
    
    @classmethod
    def setup_tensorflow(cls):
        """设置TensorFlow配置"""
        # 设置GPU内存增长
        # 设置GPU内存限制
        # 验证GPU可用性
    
    @classmethod
    def get_selected_devices(cls):
        """获取选择的设备列表"""
        # 返回要训练的设备列表
    
    @classmethod
    def display_config(cls):
        """显示当前配置"""
        # 打印所有配置参数
```

**技术特点**：
- **类方法设计**：所有配置通过类属性访问，无需实例化
- **环境设置**：在导入TensorFlow前设置环境变量
- **GPU控制**：通过环境变量精细控制GPU使用
- **目录管理**：自动创建必要的输出目录

#### 1.3.3 数据处理模块 (data_processor.py)

**功能描述**：
- 加载和验证N-BaIoT数据集
- 执行数据划分（训练/验证/测试）
- 进行特征标准化处理
- 创建TensorFlow数据集

**关键类和方法**：
```python
class NBaIoTDataProcessor:
    """N-BaIoT数据处理器"""
    
    def __init__(self, config):
        """初始化数据处理器"""
        self.config = config
        self.scaler = None
    
    def get_available_devices(self):
        """获取可用的设备列表"""
        # 扫描数据目录
        # 返回包含benign_traffic.csv的设备列表
    
    def load_device_data(self, device_name):
        """加载单个设备数据"""
        # 检查文件存在性
        # 读取CSV文件
        # 验证特征维度
        # 处理缺失值和异常值
        # 返回NumPy数组
    
    def split_data_chronologically(self, data):
        """按时间顺序划分数据"""
        # 三等分数据
        # 返回DStrn, DSopt, DStst
    
    def split_data_randomly(self, data):
        """随机划分数据"""
        # 随机打乱数据
        # 三等分数据
        # 返回DStrn, DSopt, DStst
    
    def preprocess_data(self, data, fit_scaler=False):
        """数据标准化处理"""
        # 使用StandardScaler进行Z-score标准化
        # fit_scaler=True: 计算并保存标准化参数
        # fit_scaler=False: 使用已有的标准化参数
        # 返回标准化后的数据
    
    def create_numpy_datasets(self, train_data, val_data):
        """创建NumPy数据集"""
        # 准备训练和验证数据
        # 返回(X_train, y_train), (X_val, y_val)
    
    def save_scaler(self, path):
        """保存标准化器"""
        # 使用joblib保存scaler对象
    
    def load_scaler(self, path):
        """加载标准化器"""
        # 使用joblib加载scaler对象
```

**技术特点**：
- **数据验证**：自动检查文件存在性和数据完整性
- **灵活划分**：支持按时间顺序或随机划分
- **标准化管理**：统一管理训练和推理时的标准化参数
- **错误处理**：完善的异常处理和错误提示

#### 1.3.4 模型定义模块 (model.py)

**功能描述**：
- 定义对称自编码器架构
- 支持多种激活函数和正则化选项
- 提供编码器、解码器和完整模型
- 支持自定义架构配置

**关键类和方法**：
```python
class Autoencoder:
    """对称自编码器类"""
    
    def __init__(self, config):
        """初始化自编码器"""
        self.config = config
        self.model = None
        self.encoder = None
        self.decoder = None
    
    def build(self, input_dim=None):
        """构建自编码器模型"""
        # 计算编码器各层维度
        # 计算解码器各层维度
        # 构建编码器网络
        # 构建解码器网络
        # 组合完整模型
        # 返回Keras模型
    
    def encode(self, x):
        """编码过程"""
        # 将输入编码到潜在空间
        # 返回潜在表示
    
    def decode(self, z):
        """解码过程"""
        # 从潜在空间解码
        # 返回重构输出
    
    def forward(self, x):
        """前向传播"""
        # 完整的编码-解码过程
        # 返回重构输出
    
    def get_latent_representation(self, x):
        """获取潜在表示"""
        # 返回编码后的潜在表示
```

**技术特点**：
- **对称架构**：编码器和解码器对称设计
- **模块化**：编码器和解码器可单独使用
- **灵活配置**：支持多种激活函数和正则化选项
- **动态构建**：根据输入维度自动调整架构

#### 1.3.5 训练管理模块 (trainer.py)

**功能描述**：
- 实现三阶段训练流程
- 管理训练回调函数
- 跟踪训练历史和性能指标
- 保存训练结果和模型

**关键类和方法**：
```python
class AutoencoderTrainer:
    """自编码器训练器"""
    
    def __init__(self, config, device_name):
        """初始化训练器"""
        self.config = config
        self.device_name = device_name
        self.model = None
        self.history = {}
        self.training_history = {
            'initial_train': None,
            'hyperparameter_tuning': [],
            'final_train': None,
            'best_params': None,
            'best_val_loss': float('inf')
        }
        # 创建设备特定的输出目录
    
    def create_callbacks(self, monitor='val_loss', mode='min',
                     patience=None, save_best_only=True):
        """创建训练回调函数"""
        # 早停回调
        # 学习率调整回调
        # 模型检查点回调
        # TensorBoard回调
        # 返回回调列表
    
    def train_initial(self, X_train, y_train, X_val, y_val, input_dim):
        """阶段1：初始训练"""
        # 构建模型
        # 编译模型
        # 训练模型
        # 返回训练结果
    
    def train_with_hyperparameter_tuning(self, X_train, y_train, X_val, y_val, input_dim):
        """阶段2：超参数调优"""
        # 遍历学习率和epoch组合
        # 训练每个组合
        # 记录结果
        # 返回最佳参数
    
    def train_final(self, X_train, y_train, input_dim, best_params):
        """阶段3：最终训练"""
        # 合并训练和验证数据
        # 使用最佳参数构建模型
        # 训练最终模型
        # 返回训练结果
    
    def save_model(self, model, path):
        """保存模型"""
        # 保存完整模型到H5文件
    
    def save_training_history(self):
        """保存训练历史"""
        # 保存训练历史到JSON文件
```

**技术特点**：
- **三阶段训练**：初始训练→超参数优化→最终训练
- **自动回调**：集成早停、学习率调整等回调
- **历史跟踪**：完整记录训练过程和结果
- **模型保存**：自动保存最佳模型和训练历史

#### 1.3.6 可视化模块 (visualizer.py)

**功能描述**：
- 生成训练过程可视化图表
- 支持多种图表类型
- 提供综合训练报告
- 支持设备比较分析

**关键类和方法**：
```python
class ScientificVisualizer:
    """科学可视化器"""
    
    def __init__(self, config):
        """初始化可视化器"""
        self.config = config
    
    def generate_all_plots(self, trainer, device_name, data_info):
        """生成所有图表"""
        # 训练曲线图表
        # 超参数调优图表
        # 损失分析图表
        # 性能指标图表
        # 数据分析图表
        # 时间分析图表
        # 综合报告图表
    
    def plot_training_loss_curve(self, history, title, save_path):
        """绘制训练损失曲线"""
        # 绘制训练和验证损失
        # 保存图表
    
    def plot_hyperparameter_heatmap(self, results, title, save_path):
        """绘制超参数热图"""
        # 绘制学习率vs epochs的热图
        # 保存图表
    
    def plot_loss_distribution(self, losses, title, save_path):
        """绘制损失分布图"""
        # 绘制损失分布
        # 保存图表
    
    def plot_device_comparison(self, results):
        """绘制设备比较图"""
        # 比较多个设备的性能
        # 保存图表
    
    def plot_performance_ranking(self, results):
        """绘制性能排名图"""
        # 对设备性能进行排名
        # 保存图表
    
    def plot_comprehensive_summary(self, trainer, device_name, data_info):
        """绘制综合总结图"""
        # 生成3x3网格的综合图表
        # 包含训练曲线、超参数、损失分析等
```

**技术特点**：
- **多样化图表**：支持20+种不同类型的图表
- **高质量输出**：支持高DPI和多种格式
- **综合报告**：生成完整的训练总结报告
- **比较分析**：支持多设备性能比较

#### 1.3.7 GUI主窗口模块 (gui/components/main_window.py)

**功能描述**：
- 实现应用程序主窗口
- 整合所有GUI组件
- 管理菜单栏和状态栏
- 协调各面板间的交互

**关键类和方法**：
```python
class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.setWindowTitle("N-BaIoT 自编码器训练系统 - GUI版")
        self.setMinimumSize(1400, 900)
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()
    
    def setup_ui(self):
        """设置UI"""
        # 创建选项卡控件
        # 添加训练选项卡
        # 添加入侵检测选项卡
        # 设置布局
    
    def setup_menu(self):
        """设置菜单栏"""
        # 文件菜单：保存/加载配置
        # 工具菜单：清空日志/图表
        # 帮助菜单：关于信息
    
    def setup_statusbar(self):
        """设置状态栏"""
        # 显示训练进度和状态
    
    def save_config(self):
        """保存配置"""
        # 保存当前配置到JSON文件
    
    def load_config(self):
        """加载配置"""
        # 从JSON文件加载配置
    
    def show_about(self):
        """显示关于对话框"""
        # 显示系统版本和修复信息
    
    def update_overall_progress(self, progress):
        """更新整体进度"""
        # 更新状态栏显示的进度
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 检查训练是否在进行
        # 询问用户确认
        # 停止训练或取消关闭
```

**技术特点**：
- **选项卡设计**：训练和检测功能分离
- **实时更新**：实时显示训练进度和状态
- **配置管理**：支持配置的保存和加载
- **优雅退出**：处理训练中的退出操作

#### 1.3.8 配置面板模块 (gui/panels/config_panel.py)

**功能描述**：
- 提供完整的参数配置界面
- 支持多种配置类别
- 实现配置的保存和加载
- 提供设备选择功能

**关键类和方法**：
```python
class ConfigPanel(QWidget):
    """配置面板类"""
    
    def __init__(self, parent=None):
        """初始化配置面板"""
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        # 创建选项卡控件
        # 添加基础配置页面
        # 添加模型架构页面
        # 添加训练参数页面
        # 添加设备选择页面
        # 添加保存选项页面
        # 添加高级选项页面
    
    def create_basic_config(self):
        """创建基础配置页面"""
        # 数据路径配置
        # 输出目录配置
        # GPU设置
        # 特征维度配置
    
    def create_model_config(self):
        """创建模型架构页面"""
        # 激活函数选择
        # 批量归一化设置
        # Dropout率设置
        # L2正则化设置
        # 编码器/解码器比例设置
    
    def create_training_config(self):
        """创建训练参数页面"""
        # 学习率设置
        # 批大小设置
        # 训练轮数设置
        # 早停参数设置
        # 数据划分方式设置
    
    def create_device_config(self):
        """创建设备选择页面"""
        # 显示所有可用设备
        # 支持多选
        # 提供全选/全不选按钮
    
    def create_save_config(self):
        """创建保存选项页面"""
        # 文件保存选项
        # 训练曲线图表选项
        # 超参数调优图表选项
        # 损失分析图表选项
        # 模型性能图表选项
        # 数据分析图表选项
        # 时间分析图表选项
        # 比较图表选项
        # 综合报告图表选项
    
    def get_config(self):
        """获取配置"""
        # 收集所有UI控件的值
        # 返回配置字典
    
    def load_config(self, config):
        """加载配置"""
        # 根据配置字典设置UI控件的值
```

**技术特点**：
- **分类管理**：将配置分为多个类别页面
- **实时预览**：配置修改实时生效
- **设备选择**：支持单选、多选、全选
- **图表控制**：精细控制各种图表的生成

#### 1.3.9 训练控制面板模块 (gui/panels/training_control_panel.py)

**功能描述**：
- 提供训练过程的控制功能
- 显示训练进度和状态
- 实现暂停/继续/停止操作
- 显示实时训练日志

**关键类和方法**：
```python
class TrainingControlPanel(QWidget):
    """训练控制面板类"""
    
    def __init__(self, parent=None):
        """初始化训练控制面板"""
        super().__init__(parent)
        self.main_window = parent
        self.training_worker = None
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        # 创建开始按钮
        # 创建暂停按钮
        # 创建停止按钮
        # 创建进度条
        # 创建状态标签
        # 设置布局
    
    def start_training(self):
        """开始训练"""
        # 获取配置
        # 创建训练工作线程
        # 连接信号
        # 启动训练
    
    def pause_training(self):
        """暂停训练"""
        # 调用工作线程的pause方法
        # 更新按钮状态
    
    def resume_training(self):
        """继续训练"""
        # 调用工作线程的resume方法
        # 更新按钮状态
    
    def stop_training(self):
        """停止训练"""
        # 调用工作线程的stop方法
        # 更新按钮状态
    
    def on_training_started(self, device_name):
        """训练开始回调"""
        # 更新状态显示
        # 禁用开始按钮
        # 启用暂停和停止按钮
    
    def on_training_finished(self, data):
        """训练完成回调"""
        # 显示训练结果
        # 生成设备比较图表
        # 恢复按钮状态
    
    def on_training_error(self, error):
        """训练错误回调"""
        # 显示错误信息
        # 恢复按钮状态
    
    def on_epoch_completed(self, data):
        """Epoch完成回调"""
        # 更新图表
        # 更新进度显示
    
    def on_log_received(self, log):
        """日志接收回调"""
        # 添加日志到日志窗口
    
    def on_status_update(self, status):
        """状态更新回调"""
        # 更新状态标签
```

**技术特点**：
- **实时控制**：支持训练过程中的暂停、继续、停止
- **进度显示**：实时显示训练进度和状态
- **日志输出**：实时显示训练日志信息
- **图表更新**：实时更新训练损失曲线

#### 1.3.10 入侵检测面板模块 (gui/panels/intrusion_detection_panel.py)

**功能描述**：
- 提供入侵检测评估功能
- 支持DStst文件选择（分别选择数据和标签文件）
- 实现评估过程控制
- 显示评估结果和性能指标
- 自动检查和生成DStst文件

**技术问题修复**：

**问题1：DStst文件选择功能bug**
- **问题描述**：原实现只能选择一个文件，但实际需要两个文件（dstst_data.npy和dstst_labels.npy）
- **解决方案**：
  - 将文件选择分为两个独立部分：
    - DStst数据文件选择（dstst_data.npy）
    - DStst标签文件选择（dstst_labels.npy）
  - 添加两个独立的文件选择按钮和文本框
  - 更新文件验证逻辑，支持两个独立文件的验证

**问题2：文件验证逻辑错误**
- **问题描述**：选择dstst_data.npy文件后提示"文件格式不正确"
- **解决方案**：
  - 修改验证逻辑，分别验证数据文件和标签文件
  - 检查文件是否存在
  - 验证数据和标签数量是否匹配
  - 显示详细的验证结果和样本数量

**问题3：缺少默认路径检查**
- **问题描述**：没有默认路径检查功能，用户需要手动选择文件
- **解决方案**：
  - 添加设备选择变化时的自动检查功能
  - 默认路径为：`Config.OUTPUT_DIR / device_name /`
  - 如果找到文件，自动填充文件路径并验证
  - 如果没有找到文件，显示橙色提示信息

**问题4：缺少自动生成DStst文件功能**
- **问题描述**：如果没有找到DStst文件，用户需要手动生成
- **解决方案**：
  - 添加自动生成DStst文件功能
  - 在开始评估前检查DStst文件是否存在
  - 如果不存在，弹出对话框询问用户是否自动生成
  - 调用data_integrator模块生成DStst文件
  - 生成成功后自动填充文件路径并验证

**关键类和方法**：
```python
class IntrusionDetectionPanel(QWidget):
    """入侵检测与评估面板类"""

    def __init__(self, parent=None):
        """初始化入侵检测面板"""
        super().__init__(parent)
        self.main_window = parent
        self.detection_worker = None
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        # 创建选项卡控件
        # 添加文件选择选项卡
        # 添加评估过程选项卡
        # 添加结果显示选项卡
        # 添加保存选项选项卡

    def create_file_selection_tab(self):
        """创建文件选择选项卡"""
        # 设备选择
        # DStst数据文件选择（dstst_data.npy）
        # DStst标签文件选择（dstst_labels.npy）
        # 模型文件选择

    def browse_dstst_data_file(self):
        """浏览DStst数据文件"""
        # 打开文件选择对话框
        # 选择dstst_data.npy文件
        # 验证文件

    def browse_dstst_labels_file(self):
        """浏览DStst标签文件"""
        # 打开文件选择对话框
        # 选择dstst_labels.npy文件
        # 验证文件

    def validate_dstst_files(self):
        """验证DStst文件（数据文件和标签文件）"""
        # 检查两个文件是否都已选择
        # 检查文件是否存在
        # 验证数据和标签数量是否匹配
        # 显示详细的验证结果

    def on_device_changed(self, device_name: str):
        """设备选择变化回调"""
        # 检查默认路径下是否存在DStst文件
        # 自动填充文件路径或显示提示

    def check_dstst_files(self):
        """检查默认路径下是否存在DStst文件"""
        # 构建默认路径
        # 检查dstst_data.npy和dstst_labels.npy是否存在
        # 自动填充文件路径或显示提示

    def generate_dstst_files(self):
        """自动生成DStst文件"""
        # 调用data_integrator模块
        # 创建DStst数据集
        # 保存到默认路径
        # 自动填充文件路径并验证

    def start_evaluation(self):
        """开始评估"""
        # 检查DStst文件是否存在
        # 如果不存在，询问用户是否自动生成
        # 获取配置
        # 创建检测工作线程
        # 连接信号
        # 启动评估

    def create_evaluation_tab(self):
        """创建评估过程选项卡"""
        # 开始评估按钮
        # 进度条
        # 状态标签
        # 日志显示区域

    def create_results_tab(self):
        """创建结果显示选项卡"""
        # 性能指标显示
        # 混淆矩阵显示
        # ROC曲线显示
        # 详细结果表格

    def create_save_tab(self):
        """创建保存选项选项卡"""
        # 保存评估数据
        # 保存性能指标
        # 保存可视化图表
        # 保存详细报告

    def on_evaluation_started(self):
        """评估开始回调"""
        # 更新状态显示
        # 禁用开始按钮

    def on_evaluation_finished(self, results):
        """评估完成回调"""
        # 显示评估结果
        # 生成可视化图表
        # 恢复按钮状态

    def on_evaluation_error(self, error):
        """评估错误回调"""
        # 显示错误信息
        # 恢复按钮状态

    def on_progress_update(self, data):
        """进度更新回调"""
        # 更新进度条
        # 更新状态标签

    def on_log_received(self, log):
        """日志接收回调"""
        # 添加日志到日志窗口

    def on_status_update(self, status):
        """状态更新回调"""
        # 更新状态标签

    def on_file_generated(self, message: str):
        """文件生成回调"""
        # 显示文件生成成功消息
```

**技术特点**：
- **完整评估**：支持完整的入侵检测评估流程
- **多指标显示**：显示准确率、精确率、召回率、F1分数等
- **可视化支持**：生成ROC曲线、混淆矩阵等图表
- **结果保存**：支持保存评估结果和报告
- **智能文件管理**：自动检查和生成DStst文件
- **用户友好**：提供详细的文件状态提示和错误信息

#### 1.3.11 实时图表组件模块 (gui/widgets/real_time_chart.py)

**功能描述**：
- 实现实时训练损失曲线显示
- 支持动态更新数据点
- 限制显示最近的200个epoch数据
- 提供清空图表功能

**关键类和方法**：
```python
class RealTimeChart(QWidget):
    """实时训练曲线图表组件"""
    
    def __init__(self, parent=None):
        """初始化实时图表组件"""
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.max_data_points = 200
        self.setup_plot()
    
    def setup_plot(self):
        """设置图表"""
        # 创建matplotlib图表
        # 设置图表样式
        # 初始化数据列表
        # 设置图表样式
    
    def setup_chart_style(self):
        """设置图表样式"""
        # 使用dark_background样式
        # 初始化图表
        # 设置坐标轴颜色
        # 初始化线条
        # 设置标签和标题
    
    def update_chart(self, epoch, train_loss, val_loss, phase, total_epochs):
        """更新图表数据"""
        # 添加数据点
        # 限制数据点数量
        # 更新线条数据
        # 更新当前点标记
        # 更新标题
        # 动态调整坐标轴
        # 刷新图表
    
    def clear_chart(self):
        """清空图表"""
        # 清空数据列表
        # 更新线条数据
        # 重置坐标轴范围
        # 刷新图表
    
    def resizeEvent(self, event):
        """响应窗口大小变化"""
        # 调整图表布局
```

**技术特点**：
- **实时更新**：训练过程中实时更新曲线
- **性能优化**：限制显示200个点，避免性能问题
- **自适应坐标轴**：根据数据自动调整坐标轴范围
- **高亮标记**：高亮显示当前数据点

#### 1.3.12 训练工作线程模块 (core/training/training_worker.py)

**功能描述**：
- 在后台线程中执行训练任务
- 实现训练过程的控制（暂停/继续/停止）
- 通过信号与GUI通信
- 执行三阶段训练流程

**关键类和方法**：
```python
class TrainingWorker(QThread):
    """训练工作线程"""
    
    def __init__(self, config, signals):
        """初始化训练工作线程"""
        super().__init__()
        self.config = config
        self.signals = signals
        self.is_paused = False
        self.should_stop = False
        self.is_running = False
        self.mutex = QMutex()
    
    def run(self):
        """执行训练"""
        # 设置环境
        # 初始化数据处理器
        # 获取要训练的设备列表
        # 初始化可视化器
        # 遍历设备进行训练
        # 训练完成
    
    def _setup_environment(self):
        """设置环境"""
        # 设置GPU配置
        # 设置数据路径
        # 设置训练参数
        # 设置保存选项
        # 设置图表选项
    
    def _get_devices_to_train(self, data_processor):
        """获取要训练的设备列表"""
        # 获取选择的设备
        # 验证设备有效性
        # 返回设备列表
    
    def _train_device(self, device_name, data_processor, visualizer):
        """训练单个设备"""
        # 加载数据
        # 划分数据
        # 预处理数据
        # 阶段1：初始训练
        # 阶段2：超参数调优
        # 阶段3：最终训练
        # 保存模型
        # 生成可视化
        # 返回训练结果
    
    def _train_with_callback(self, trainer, X_train, y_train, X_val, y_val,
                         phase_name, total_epochs, control_callback, input_dim):
        """带回调的训练"""
        # 创建模型
        # 编译模型
        # 创建回调
        # 训练模型
        # 返回训练结果
    
    def _train_with_hyperparameter_tuning(self, trainer, X_train, y_train, X_val, y_val,
                                       control_callback, input_dim):
        """超参数调优"""
        # 遍历学习率和epoch组合
        # 训练每个组合
        # 记录结果
        # 返回最佳参数
    
    def pause(self):
        """暂停训练"""
        # 设置暂停标志
    
    def resume(self):
        """继续训练"""
        # 清除暂停标志
    
    def stop(self):
        """停止训练"""
        # 设置停止标志
```

**技术特点**：
- **后台执行**：在独立线程中执行训练，不阻塞GUI
- **线程安全**：使用互斥锁保护共享变量
- **信号通信**：通过PyQt信号与GUI通信
- **状态控制**：支持暂停、继续、停止操作

#### 1.3.13 入侵检测工作线程模块 (core/detection/intrusion_detection_worker.py)

**功能描述**：
- 在后台线程中执行入侵检测评估
- 加载训练好的模型
- 加载DStst测试数据（分别加载数据和标签文件）
- 执行异常检测评估
- 计算性能指标

**配置参数**：
```python
config = {
    'device_name': str,              # 设备名称
    'dstst_data_file': str,         # DStst数据文件路径 (dstst_data.npy)
    'dstst_labels_file': str,        # DStst标签文件路径 (dstst_labels.npy)
    'model_file': str,               # 模型文件路径
    'save_path': str,                # 保存路径
    'save_data': bool,               # 是否保存评估数据
    'save_images': bool              # 是否保存评估图表
}
```

**关键类和方法**：
```python
class IntrusionDetectionWorker(QThread):
    """入侵检测工作线程"""

    def __init__(self, config: dict, signals: IntrusionDetectionSignals):
        """初始化入侵检测工作线程

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
        """执行入侵检测评估"""
        # 获取配置参数
        # 验证配置参数
        # 验证文件存在性
        # 加载DStst文件（数据和标签）
        # 加载模型文件
        # 加载标准化器
        # 执行异常检测
        # 计算性能指标
        # 生成可视化
        # 评估完成

    def load_model(self, model_path):
        """加载模型

        Args:
            model_path: 模型文件路径

        Returns:
            加载的模型对象
        """
        # 加载H5模型文件
        # 返回模型对象

    def load_scaler(self, scaler_path):
        """加载标准化器

        Args:
            scaler_path: 标准化器文件路径

        Returns:
            加载的scaler对象
        """
        # 加载scaler.pkl文件
        # 返回scaler对象

    def load_test_data(self, data_path, labels_path):
        """加载测试数据

        Args:
            data_path: 数据文件路径 (dstst_data.npy)
            labels_path: 标签文件路径 (dstst_labels.npy)

        Returns:
            测试数据和标签
        """
        # 加载dstst_data.npy
        # 加载dstst_labels.npy
        # 验证数据和标签数量是否匹配
        # 返回测试数据和标签

    def detect_anomalies(self, model, scaler, test_data):
        """检测异常

        Args:
            model: 训练好的模型
            scaler: 标准化器
            test_data: 测试数据

        Returns:
            异常检测结果
        """
        # 标准化测试数据
        # 模型推理
        # 计算重构误差
        # 应用阈值判断
        # 返回异常检测结果

    def calculate_metrics(self, y_true, y_pred):
        """计算性能指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            指标字典（准确率、精确率、召回率、F1分数、假阳性率等）
        """
        # 计算准确率 (Accuracy)
        # 计算精确率 (Precision)
        # 计算召回率 (Recall)
        # 计算F1分数 (F1 Score)
        # 计算假阳性率 (FPR)
        # 计算混淆矩阵 (TP, TN, FP, FN)
        # 返回指标字典
```

**算法逻辑**：

1. **文件验证**：
   - 验证dstst_data_file和dstst_labels_file是否都已提供
   - 验证两个文件是否存在
   - 验证数据和标签数量是否匹配

2. **数据加载**：
   - 使用np.load()加载dstst_data.npy
   - 使用np.load()加载dstst_labels.npy
   - 验证加载后的数据和标签数量是否匹配

3. **异常检测**：
   - 使用标准化器对测试数据进行标准化
   - 使用模型进行推理，得到重构输出
   - 计算重构误差（MSE）
   - 应用阈值判断是否为异常

4. **性能评估**：
   - 计算准确率：(TP + TN) / (TP + TN + FP + FN)
   - 计算精确率：TP / (TP + FP)
   - 计算召回率：TP / (TP + FN)
   - 计算F1分数：2 * (Precision * Recall) / (Precision + Recall)
   - 计算假阳性率：FP / (FP + TN)

**技术特点**：
- **后台执行**：在独立线程中执行评估，不阻塞GUI
- **模型加载**：支持加载训练好的模型
- **异常检测**：基于重构误差进行异常检测
- **性能评估**：计算多种性能指标
- **文件验证**：完善的文件验证逻辑，确保数据和标签的一致性

#### 1.3.14 信号定义模块 (core/signals/)

**功能描述**：
- 定义训练和检测过程中的信号
- 实现线程间通信
- 支持实时状态更新
- 提供错误处理机制

**训练信号类 (training_signals.py)**：
```python
class TrainingSignals(QObject):
    """训练信号类"""
    
    started = pyqtSignal(str)                    # 训练开始信号
    finished = pyqtSignal(dict)                 # 训练完成信号
    error = pyqtSignal(str)                     # 训练错误信号
    progress = pyqtSignal(dict)                 # 进度更新信号
    epoch_completed = pyqtSignal(dict)           # Epoch完成信号
    device_completed = pyqtSignal(dict)          # 设备完成信号
    status_update = pyqtSignal(str)             # 状态更新信号
    log = pyqtSignal(str)                       # 日志信号
```

**检测信号类 (detection_signals.py)**：
```python
class IntrusionDetectionSignals(QObject):
    """入侵检测信号类"""
    
    started = pyqtSignal()                      # 评估开始信号
    finished = pyqtSignal(dict)                 # 评估完成信号
    error = pyqtSignal(str)                     # 评估错误信号
    progress = pyqtSignal(dict)                 # 进度更新信号
    status_update = pyqtSignal(str)             # 状态更新信号
    log = pyqtSignal(str)                       # 日志信号
    data_update = pyqtSignal(dict)              # 数据更新信号
```

**技术特点**：
- **PyQt信号**：使用PyQt的信号槽机制
- **线程安全**：支持跨线程通信
- **丰富信号**：覆盖训练和检测的各个阶段
- **错误处理**：提供错误信号用于异常处理

---

### 1.4 模块间交互关系

#### 1.4.1 系统整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     主程序入口 (app.py)                      │
│                  启动GUI应用程序                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                 主窗口 (MainWindow)                            │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              选项卡控件 (QTabWidget)              │   │
│  │  ┌────────────────┐  ┌──────────────────────┐  │   │
│  │  │  模型训练选项卡  │  │ 入侵检测与评估选项卡 │  │   │
│  │  └────────────────┘  └──────────────────────┘  │   │
│  └───────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│  训练选项卡      │    │  检测选项卡          │
│  ┌────────────┐  │    │  ┌────────────────┐  │
│  │配置面板    │  │    │  │检测面板        │  │
│  │(ConfigPanel)│  │    │  │(Intrusion...   │  │
│  └────────────┘  │    │  └────────────────┘  │
│  ┌────────────┐  │    │  ┌────────────────┐  │
│  │训练控制面板 │  │    │  │文件选择        │  │
│  │(Training... │  │    │  │评估过程        │  │
│  └────────────┘  │    │  │结果显示        │  │
│  ┌────────────┐  │    │  │保存选项        │  │
│  │实时图表    │  │    │  └────────────────┘  │
│  │(RealTime... │  │    └──────────────────────┘
│  └────────────┘  │
│  ┌────────────┐  │
│  │日志显示    │  │
│  └────────────┘  │
└──────────────────┘
```

#### 1.4.2 训练流程数据流向

```
用户操作 (GUI)
    │
    ▼
配置面板 (ConfigPanel)
    │ 获取配置参数
    ▼
训练控制面板 (TrainingControlPanel)
    │ 启动训练
    ▼
训练工作线程 (TrainingWorker)
    │
    ├─→ 设置环境 (Config.setup_environment)
    │
    ├─→ 初始化数据处理器 (NBaIoTDataProcessor)
    │   │
    │   ├─→ 加载数据 (load_device_data)
    │   │
    │   ├─→ 划分数据 (split_data_chronologically)
    │   │
    │   └─→ 预处理数据 (preprocess_data)
    │
    ├─→ 初始化训练器 (AutoencoderTrainer)
    │   │
    │   ├─→ 阶段1: 初始训练
    │   │   │
    │   │   ├─→ 构建模型 (Autoencoder.build)
    │   │   │
    │   │   ├─→ 编译模型
    │   │   │
    │   │   └─→ 训练模型 (model.fit)
    │   │
    │   ├─→ 阶段2: 超参数调优
    │   │   │
    │   │   ├─→ 遍历超参数组合
    │   │   │
    │   │   ├─→ 训练每个组合
    │   │   │
    │   │   └─→ 选择最佳参数
    │   │
    │   └─→ 阶段3: 最终训练
    │       │
    │       ├─→ 合并训练数据
    │       │
    │       ├─→ 使用最佳参数训练
    │       │
    │       └─→ 保存模型 (model.save)
    │
    ├─→ 初始化可视化器 (ScientificVisualizer)
    │   │
    │   └─→ 生成图表 (generate_all_plots)
    │
    └─→ 发送信号 (TrainingSignals)
        │
        ├─→ started: 训练开始
        ├─→ progress: 进度更新
        ├─→ epoch_completed: Epoch完成
        ├─→ device_completed: 设备完成
        ├─→ finished: 训练完成
        ├─→ error: 训练错误
        └─→ log: 日志信息
            │
            ▼
        GUI更新 (MainWindow)
            │
            ├─→ 更新实时图表 (RealTimeChart.update_chart)
            │
            ├─→ 更新进度条 (TrainingControlPanel)
            │
            ├─→ 更新状态标签 (TrainingControlPanel)
            │
            └─→ 添加日志信息 (MainWindow.log_widget)
```

#### 1.4.3 检测流程数据流向

```
用户操作 (GUI)
    │
    ▼
检测面板 (IntrusionDetectionPanel)
    │ 选择文件和模型
    ▼
检测工作线程 (IntrusionDetectionWorker)
    │
    ├─→ 加载模型 (load_model)
    │
    ├─→ 加载标准化器 (load_scaler)
    │
    ├─→ 加载测试数据 (load_test_data)
    │
    ├─→ 执行异常检测 (detect_anomalies)
    │   │
    │   ├─→ 标准化数据
    │   │
    │   ├─→ 模型推理 (model.predict)
    │   │
    │   ├─→ 计算重构误差
    │   │
    │   └─→ 应用阈值判断
    │
    ├─→ 计算性能指标 (calculate_metrics)
    │   │
    │   ├─→ 准确率 (Accuracy)
    │   │
    │   ├─→ 精确率 (Precision)
    │   │
    │   ├─→ 召回率 (Recall)
    │   │
    │   ├─→ F1分数 (F1 Score)
    │   │
    │   └─→ 假阳性率 (FPR)
    │
    └─→ 发送信号 (IntrusionDetectionSignals)
        │
        ├─→ started: 评估开始
        ├─→ progress: 进度更新
        ├─→ finished: 评估完成
        ├─→ error: 评估错误
        └─→ log: 日志信息
            │
            ▼
        GUI更新 (MainWindow)
            │
            ├─→ 更新进度条 (IntrusionDetectionPanel)
            │
            ├─→ 更新状态标签 (IntrusionDetectionPanel)
            │
            ├─→ 显示性能指标 (IntrusionDetectionPanel)
            │
            └─→ 添加日志信息 (IntrusionDetectionPanel)
```

#### 1.4.4 模块依赖关系

```
app.py (主程序入口)
    │
    ├─→ gui.components.main_window (主窗口)
    │       │
    │       ├─→ gui.panels.config_panel (配置面板)
    │       │       │
    │       │       ├─→ config (配置管理)
    │       │       │
    │       │       └─→ PyQt5 (GUI框架)
    │       │
    │       ├─→ gui.panels.training_control_panel (训练控制面板)
    │       │       │
    │       │       ├─→ core.signals.training_signals (训练信号)
    │       │       │
    │       │       ├─→ core.training.training_worker (训练工作线程)
    │       │       │       │
    │       │       │       ├─→ config (配置管理)
    │       │       │       │
    │       │       │       ├─→ data_processor (数据处理)
    │       │       │       │
    │       │       │       ├─→ model (模型定义)
    │       │       │       │
    │       │       │       ├─→ trainer (训练管理)
    │       │       │       │
    │       │       │       └─→ visualizer (可视化)
    │       │       │
    │       │       └─→ PyQt5 (GUI框架)
    │       │
    │       ├─→ gui.panels.intrusion_detection_panel (检测面板)
    │       │       │
    │       │       ├─→ core.signals.detection_signals (检测信号)
    │       │       │
    │       │       ├─→ core.detection.intrusion_detection_worker (检测工作线程)
    │       │       │       │
    │       │       │       ├─→ config (配置管理)
    │       │       │       │
    │       │       │       ├─→ data_processor (数据处理)
    │       │       │       │
    │       │       │       ├─→ model (模型定义)
    │       │       │       │
    │       │       │       └─→ visualizer (可视化)
    │       │       │
    │       │       └─→ PyQt5 (GUI框架)
    │       │
    │       └─→ gui.widgets.real_time_chart (实时图表)
    │               │
    │               ├─→ PyQt5 (GUI框架)
    │               │
    │               └─→ matplotlib (绘图库)
    │
    └─→ PyQt5 (GUI框架)
```

#### 1.4.5 接口定义

**配置接口 (Config)**：
```python
class Config:
    """配置管理接口"""
    
    @classmethod
    def setup_environment(cls) -> bool:
        """设置环境变量"""
        pass
    
    @classmethod
    def setup_tensorflow(cls) -> bool:
        """设置TensorFlow配置"""
        pass
    
    @classmethod
    def get_selected_devices(cls) -> List[str]:
        """获取选择的设备列表"""
        pass
    
    @classmethod
    def display_config(cls) -> None:
        """显示当前配置"""
        pass
```

**数据处理接口 (NBaIoTDataProcessor)**：
```python
class NBaIoTDataProcessor:
    """数据处理接口"""
    
    def get_available_devices(self) -> List[str]:
        """获取可用的设备列表"""
        pass
    
    def load_device_data(self, device_name: str) -> np.ndarray:
        """加载单个设备数据"""
        pass
    
    def split_data_chronologically(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按时间顺序划分数据"""
        pass
    
    def preprocess_data(self, data: np.ndarray, fit_scaler: bool) -> np.ndarray:
        """数据标准化处理"""
        pass
    
    def create_numpy_datasets(self, train_data: np.ndarray, val_data: np.ndarray) -> Tuple[Tuple, Tuple]:
        """创建NumPy数据集"""
        pass
```

**模型接口 (Autoencoder)**：
```python
class Autoencoder:
    """模型接口"""
    
    def build(self, input_dim: int = None) -> tf.keras.Model:
        """构建自编码器模型"""
        pass
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """编码过程"""
        pass
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """解码过程"""
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        pass
```

**训练器接口 (AutoencoderTrainer)**：
```python
class AutoencoderTrainer:
    """训练器接口"""
    
    def create_callbacks(self, monitor: str = 'val_loss', mode: str = 'min',
                     patience: int = None, save_best_only: bool = True) -> List[Callback]:
        """创建训练回调函数"""
        pass
    
    def train_initial(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    input_dim: int) -> Dict:
        """阶段1：初始训练"""
        pass
    
    def train_with_hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray,
                                       input_dim: int) -> Dict:
        """阶段2：超参数调优"""
        pass
    
    def train_final(self, X_train: np.ndarray, y_train: np.ndarray,
                  input_dim: int, best_params: Dict) -> Dict:
        """阶段3：最终训练"""
        pass
```

**可视化接口 (ScientificVisualizer)**：
```python
class ScientificVisualizer:
    """可视化接口"""
    
    def generate_all_plots(self, trainer: AutoencoderTrainer,
                         device_name: str, data_info: Dict) -> None:
        """生成所有图表"""
        pass
    
    def plot_training_loss_curve(self, history: Dict, title: str,
                              save_path: str) -> None:
        """绘制训练损失曲线"""
        pass
    
    def plot_hyperparameter_heatmap(self, results: List[Dict], title: str,
                                  save_path: str) -> None:
        """绘制超参数热图"""
        pass
    
    def plot_device_comparison(self, results: List[Dict]) -> None:
        """绘制设备比较图"""
        pass
```

---

## 二、用户界面使用指南

### 2.1 界面布局结构

#### 2.1.1 主窗口布局

系统主窗口采用**选项卡式布局**，将功能分为两个主要选项卡：

**主窗口结构**：
```
┌─────────────────────────────────────────────────────────────────┐
│ 菜单栏: 文件 | 工具 | 帮助                              │
├─────────────────────────────────────────────────────────────────┤
│ 选项卡: [模型训练] [入侵检测与评估]                        │
├─────────────────────────────────────────────────────────────────┤
│                                                           │
│  选项卡内容区域                                             │
│                                                           │
│                                                           │
│                                                           │
│                                                           │
├─────────────────────────────────────────────────────────────────┤
│ 状态栏: 就绪 - 请配置参数并点击开始训练                      │
└─────────────────────────────────────────────────────────────────┘
```

**菜单栏功能**：
- **文件菜单**：
  - 保存配置：将当前配置保存到JSON文件
  - 加载配置：从JSON文件加载配置
  - 退出：退出应用程序

- **工具菜单**：
  - 清空日志：清空训练日志窗口
  - 清空图表：清空实时训练损失曲线

- **帮助菜单**：
  - 关于：显示系统版本和修复信息

#### 2.1.2 模型训练选项卡布局

**训练选项卡采用分割布局**，分为上下两部分：

**上半部分（配置和图表）**：
```
┌─────────────────────────────────────────────────────────────────┐
│  配置面板 (左)          │  实时图表和进度 (右)         │
│  ┌────────────────────┐    │  ┌──────────────────────┐   │
│  │ 基础配置        │    │  │ 📈 Real-time       │   │
│  │ 模型架构        │    │  │    Training Loss    │   │
│  │ 训练参数        │    │  │    Curve           │   │
│  │ 设备选择        │    │  │    (Max 200        │   │
│  │ 保存选项        │    │  │    epochs)         │   │
│  │ 高级选项        │    │  │                    │   │
│  └────────────────────┘    │  │    [图表区域]      │   │
│                          │  └──────────────────────┘   │
│                          │  ┌──────────────────────┐   │
│                          │  │ Epoch: 0/0        │   │
│                          │  │ 训练损失: -       │   │
│                          │  │ 验证损失: -       │   │
│                          │  │ 阶段: -          │   │
│                          │  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**下半部分（日志和控制）**：
```
┌─────────────────────────────────────────────────────────────────┐
│  训练日志 (左)           │  训练控制 (右)              │
│  ┌────────────────────┐    │  ┌──────────────────────┐   │
│  │ 📋 训练日志      │    │  │ [▶ 开始训练]       │   │
│  │                  │    │  │ [⏸ 暂停]         │   │
│  │ [日志内容区域]    │    │  │ [⏹ 停止]         │   │
│  │                  │    │  │                    │   │
│  │                  │    │  │ [进度条]           │   │
│  └────────────────────┘    │  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.1.3 入侵检测与评估选项卡布局

**检测选项卡采用选项卡布局**，分为四个子选项卡：

```
┌─────────────────────────────────────────────────────────────────┐
│  选项卡: [文件选择] [评估过程] [结果显示] [保存选项]      │
├─────────────────────────────────────────────────────────────────┤
│                                                           │
│  子选项卡内容区域                                           │
│                                                           │
│                                                           │
│                                                           │
│                                                           │
│                                                           │
│                                                           │
│                                                           │
│                                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 功能区域说明

#### 2.2.1 配置面板功能区域

**基础配置页面**：
- **数据根目录**：设置N-BaIoT数据集的根目录路径
- **输出目录**：设置训练结果输出目录路径
- **GPU设置**：
  - 启用GPU加速：勾选启用GPU训练
  - GPU内存限制：设置GPU内存上限（MB）
- **特征维度**：设置输入特征维度（默认115）

**模型架构页面**：
- **激活函数**：选择ReLU、Leaky ReLU、Tanh或Sigmoid
- **批量归一化**：启用或禁用批量归一化
- **Dropout率**：设置Dropout比例（0.0-1.0）
- **L2正则化**：设置L2正则化系数
- **编码器维度比例**：设置编码器各层的压缩比例
- **解码器维度比例**：设置解码器各层的扩展比例

**训练参数页面**：
- **默认学习率**：设置初始学习率
- **默认批大小**：设置训练批大小
- **默认训练轮数**：设置训练轮数
- **早停耐心值**：设置早停机制的耐心值
- **学习率调整耐心值**：设置学习率调整的耐心值
- **学习率调整因子**：设置学习率衰减因子
- **数据划分方式**：选择按时间顺序或随机划分
- **随机种子**：设置随机种子以确保可复现性
- **可视化设置**：勾选保存图表到文件

**设备选择页面**：
- **设备列表**：显示所有可用的IoT设备
- **多选功能**：支持选择多个设备进行训练
- **全选按钮**：选择所有设备
- **全不选按钮**：取消所有设备选择
- **设备统计**：显示设备总数和已选数量

**保存选项页面**：
- **文件保存选项**：
  - 保存训练日志
  - 保存模型文件
  - 仅保存最佳模型
  - 保存训练历史
  - 保存超参数调优结果
  - 保存数据标准化器
  - 保存TensorBoard日志
  - 保存可视化图表

- **训练曲线图表**：
  - 训练损失曲线
  - 训练MAE曲线
  - 学习率变化曲线

- **超参数调优图表**：
  - 超参数热图
  - 超参数等高线图
  - 超参数3D图

- **损失分析图表**：
  - 损失分布图
  - 损失直方图
  - 损失箱线图
  - 损失小提琴图

- **模型性能图表**：
  - 性能指标图
  - 学习率调度图
  - 梯度流图

- **数据分析图表**：
  - 数据分布图
  - 特征相关性图
  - PCA可视化

- **时间分析图表**：
  - 训练时间分析
  - Epoch时间分布

- **比较图表**：
  - 设备比较图
  - 训练阶段比较
  - 性能排名图

- **综合报告图表**：
  - 综合总结图
  - 训练报告

**高级选项页面**：
- **学习率搜索空间**：设置学习率搜索范围
- **训练轮数搜索空间**：设置训练轮数搜索范围
- **批大小搜索空间**：设置批大小搜索范围
- **输出激活函数**：设置输出层激活函数
- **优化器**：选择优化器类型（Adam、RMSprop、SGD）

#### 2.2.2 训练控制面板功能区域

**控制按钮**：
- **开始训练按钮**：启动训练过程
  - 按钮样式：绿色背景，白色文字
  - 功能：初始化训练工作线程并启动训练
  - 状态：训练进行中时禁用

- **暂停按钮**：暂停训练过程
  - 按钮样式：黄色背景，白色文字
  - 功能：暂停训练工作线程
  - 状态：训练进行中时启用，暂停时显示"继续"

- **停止按钮**：停止训练过程
  - 按钮样式：红色背景，白色文字
  - 功能：停止训练工作线程
  - 状态：训练进行中时启用

**进度显示**：
- **进度条**：显示训练整体进度
  - 范围：0-100%
  - 更新：根据训练进度实时更新

- **状态标签**：显示当前训练状态
  - 内容：显示训练阶段、设备名称、进度等
  - 样式：根据状态改变颜色

#### 2.2.3 实时图表功能区域

**图表显示**：
- **训练损失曲线**：实时显示训练和验证损失
  - X轴：Epoch数
  - Y轴：损失值（MSE）
  - 线条：蓝色为训练损失，红色为验证损失
  - 当前点：绿色圆点标记当前Epoch

- **图表标题**：显示当前训练阶段和进度
  - 格式："{阶段名称} - Loss Curve (Epoch {当前}/{总})"
  - 更新：每个Epoch完成后更新

- **坐标轴**：
  - X轴：自动调整范围，显示最近的200个Epoch
  - Y轴：根据损失值自动调整范围

**图表控制**：
- **清空图表**：通过菜单栏的"工具 > 清空图表"清空图表数据
- **自动缩放**：根据数据自动调整坐标轴范围
- **性能优化**：限制显示200个数据点，避免性能问题

#### 2.2.4 训练日志功能区域

**日志显示**：
- **日志窗口**：显示训练过程中的所有日志信息
  - 样式：深色背景，浅色文字
  - 字体：等宽字体（Consolas）
  - 只读：用户不能编辑

- **日志内容**：
  - 系统启动信息
  - 数据加载信息
  - 训练进度信息
  - Epoch完成信息
  - 错误和警告信息
  - 训练完成信息

- **日志操作**：
  - 清空日志：通过菜单栏的"工具 > 清空日志"清空日志
  - 自动滚动：新日志自动滚动到底部

#### 2.2.5 入侵检测面板功能区域

**文件选择页面**：
- **设备选择**：
  - 下拉列表：选择要评估的设备
  - 自动检查：设备选择变化时自动检查默认路径下的DStst文件

- **DStst数据文件选择**：
  - 文本框：显示选择的dstst_data.npy文件路径
  - 浏览按钮：打开文件选择对话框
  - 占位符：选择dstst_data.npy文件

- **DStst标签文件选择**：
  - 文本框：显示选择的dstst_labels.npy文件路径
  - 浏览按钮：打开文件选择对话框
  - 占位符：选择dstst_labels.npy文件

- **文件状态显示**：
  - 状态标签：显示文件验证结果
  - 颜色指示：绿色=有效，红色=无效，橙色=未找到
  - 详细信息：显示样本数量和验证状态

- **模型文件选择**：
  - 文本框：显示选择的模型文件路径
  - 浏览按钮：打开文件选择对话框
  - 文件格式：*.h5, *.hdf5, *.keras

**评估过程页面**：
- **开始评估按钮**：启动评估过程
- **进度条**：显示评估进度（0-100%）
- **状态标签**：显示当前评估状态
- **日志显示区域**：显示评估过程中的日志信息

**结果显示页面**：
- **性能指标显示**：
  - 准确率 (Accuracy)
  - 精确率 (Precision)
  - 召回率 (Recall)
  - F1分数 (F1 Score)
  - 假阳性率 (FPR)

- **混淆矩阵显示**：
  - 真阳性 (TP)
  - 假阳性 (FP)
  - 假阴性 (FN)
  - 真阴性 (TN)

- **ROC曲线显示**：
  - ROC曲线图
  - AUC值

- **详细结果表格**：
  - 样本ID
  - 真实标签
  - 预测标签
  - 异常分数
  - 判断结果

**保存选项页面**：
- **保存评估数据**：保存评估结果到CSV文件
- **保存性能指标**：保存性能指标到JSON文件
- **保存可视化图表**：生成并保存ROC曲线、混淆矩阵等图表
- **保存详细报告**：生成HTML格式的详细报告

### 2.3 操作流程指南

#### 2.3.1 模型训练操作流程

**步骤1：配置训练参数**
1. 打开应用程序
2. 在"模型训练"选项卡中，点击"基础配置"页面
3. 设置数据根目录路径（指向N-BaIoT数据集）
4. 设置输出目录路径（训练结果保存位置）
5. 根据需要启用GPU加速
6. 点击"模型架构"页面，设置模型参数
7. 点击"训练参数"页面，设置训练参数
8. 点击"设备选择"页面，选择要训练的设备
9. 点击"保存选项"页面，配置文件和图表保存选项

**步骤2：启动训练**
1. 确认所有配置参数正确
2. 点击训练控制面板的"开始训练"按钮
3. 观察实时图表和日志输出
4. 等待训练完成

**步骤3：监控训练过程**
1. 观察实时训练损失曲线
2. 查看训练日志输出
3. 监控训练进度和状态
4. 如需暂停，点击"暂停"按钮
5. 如需继续，点击"继续"按钮
6. 如需停止，点击"停止"按钮

**步骤4：查看训练结果**
1. 训练完成后，查看日志窗口的完成信息
2. 检查输出目录中的训练结果
3. 查看生成的可视化图表
4. 分析训练历史和性能指标

#### 2.3.2 入侵检测评估操作流程

**步骤1：准备评估数据**
1. 确保已完成模型训练
2. 系统会自动检查默认路径下是否存在DStst文件
3. 默认路径为：`Config.OUTPUT_DIR / device_name /`

**步骤2：配置评估参数**
1. 切换到"入侵检测与评估"选项卡
2. 点击"文件选择"页面
3. 选择要评估的设备
4. 系统会自动检查DStst文件：
   - 如果找到文件，自动填充文件路径并显示绿色状态
   - 如果没有找到文件，显示橙色提示信息

**步骤3：选择DStst文件**
1. 如果系统没有自动找到DStst文件，可以手动选择：
   - 点击"DStst数据文件"的"浏览..."按钮，选择dstst_data.npy文件
   - 点击"DStst标签文件"的"浏览..."按钮，选择dstst_labels.npy文件
2. 系统会自动验证文件：
   - 检查文件是否存在
   - 验证数据和标签数量是否匹配
   - 显示详细的验证结果和样本数量

**步骤4：选择模型文件**
1. 点击"模型文件选择"的"浏览..."按钮
2. 选择训练好的模型文件（*.h5, *.hdf5, *.keras）
3. 系统会自动验证模型文件

**步骤5：启动评估**
1. 确认所有配置参数正确
2. 点击"开始评估"按钮
3. 如果DStst文件不存在，系统会弹出对话框：
   - 询问："未找到DStst文件。是否自动生成DStst文件？"
   - 点击"是"自动生成DStst文件
   - 点击"否"取消评估
4. 观察评估进度和日志输出
5. 等待评估完成

**步骤6：查看评估结果**
1. 评估完成后，切换到"评估结果"页面
2. 查看性能指标（准确率、精确率、召回率等）
3. 查看混淆矩阵和ROC曲线
4. 查看详细结果表格

**步骤7：保存评估结果**
1. 切换到"保存选项"页面
2. 勾选需要保存的内容
3. 点击保存按钮
4. 查看保存的文件和图表

#### 2.3.3 配置保存和加载流程

**保存配置**：
1. 配置好所有参数后
2. 点击菜单栏的"文件 > 保存配置"
3. 选择保存位置和文件名
4. 点击保存按钮
5. 确认保存成功

**加载配置**：
1. 点击菜单栏的"文件 > 加载配置"
2. 选择之前保存的配置文件
3. 点击打开按钮
4. 确认配置加载成功
5. 检查所有参数是否正确

### 2.4 交互逻辑与注意事项

#### 2.4.1 交互逻辑

**训练过程中的交互**：
- **开始训练**：
  - 点击"开始训练"按钮后，按钮变为禁用状态
  - "暂停"和"停止"按钮变为启用状态
  - 实时图表开始显示训练曲线
  - 日志窗口开始输出训练信息
  - 状态栏显示训练进度

- **暂停训练**：
  - 点击"暂停"按钮后，按钮文本变为"继续"
  - 训练工作线程暂停执行
  - 状态栏显示"已暂停 - 等待恢复..."
  - 可以点击"继续"按钮恢复训练

- **继续训练**：
  - 点击"继续"按钮后，按钮文本变回"暂停"
  - 训练工作线程继续执行
  - 状态栏显示"正在恢复训练..."

- **停止训练**：
  - 点击"停止"按钮后，训练工作线程停止
  - 所有按钮恢复到初始状态
  - 日志窗口显示停止信息
  - 状态栏显示"训练已停止"

**评估过程中的交互**：
- **开始评估**：
  - 点击"开始评估"按钮后，按钮变为禁用状态
  - 进度条开始显示评估进度
  - 日志窗口开始输出评估信息
  - 状态标签显示评估状态

- **评估完成**：
  - 评估完成后，"开始评估"按钮恢复启用状态
  - 进度条显示100%
  - 结果显示页面更新评估结果
  - 可以查看和保存评估结果

#### 2.4.2 使用注意事项

**数据准备注意事项**：
1. **数据集路径**：确保DATA_ROOT路径正确指向N-BaIoT数据集
2. **数据格式**：确保数据文件格式为CSV，包含115个特征
3. **数据完整性**：确保benign_traffic.csv文件存在且完整
4. **数据划分**：根据需要选择按时间顺序或随机划分

**GPU使用注意事项**：
1. **GPU驱动**：确保安装了正确的NVIDIA GPU驱动
2. **CUDA和cuDNN**：确保安装了兼容的CUDA和cuDNN版本
3. **TensorFlow版本**：确保使用TensorFlow GPU版本
4. **内存限制**：显存不足时设置GPU_MEMORY_LIMIT
5. **CPU回退**：GPU不可用时系统会自动使用CPU

**训练参数注意事项**：
1. **学习率**：过大导致训练不稳定，过小导致收敛慢
2. **批大小**：过大导致显存不足，过小导致训练慢
3. **训练轮数**：过多导致过拟合，过少导致欠拟合
4. **早停耐心值**：过小导致过早停止，过大导致浪费时间
5. **正则化**：过拟合时增加正则化强度

**模型保存注意事项**：
1. **保存模型**：勾选"保存模型文件"选项
2. **保存标准化器**：勾选"保存数据标准化器"选项（推理时必需）
3. **保存训练历史**：勾选"保存训练历史"选项（用于后续分析）
4. **保存图表**：勾选"保存可视化图表"选项（用于结果展示）

**性能优化注意事项**：
1. **减少图表**：训练速度慢时减少生成的图表类型
2. **禁用TensorBoard**：不需要TensorBoard时禁用以节省资源
3. **减小批大小**：显存不足时减小批大小
4. **限制设备数量**：同时训练多个设备时注意资源限制

**错误处理注意事项**：
1. **数据加载错误**：检查数据路径和文件完整性
2. **训练错误**：查看日志窗口的错误信息，调整参数后重试
3. **GPU错误**：检查GPU配置，必要时使用CPU
4. **内存错误**：减小批大小或启用GPU内存限制

**退出注意事项**：
1. **训练中退出**：训练进行中退出时会提示确认
2. **保存配置**：退出前建议保存当前配置
3. **优雅退出**：等待当前设备训练完成后再退出

#### 2.4.3 常见问题解决

**问题1：无法启动GPU训练**
- **原因**：GPU驱动、CUDA、cuDNN未正确安装
- **解决**：
  1. 检查NVIDIA驱动是否正确安装
  2. 安装CUDA 11.2（TensorFlow 2.10.0兼容版本）
  3. 安装cuDNN 8.1
  4. 确保CUDA和cuDNN路径在系统PATH中
  5. 验证TensorFlow GPU版本：`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

**问题2：训练过程中显存不足**
- **原因**：批大小过大或模型过于复杂
- **解决**：
  1. 减小批大小（如从128减到64）
  2. 设置GPU_MEMORY_LIMIT（如4096MB）
  3. 简化模型架构（减少层数或神经元数量）
  4. 禁用批量归一化

**问题3：训练不收敛**
- **原因**：学习率过大或模型不合适
- **解决**：
  1. 降低学习率（如从0.001减到0.0001）
  2. 增加训练轮数
  3. 调整模型架构（增加层数或神经元数量）
  4. 检查数据标准化是否正确

**问题4：过拟合严重**
- **原因**：模型过于复杂或训练数据不足
- **解决**：
  1. 增加Dropout率（如从0.0增加到0.3）
  2. 增加L2正则化（如从0.001增加到0.01）
  3. 减少模型层数或神经元数量
  4. 启用早停机制

**问题5：评估结果不准确**
- **原因**：模型未充分训练或阈值设置不当
- **解决**：
  1. 增加训练轮数
  2. 调整异常阈值（如从μ+2σ调整到μ+3σ）
  3. 使用更多训练数据
  4. 优化模型架构

---

## 附录

### A. 技术栈信息

**深度学习框架**：
- TensorFlow 2.10.0
- Keras 2.10.0

**GUI框架**：
- PyQt5 5.15.11

**数据处理**：
- NumPy 1.21.6
- pandas 1.5.3
- scikit-learn 1.1.3

**可视化**：
- Matplotlib 3.5.3
- Seaborn 0.12.2

**辅助工具**：
- SciPy 1.9.3
- joblib 1.2.0

### B. 系统要求

**硬件要求**：
- CPU：双核及以上
- 内存：8GB及以上
- GPU（可选）：NVIDIA GPU，支持CUDA 11.2
- 硬盘：10GB及以上可用空间

**软件要求**：
- 操作系统：Windows 10/11，Linux，macOS
- Python：3.7及以上
- CUDA：11.2（GPU训练时）
- cuDNN：8.1（GPU训练时）

### C. 依赖包列表

```
tensorflow>=2.10.0
PyQt5>=5.15.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.12.0
scipy>=1.9.0
joblib>=1.2.0
```

### D. 文件输出结构

```
training_results/
├── [设备名称]/
│   ├── training_plots/              # 训练图表
│   │   ├── [设备]_comprehensive_summary_[时间戳].png
│   │   ├── [设备]_training_loss_curve_[时间戳].png
│   │   ├── [设备]_hyperparameter_heatmap_[时间戳].png
│   │   └── ...
│   ├── metrics/                   # 性能指标图表
│   │   ├── [设备]_performance_metrics_[时间戳].png
│   │   ├── [设备]_loss_distribution_[时间戳].png
│   │   └── ...
│   ├── data/                      # 数据分析图表
│   │   ├── [设备]_data_distribution_[时间戳].png
│   │   └── ...
│   ├── final_model.h5             # 最终模型
│   ├── scaler.pkl                 # 标准化器
│   ├── hyperparameter_tuning.json  # 超参数调优结果
│   └── training_report.txt        # 训练报告
└── comparison_plots/              # 比较图表
    └── detailed_comparison_[时间戳].png
```

### E. 参考文献

1. Meidan, Y., et al. (2018). "N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders". *IEEE Pervasive Computing and Communications Workshops*.

2. Mirsky, Y., et al. (2018). "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection". *NDSS*.

3. Vincent, P., et al. (2008). "Extracting and Composing Robust Features with Denoising Autoencoders". *ICML*.

4. Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization". *ICLR*.

---

## 附录

### E. DStst文件配置示例

**DStst文件格式**：
- **数据文件**：`dstst_data.npy` - 包含测试数据（NumPy数组）
- **标签文件**：`dstst_labels.npy` - 包含测试标签（NumPy数组）

**默认路径结构**：
```
training_results/
├── [设备名称]/
│   ├── dstst_data.npy          # DStst数据文件
│   ├── dstst_labels.npy        # DStst标签文件
│   ├── final_model.h5          # 训练好的模型
│   ├── scaler.pkl              # 数据标准化器
│   └── training_plots/         # 训练图表
└── ...
```

**自动生成DStst文件**：
```python
from data_integrator import DStstIntegrator
from config import Config

# 创建DStst整合器
integrator = DStstIntegrator(Config)

# 为指定设备生成DStst文件
device_name = "Danmini_Doorbell"
dstst_data, dstst_labels = integrator.create_dstst(device_name)

# 保存DStst文件
save_dir = os.path.join(Config.OUTPUT_DIR, device_name)
data_path, labels_path = integrator.save_dstst(
    device_name, dstst_data, dstst_labels, save_dir
)

print(f"DStst文件已生成：")
print(f"  数据文件: {data_path}")
print(f"  标签文件: {labels_path}")
print(f"  数据样本数: {len(dstst_data)}")
print(f"  标签样本数: {len(dstst_labels)}")
```

**手动选择DStst文件**：
```python
import numpy as np

# 加载DStst数据文件
dstst_data_path = "training_results/Danmini_Doorbell/dstst_data.npy"
dstst_data = np.load(dstst_data_path)

# 加载DStst标签文件
dstst_labels_path = "training_results/Danmini_Doorbell/dstst_labels.npy"
dstst_labels = np.load(dstst_labels_path)

# 验证数据和标签数量是否匹配
if len(dstst_data) != len(dstst_labels):
    print(f"错误：数据和标签数量不匹配！")
    print(f"  数据样本数: {len(dstst_data)}")
    print(f"  标签样本数: {len(dstst_labels)}")
else:
    print(f"验证成功：数据和标签数量一致 ({len(dstst_data)} 个样本）")
```

### F. 入侵检测配置示例

**完整的入侵检测配置**：
```python
config = {
    # 设备配置
    'device_name': 'Danmini_Doorbell',

    # DStst文件配置
    'dstst_data_file': 'training_results/Danmini_Doorbell/dstst_data.npy',
    'dstst_labels_file': 'training_results/Danmini_Doorbell/dstst_labels.npy',

    # 模型文件配置
    'model_file': 'training_results/Danmini_Doorbell/final_model.h5',

    # 保存配置
    'save_path': 'training_results/intrusion_detection',
    'save_data': True,
    'save_images': True
}
```

**使用GUI配置入侵检测**：
1. 打开应用程序
2. 切换到"入侵检测与评估"选项卡
3. 在"文件选择"页面中：
   - 选择要评估的设备
   - 系统会自动检查默认路径下的DStst文件
   - 如果没有找到文件，可以手动选择：
     * DStst数据文件：dstst_data.npy
     * DStst标签文件：dstst_labels.npy
   - 选择训练好的模型文件
4. 点击"开始评估"按钮
5. 如果DStst文件不存在，系统会询问是否自动生成
6. 等待评估完成
7. 查看评估结果和性能指标

### G. 常见问题解决

**问题1：DStst文件选择后提示"文件格式不正确"**
- **原因**：原实现只支持单个文件，但实际需要两个文件
- **解决方案**：
  - 确保分别选择了dstst_data.npy和dstst_labels.npy
  - 检查文件是否存在
  - 验证数据和标签数量是否匹配

**问题2：找不到DStst文件**
- **原因**：DStst文件尚未生成
- **解决方案**：
  - 方法1：让系统自动生成
    - 点击"开始评估"按钮
    - 在弹出的对话框中点击"是"
    - 系统会自动调用data_integrator生成DStst文件
  - 方法2：手动生成
    - 使用data_integrator模块生成DStst文件
    - 参考附录E中的代码示例

**问题3：数据和标签数量不匹配**
- **原因**：dstst_data.npy和dstst_labels.npy的样本数量不一致
- **解决方案**：
  - 重新生成DStst文件
  - 检查data_integrator模块的实现
  - 确保良性数据和攻击数据的划分正确

**问题4：评估时提示"DStst数据文件不存在"**
- **原因**：文件路径错误或文件不存在
- **解决方案**：
  - 检查文件路径是否正确
  - 确保文件存在于指定路径
  - 使用"浏览..."按钮重新选择文件

**问题5：评估过程数据异常（误报率100%、准确率不为0、召回率100%）**
- **原因**：GUI的intrusion_detection_worker.py实现与evaluate_anomaly_detection.py脚本的实现不一致
- **解决方案**：
  - 已修复：更新intrusion_detection_worker.py，使其与evaluate_anomaly_detection.py脚本的实现一致
  - 使用AnomalyDetector.evaluate_performance()方法进行评估
  - 正确计算异常阈值和滑动窗口大小
  - 应用滑动窗口多数投票机制
  - 确保评估指标计算正确

**问题6：评估结果数据不合理（准确率5.71%、精确率5.71%、召回率100%、误报率100%、F1分数10.8%）**
- **原因**：评估逻辑实现错误，导致性能指标计算不正确
- **解决方案**：
  - 已修复：更新评估逻辑，使用正确的混淆矩阵计算
  - 确保准确率、精确率、召回率、F1分数、误报率计算正确
  - 使用分批处理实现实时数据更新
  - 确保评估结果与实际数据一致

**问题7：评估过程图表缺乏实际参考价值**
- **原因**：原RealTimeChart组件仅用于训练过程，不适用于评估过程
- **解决方案**：
  - 已修复：创建新的EvaluationChart组件
  - 实现评估过程的动态性能指标变化曲线
  - 显示时间序列数据点和关键指标变化
  - 提供评估过程的可追溯性与直观性
  - 包含准确率、精确率、召回率、F1分数、误报率等指标

**问题8：保存选项面板功能缺陷**
- **原因**：保存逻辑不完整，仅生成JSON文件
- **解决方案**：
  - 已修复：更新保存选项面板功能
  - 确保勾选"保存评估图表"与"保存评估数据"选项后，系统能正确生成并保存所有相关文件
  - 生成性能指标图（条形图）
  - 生成混淆矩阵图（热图）
  - 生成ROC曲线图
  - 保存评估结果到JSON文件
  - 确保文件生成逻辑、路径配置及权限设置正确

---

**文档结束**

**版本历史**：
- v1.0 (2026-01-29): 初始版本，命令行界面
- v2.0 (2026-01-29): 添加GUI界面
- v3.0 (2026-02-05): 完整模块化重构，修复模型保存bug
- v3.1 (2026-02-05): 修复入侵检测面板的DStst文件选择功能bug
  - 支持分别选择dstst_data.npy和dstst_labels.npy两个文件
  - 添加默认路径检查和自动生成DStst文件功能
  - 更新相关模块的实现细节和参数配置
  - 修正文档中与实际修复结果不一致的描述
  - 补充必要的代码示例和配置说明
- v3.2 (2026-02-05): 入侵检测与评估面板优化
  - 修复评估逻辑，使其与evaluate_anomaly_detection.py脚本的实现一致
  - 创建EvaluationChart组件，实现评估过程的动态性能指标变化曲线
  - 修复保存选项面板功能，确保能正确生成并保存所有相关文件
  - 优化评估过程子面板的动态图表功能，实现时间序列数据点和关键指标变化曲线
  - 修正评估结果数据合理性问题
  - 更新相关模块的实现细节、参数配置及算法逻辑

**联系方式**：
如有问题或建议，请通过项目仓库提交Issue。

**许可证**：
本系统仅供教育研究用途，可自由修改和使用。商业使用请联系作者。

**致谢**：
感谢N-BaIoT数据集的作者，以及TensorFlow、PyQt5等开源项目的贡献者。
