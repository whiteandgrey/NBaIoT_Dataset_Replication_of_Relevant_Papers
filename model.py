"""
模型定义 - 对称编码器-解码器
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


class Autoencoder:
    """对称自编码器类"""

    def __init__(self, config):
        """
        初始化自编码器

        Args:
            config: 配置对象
        """
        self.config = config
        self.model = None
        self.encoder = None
        self.decoder = None

    def build(self, input_dim=None):
        """
        构建自编码器模型

        Args:
            input_dim: 输入维度，如果为None则使用config中的FEATURE_DIM

        Returns:
            构建好的Keras模型
        """
        if input_dim is None:
            input_dim = self.config.FEATURE_DIM

        print(f"🔨 Building Autoencoder with input_dim={input_dim}")

        # 计算编码器各层维度
        encoder_dims = [input_dim] + [int(input_dim * ratio)
                                      for ratio in self.config.ENCODER_RATIOS]

        # 计算解码器各层维度（对称结构）
        decoder_dims = [int(input_dim * ratio)
                        for ratio in self.config.DECODER_RATIOS] + [input_dim]

        # 打印架构信息
        print(f"   Encoder architecture: {' -> '.join(map(str, encoder_dims))}")
        print(f"   Decoder architecture: {' -> '.join(map(str, decoder_dims))}")
        print(f"   Activation: {self.config.ACTIVATION}")
        print(f"   Batch normalization: {self.config.USE_BATCH_NORM}")
        print(f"   Dropout rate: {self.config.DROPOUT_RATE}")

        # 输入层
        inputs = layers.Input(shape=(input_dim,))
        x = inputs

        # ============ 编码器部分 ============
        encoder_layers = []

        for i in range(len(encoder_dims) - 1):
            # 全连接层
            x = layers.Dense(
                encoder_dims[i + 1],
                activation=None,  # 先不加激活函数，用于BatchNorm
                kernel_regularizer=regularizers.l2(self.config.L2_REGULARIZATION),
                name=f'encoder_dense_{i}'
            )(x)

            # Batch Normalization
            if self.config.USE_BATCH_NORM:
                x = layers.BatchNormalization(name=f'encoder_bn_{i}')(x)

            # 激活函数
            if self.config.ACTIVATION == 'relu':
                x = layers.ReLU(name=f'encoder_relu_{i}')(x)
            elif self.config.ACTIVATION == 'leaky_relu':
                x = layers.LeakyReLU(alpha=0.1, name=f'encoder_leaky_relu_{i}')(x)
            elif self.config.ACTIVATION == 'tanh':
                x = layers.Activation('tanh', name=f'encoder_tanh_{i}')(x)
            elif self.config.ACTIVATION == 'sigmoid':
                x = layers.Activation('sigmoid', name=f'encoder_sigmoid_{i}')(x)
            else:
                x = layers.ReLU(name=f'encoder_relu_{i}')(x)

            # Dropout
            if self.config.DROPOUT_RATE > 0:
                x = layers.Dropout(self.config.DROPOUT_RATE,
                                   name=f'encoder_dropout_{i}')(x)

            encoder_layers.append(x)

        # 编码器输出（潜在空间表示）
        latent_representation = encoder_layers[-1]

        # ============ 解码器部分 ============
        x = latent_representation

        for i in range(len(decoder_dims) - 1):
            # 全连接层
            x = layers.Dense(
                decoder_dims[i + 1],
                activation=None,  # 先不加激活函数，用于BatchNorm
                kernel_regularizer=regularizers.l2(self.config.L2_REGULARIZATION),
                name=f'decoder_dense_{i}'
            )(x)

            # 最后一层（输出层）的特殊处理
            if i == len(decoder_dims) - 2:  # 输出层
                if self.config.OUTPUT_ACTIVATION:
                    x = layers.Activation(self.config.OUTPUT_ACTIVATION,
                                          name='output_activation')(x)
            else:
                # Batch Normalization（输出层前一层不使用）
                if self.config.USE_BATCH_NORM:
                    x = layers.BatchNormalization(name=f'decoder_bn_{i}')(x)

                # 激活函数
                if self.config.ACTIVATION == 'relu':
                    x = layers.ReLU(name=f'decoder_relu_{i}')(x)
                elif self.config.ACTIVATION == 'leaky_relu':
                    x = layers.LeakyReLU(alpha=0.1, name=f'decoder_leaky_relu_{i}')(x)
                elif self.config.ACTIVATION == 'tanh':
                    x = layers.Activation('tanh', name=f'decoder_tanh_{i}')(x)
                elif self.config.ACTIVATION == 'sigmoid':
                    x = layers.Activation('sigmoid', name=f'decoder_sigmoid_{i}')(x)
                else:
                    x = layers.ReLU(name=f'decoder_relu_{i}')(x)

                # Dropout（输出层前一层不使用）
                if self.config.DROPOUT_RATE > 0:
                    x = layers.Dropout(self.config.DROPOUT_RATE,
                                       name=f'decoder_dropout_{i}')(x)

        # 解码器输出
        outputs = x

        # ============ 创建模型 ============
        self.model = models.Model(inputs=inputs, outputs=outputs, name='autoencoder')

        # 创建编码器模型
        self.encoder = models.Model(inputs=inputs, outputs=latent_representation,
                                    name='encoder')

        # 创建解码器模型
        latent_input = layers.Input(shape=(encoder_dims[-1],))
        
        # 找到第一个解码器 Dense 层（以 'decoder_dense_' 开头的层）
        decoder_output = None
        first_decoder_layer_idx = None
        
        for idx, layer in enumerate(self.model.layers):
            if layer.name.startswith('decoder_dense_'):
                first_decoder_layer_idx = idx
                break
        
        if first_decoder_layer_idx is not None:
            decoder_output = self.model.layers[first_decoder_layer_idx](latent_input)
            # 处理剩余的解码器层
            for layer in self.model.layers[first_decoder_layer_idx + 1:]:
                decoder_output = layer(decoder_output)
        else:
            raise ValueError("Could not find decoder layers in the model")

        self.decoder = models.Model(inputs=latent_input, outputs=decoder_output,
                                    name='decoder')

        # 编译模型
        self.compile()

        return self.model

    def compile(self, learning_rate=None):
        """
        编译模型

        Args:
            learning_rate: 学习率，如果为None则使用config中的默认值
        """
        if learning_rate is None:
            learning_rate = self.config.DEFAULT_LEARNING_RATE

        if self.config.OPTIMIZER.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=self.config.BETA_1,
                beta_2=self.config.BETA_2,
                epsilon=self.config.EPSILON
            )
        elif self.config.OPTIMIZER.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif self.config.OPTIMIZER.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # 均方误差
            metrics=['mae']  # 平均绝对误差
        )

        print(f"✅ Model compiled with {self.config.OPTIMIZER} optimizer, "
              f"LR={learning_rate:.6f}")

    def summary(self):
        """打印模型摘要"""
        if self.model:
            self.model.summary()
        else:
            print("⚠️ Model not built yet. Call build() first.")

    def get_model(self):
        """获取模型"""
        return self.model

    def get_encoder(self):
        """获取编码器"""
        return self.encoder

    def get_decoder(self):
        """获取解码器"""
        return self.decoder

    def save(self, filepath):
        """保存模型"""
        if self.model:
            self.model.save(filepath)
            print(f"✅ Model saved to: {filepath}")
        else:
            print("⚠️ No model to save")

    def load(self, filepath):
        """加载模型"""
        self.model = tf.keras.models.load_model(filepath)

        # 重建编码器和解码器
        inputs = self.model.input
        # 找到潜在层（编码器的最后一层）
        encoder_output = None
        for layer in self.model.layers:
            if 'encoder' in layer.name and layer.name.endswith('_relu_3'):
                encoder_output = layer.output
                break

        if encoder_output is None:
            # 如果找不到特定名称，使用中间层
            num_layers = len(self.model.layers)
            latent_layer_idx = num_layers // 2 - 1
            encoder_output = self.model.layers[latent_layer_idx].output

        self.encoder = models.Model(inputs=inputs, outputs=encoder_output,
                                    name='encoder')

        print(f"✅ Model loaded from: {filepath}")
        return self.model