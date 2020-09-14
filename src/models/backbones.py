import tensorflow as tf
from .modules import Conv, ConvDW, ShuffleV2Block, make_bottleneck_layers


class MobileNetV2(tf.keras.layers.Layer):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.backbone = tf.keras.Sequential([
            Conv(32, stride=2, bias=False, name='mbn2_conv1'),
            ConvDW(64, bn=True, relu='relu', name='mbn2_convdw1'),
            ConvDW(128, stride=2, bn=True, relu='relu', name='mbn2_convdw2'),
            ConvDW(128, bn=True, relu='relu', name='mbn2_convdw3'),
            ConvDW(256, stride=2, bn=True, relu='relu', name='mbn2_convdw4'),
            ConvDW(256, bn=True, relu='relu', name='mbn2_convdw5'),
            ConvDW(512, bn=True, relu='relu', name='mbn2_convdw6'),
            ConvDW(512, bn=True, relu='relu', dilation=2, name='mbn2_convdw7'),
            ConvDW(512, bn=True, relu='relu', name='mbn2_convdw8'),
            ConvDW(512, bn=True, relu='relu', name='mbn2_convdw9'),
            ConvDW(512, bn=True, relu='relu', name='mbn2_convdw10'),
            ConvDW(512, bn=True, relu='relu', name='mbn2_convdw11'),
        ])

    def call(self, inputs, training):
        return self.backbone(inputs, training)


class ShuffleNetV2(tf.keras.layers.Layer):
    def __init__(self):
        super(ShuffleNetV2, self).__init__()
        stage_out_channels = [-1, 48, 232, 464]
        stage_repeats = [4, 8]
        input_channel = stage_out_channels[1]
        self.first_conv = tf.keras.layers.Conv2D(filters=input_channel,
                                                 kernel_size=3, strides=2,
                                                 padding='same', use_bias=False,
                                                 name='shf2_conv1')
        self.first_bn = tf.keras.layers.BatchNormalization(name='shf2_bn1')

        self.features = tf.keras.Sequential()
        for idxstage in range(len(stage_repeats)):
            num_repeat = stage_repeats[idxstage]
            output_channel = stage_out_channels[idxstage + 2]
            for i in range(num_repeat):
                if i == 0:
                    self.features.add(ShuffleV2Block(inp_channels=input_channel,
                                                     out_channels=output_channel,
                                                     mid_channels=output_channel // 2,
                                                     ksize=3, stride=2,
                                                     name=f'stage_{idxstage}_block_{i}'))
                else:
                    self.features.add(ShuffleV2Block(inp_channels=input_channel // 2,
                                                     out_channels=output_channel,
                                                     mid_channels=output_channel // 2,
                                                     ksize=3, stride=1,
                                                     name=f'stage_{idxstage}_block_{i}'))
                input_channel = output_channel

    def call(self, inputs, training):
        x = self.first_conv(inputs)
        x = self.first_bn(x, training=training)
        x = self.features(x, training=training)
        return x


class ResNet50(tf.keras.layers.Layer):
    def __init__(self, mobile, layer_params=(2, 4, 3)):
        super(ResNet50, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7,
                                            strides=2, padding='same', name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2, padding='same')
        self.block1 = make_bottleneck_layers(num_filters=64, num_blocks=layer_params[0],
                                             mobile=mobile, idx=1)
        self.block2 = make_bottleneck_layers(num_filters=128, num_blocks=layer_params[1],
                                             mobile=mobile, stride=2, idx=2)
        self.block3 = make_bottleneck_layers(num_filters=256, num_blocks=layer_params[2],
                                             mobile=mobile, stride=2, idx=3)

    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        return x
