import tensorflow as tf
from .modules import Conv, ConvDW, ShuffleV2Block


class MobileNetV2(tf.keras.layers.Layer):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.conv = Conv(32, stride=2, bias=False, name='mbn2_conv1')
        self.conv_dw1 = ConvDW(64, bn=True, relu='relu', name='mbn2_convdw1')
        self.conv_dw2 = ConvDW(128, stride=2, bn=True, relu='relu', name='mbn2_convdw2')
        self.conv_dw3 = ConvDW(128, bn=True, relu='relu', name='mbn2_convdw3')
        self.conv_dw4 = ConvDW(256, stride=2, bn=True, relu='relu', name='mbn2_convdw4')
        self.conv_dw5 = ConvDW(256, bn=True, relu='relu', name='mbn2_convdw5')
        self.conv_dw6 = ConvDW(512, bn=True, relu='relu', name='mbn2_convdw6')
        self.conv_dw7 = ConvDW(512, bn=True, relu='relu', dilation=2, name='mbn2_convdw7')
        self.conv_dw8 = ConvDW(512, bn=True, relu='relu', name='mbn2_convdw8')
        self.conv_dw9 = ConvDW(512, bn=True, relu='relu', name='mbn2_convdw9')
        self.conv_dw10 = ConvDW(512, bn=True, relu='relu', name='mbn2_convdw10')
        self.conv_dw11 = ConvDW(512, bn=True, relu='relu', name='mbn2_convdw11')

    def call(self, inputs, training):
        x = self.conv(inputs, training=training)
        x = self.conv_dw1(x, training=training)
        x = self.conv_dw2(x, training=training)
        x = self.conv_dw3(x, training=training)
        x = self.conv_dw4(x, training=training)
        x = self.conv_dw5(x, training=training)
        x = self.conv_dw6(x, training=training)
        x = self.conv_dw7(x, training=training)
        x = self.conv_dw8(x, training=training)
        x = self.conv_dw9(x, training=training)
        x = self.conv_dw10(x, training=training)
        x = self.conv_dw11(x, training=training)
        return x


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

        self.features = []
        for idxstage in range(len(stage_repeats)):
            num_repeat = stage_repeats[idxstage]
            output_channel = stage_out_channels[idxstage + 2]
            for i in range(num_repeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(inp_channels=input_channel,
                                                        out_channels=output_channel,
                                                        mid_channels=output_channel // 2,
                                                        ksize=3, stride=2,
                                                        name=f'stage_{idxstage}_block_{i}'))
                else:
                    self.features.append(ShuffleV2Block(inp_channels=input_channel // 2,
                                                        out_channels=output_channel,
                                                        mid_channels=output_channel // 2,
                                                        ksize=3, stride=1,
                                                        name=f'stage_{idxstage}_block_{i}'))
                input_channel = output_channel

    def call(self, inputs, training):
        x = self.first_conv(inputs)
        x = self.first_bn(x, training=training)
        for module in self.features:
            x = module(x, training=training)
        return x
