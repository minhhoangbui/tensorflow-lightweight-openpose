import tensorflow as tf


class Conv(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3,
                 bn=True, dilation=1, stride=1, relu=True,
                 bias=True, name=''
                 ):
        super(Conv, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=kernel_size, strides=stride, padding='same',
            dilation_rate=dilation, use_bias=bias, name=name+'_conv2d'
        )
        if bn:
            self.bn = tf.keras.layers.BatchNormalization(name=name+'_bn')
        if relu:
            self.relu = tf.keras.layers.ReLU(name=name+'_relu')

    def call(self, inputs, training):
        net = self.conv2d(inputs)
        if hasattr(self, 'bn'):
            net = self.bn(net, training=training)
        if hasattr(self, 'relu'):
            net = self.relu(net)
        return net


class ConvDW(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3,
                 bn=True, dilation=1, stride=1, relu='relu',
                 bias=True, name=''):
        super(ConvDW, self).__init__()
        assert relu in ['relu', 'elu'], \
            'Error! Activation must be None or in [\'relu\', \'elu\'], but got {} actually.'.format(relu)
        self.s_conv = tf.keras.layers.SeparableConv2D(filters=out_channels, kernel_size=kernel_size, strides=stride,
                                                      padding='same', dilation_rate=dilation, use_bias=bias,
                                                      name=name+'_sep_conv2d')
        if bn:
            self.bn = tf.keras.layers.BatchNormalization(name=name+'_bn')
        if relu == 'relu':
            self.relu = tf.keras.layers.ReLU(name=name+'_relu')
        elif relu == 'elu':
            self.relu = tf.keras.layers.ELU(name=name+'_elu')

    def call(self, inputs, training):
        net = self.s_conv(inputs)
        if hasattr(self, 'bn'):
            net = self.bn(net, training=training)
        net = self.relu(net)
        return net


class RefinementBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, name=''):
        super(RefinementBlock, self).__init__()
        self.initial_conv = Conv(out_channels=out_channels, kernel_size=1, name=name+'_initial')
        self.conv_truck = tf.keras.Sequential([
            Conv(out_channels=out_channels, name=name + '_truck1'),
            Conv(out_channels=out_channels, name=name + '_truck2')
        ])
        self.add = tf.keras.layers.Add(name=name+'_add')

    def call(self, inputs, training):
        net_initial = self.initial_conv(inputs, training=training)
        net = self.conv_truck(net_initial, training=training)
        net = self.add([net_initial, net])
        return net
