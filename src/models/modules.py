import tensorflow as tf
import tensorflow.keras.backend as K


class Conv(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3,
                 bn=True, dilation=1, stride=1, relu=True,
                 bias=True, mobile=False, name=''
                 ):
        super(Conv, self).__init__()
        if mobile:
            groups = 4
        else:
            groups = 1
        self.conv2d = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=kernel_size, strides=stride,
            padding='same', groups=groups, dilation_rate=dilation, use_bias=bias,
            name=name+'_conv2d',
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
    def __init__(self, out_channels, mobile=False, name=''):
        super(RefinementBlock, self).__init__()
        self.initial_conv = Conv(out_channels=out_channels, kernel_size=1,
                                 mobile=mobile, name=name+'_initial')
        self.conv_truck = tf.keras.Sequential([
            Conv(out_channels=out_channels, mobile=mobile, name=name + '_truck1'),
            Conv(out_channels=out_channels, mobile=mobile, name=name + '_truck2')
        ])
        self.add = tf.keras.layers.Add(name=name+'_add')

    def call(self, inputs, training):
        net_initial = self.initial_conv(inputs, training=training)
        net = self.conv_truck(net_initial, training=training)
        net = self.add([net_initial, net])
        return net


class ShuffleV2Block(tf.keras.layers.Layer):
    def __init__(self, inp_channels, out_channels, mid_channels, *, ksize, stride, name):
        super(ShuffleV2Block, self).__init__()
        assert stride in [1, 2]

        self.main_branch = ShuffleV2MainBranch(mid_channels=mid_channels, out_channels=inp_channels,
                                               ksize=ksize, stride=stride, name=name + '_main')
        if stride == 2:
            self.proj_branch = ShuffleNetV2ProjBranch(out_channels=out_channels - inp_channels, ksize=ksize,
                                                      stride=stride, name=name + '_proj')

    def call(self, inputs, training):
        if not hasattr(self, 'proj_branch'):
            x_proj, x = tf.split(inputs, num_or_size_splits=2, axis=3)
            return tf.concat([x_proj, self.main_branch(x, training=training)], axis=3)
        else:
            return tf.concat([self.proj_branch(inputs, training=training),
                              self.main_branch(inputs, training=training)], axis=3)


class ShuffleV2MainBranch(tf.keras.layers.Layer):
    def __init__(self, mid_channels, out_channels, ksize, stride, name):
        super(ShuffleV2MainBranch, self).__init__()
        self.pw_conv = tf.keras.layers.Conv2D(filters=mid_channels, kernel_size=1,
                                              use_bias=False, name=name + '_pw_conv')
        self.pw_bn = tf.keras.layers.BatchNormalization(name=name + '_pw_bn')

        self.dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=ksize, strides=stride, padding='same',
                                                       use_bias=False, name=name + '_dw_conv')

        self.dw_bn = tf.keras.layers.BatchNormalization(name=name + '_dw_bn')

        self.pwln_conv = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,
                                                use_bias=False, name=name + '_pwln_conv')
        self.pwln_bn = tf.keras.layers.BatchNormalization(name=name + '_pwln_bn')

    def call(self, inputs, training):
        x = self.pw_conv(inputs)
        x = self.pw_bn(x, training=training)
        x = K.relu(x)

        x = self.dw_conv(x)
        x = self.dw_bn(x, training=training)

        x = self.pwln_conv(x)
        x = self.pwln_bn(x, training=training)
        x = K.relu(x)
        return x


class ShuffleNetV2ProjBranch(tf.keras.layers.Layer):
    def __init__(self, ksize, stride, out_channels, name):
        super(ShuffleNetV2ProjBranch, self).__init__()
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=ksize, strides=stride, padding='same',
                                                       use_bias=False, name=name + '_dw_conv')
        self.dw_bn = tf.keras.layers.BatchNormalization(name=name + '_dw_bn')

        self.pwln_conv = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,
                                                use_bias=False, name=name + '_pwln_linear')
        self.pwln_bn = tf.keras.layers.BatchNormalization(name=name + '_pwln_bn')

    def call(self, inputs, training):
        x = self.dw_conv(inputs)
        x = self.dw_bn(x, training=training)

        x = self.pwln_conv(x)
        x = self.pwln_bn(x, training=training)
        x = K.relu(x)
        return x


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, stride=1, name='BasicBlock', idx=0):
        super(BasicBlock, self).__init__()
        name = name + f'_block{idx}'
        self.conv1 = tf.keras.layers.Conv2D(filters=num_filters,
                                            kernel_size=3,
                                            strides=stride, padding='same',
                                            name=name+'_conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name=name+'_bn1')

        self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3,
                                            strides=1, padding='same',
                                            name=name+'_conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(name=name+'_bn2')
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=1,
                                                       strides=stride, name=name+'_conv_down'))
            self.downsample.add(tf.keras.layers.BatchNormalization(name=name+'_bn_down'))

    def call(self, inputs, training):
        if hasattr(self, 'downsample'):
            residual = self.downsample(inputs)
        else:
            residual = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.keras.layers.add([residual, x])
        return tf.nn.relu(output)


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, num_filters, stride=1, name='BottleNeck'):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=1,
                                            strides=1, padding='same',
                                            name=name+'_conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name=name+'_bn1')
        self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3,
                                            strides=stride, padding='same',
                                            name=name+'_conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(name=name+'_bn2')

        self.conv3 = tf.keras.layers.Conv2D(filters=num_filters * 4, kernel_size=1,
                                            strides=1, padding='same',
                                            name=name+'_conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(name=name+'_bn3')

        self.downsample = tf.keras.Sequential()

        self.downsample.add(tf.keras.layers.Conv2D(filters=num_filters*4, kernel_size=1,
                                                   strides=stride, name=name+'_conv_down'))
        self.downsample.add(tf.keras.layers.BatchNormalization(name=name+'_bn_down'))

    def call(self, inputs, training):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        output = tf.keras.layers.add([residual, x])
        return tf.nn.relu(output)


def make_bottleneck_layers(num_filters, num_blocks, stride=1, idx=0):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(num_filters, stride, name=f'BottleNeck_{idx}_block_{0}'))
    for i in range(1, num_blocks):
        res_block.add(BottleNeck(num_filters, 1, name=f'BottleNeck_{idx}_block_{i}'))
    return res_block





