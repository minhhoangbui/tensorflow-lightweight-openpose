import tensorflow as tf
from .modules import Conv, ConvDW, RefinementBlock


class CPM(tf.keras.layers.Layer):
    def __init__(self, out_channels, mobile):
        super(CPM, self).__init__()
        self.align = Conv(out_channels=out_channels, kernel_size=1,
                          bn=False, name='cpm_align', mobile=mobile)
        self.truck = tf.keras.Sequential([
            ConvDW(out_channels, bn=False, relu='elu', name='cpm_truck1'),
            ConvDW(out_channels, bn=False, relu='elu', name='cpm_truck2'),
            ConvDW(out_channels, bn=False, relu='elu', name='cpm_truck3')
        ])
        self.add = tf.keras.layers.Add(name='cpm_add')
        self.cpm = Conv(out_channels=out_channels, bn=False, name='cpm_cpm', mobile=mobile)

    def call(self, inputs, training):
        align = self.align(inputs, training=training)
        truck = self.truck(align, training=training)
        align_truck = self.add([align, truck])
        return self.cpm(align_truck, training=training)


class InitialStage(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_heatmaps, num_pafs, mobile):
        super(InitialStage, self).__init__()
        self.trunk = tf.keras.Sequential([
            Conv(num_channels, bn=False, name='InitialStage_conv1', mobile=mobile),
            Conv(num_channels, bn=False, name='InitialStage_conv2', mobile=mobile),
            Conv(num_channels, bn=False, name='InitialStage_conv3', mobile=mobile)
        ])
        self.heatmaps = tf.keras.Sequential([
            Conv(512, kernel_size=1, bn=False, name='InitialStage_conv4', mobile=mobile),
            Conv(num_heatmaps, kernel_size=1, bn=False, relu=False, name='InitialStage_heat')
        ])
        self.pafs = tf.keras.Sequential([
            Conv(512, kernel_size=1, bn=False, name='InitialStage_conv5', mobile=mobile),
            Conv(num_pafs, kernel_size=1, bn=False, relu=False, name='InitialStage_paf')
        ])

    def call(self, inputs, training):
        trunks = self.trunk(inputs, training=training)
        heatmaps = self.heatmaps(trunks, training=training)
        pafs = self.pafs(trunks, training=training)
        return [heatmaps, pafs]


class RefinementStage(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_heatmaps, num_pafs, mobile):
        super(RefinementStage, self).__init__()
        self.trunk = tf.keras.Sequential([
            RefinementBlock(num_channels, name='RefinementStage_block1'),
            RefinementBlock(num_channels, mobile=mobile, name='RefinementStage_block2'),
            RefinementBlock(num_channels, mobile=mobile, name='RefinementStage_block3'),
            RefinementBlock(num_channels, mobile=mobile, name='RefinementStage_block4'),
            RefinementBlock(num_channels, mobile=mobile, name='RefinementStage_block5')
        ])
        self.heatmaps = tf.keras.Sequential([
            Conv(512, kernel_size=1, bn=False, name='RefinementStage_conv1', mobile=mobile),
            Conv(num_heatmaps, kernel_size=1, bn=False, relu=False, name='RefinementStage_heat')
        ])
        self.pafs = tf.keras.Sequential([
            Conv(512, kernel_size=1, bn=False, name='RefinementStage_conv', mobile=mobile),
            Conv(num_pafs, kernel_size=1, bn=False, relu=False, name='RefinementStage_paf')
        ])

    def call(self, inputs, training):
        trunks = self.trunk(inputs, training=training)
        heatmaps = self.heatmaps(trunks, training=training)
        pafs = self.pafs(trunks, training=training)
        return [heatmaps, pafs]


class LightWeightOpenPose(tf.keras.Model):
    def __init__(self, num_channels, num_refinement_stages=1, num_joints=19, num_pafs=38, mobile=False):
        super(LightWeightOpenPose, self).__init__()
        self.backbone = tf.keras.Sequential([
            Conv(32, stride=2, bias=False, name='backbone_conv1'),
            ConvDW(64, bn=True, relu='relu', name='backbone_convdw1'),
            ConvDW(128, stride=2, bn=True, relu='relu', name='backbone_convdw2'),
            ConvDW(128, bn=True, relu='relu', name='backbone_convdw3'),
            ConvDW(256, stride=2, bn=True, relu='relu', name='backbone_convdw4'),
            ConvDW(256, bn=True, relu='relu', name='backbone_convdw5'),
            ConvDW(512, bn=True, relu='relu', name='backbone_convdw6'),
            ConvDW(512, bn=True, relu='relu', dilation=2, name='backbone_convdw7'),
            ConvDW(512, bn=True, relu='relu', name='backbone_convdw8'),
            ConvDW(512, bn=True, relu='relu', name='backbone_convdw9'),
            ConvDW(512, bn=True, relu='relu', name='backbone_convdw10'),
            ConvDW(512, bn=True, relu='relu', name='backbone_convdw11'),
        ])
        self.cpm = CPM(num_channels, mobile)
        self.initial_stage = InitialStage(num_channels, num_joints, num_pafs, mobile)
        self.refinement_stages = [
            RefinementStage(num_channels, num_joints, num_pafs, mobile) for _ in range(num_refinement_stages)
        ]
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs, training=False, mask=None):
        backbone_features = self.backbone(inputs, training=training)
        backbone_features = self.cpm(backbone_features, training=training)

        stages_output = [self.initial_stage(backbone_features, training=training)]

        for refinement_stage in self.refinement_stages:
            stages_output.append(
                refinement_stage(self.concat([backbone_features, stages_output[-1][0], stages_output[-1][1]]),
                                 training=training)
            )
        return stages_output


def lw(**kwargs):
    return LightWeightOpenPose(num_channels=kwargs['num_channels'], mobile=kwargs['mobile'],
                               num_refinement_stages=kwargs['num_refinement_stages'],
                               num_joints=kwargs['num_joints'], num_pafs=kwargs['num_pafs'])



