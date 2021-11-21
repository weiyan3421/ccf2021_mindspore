import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.common.tensor import mstype
import numpy as np
from scipy.stats import truncnorm


def _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _conv1x1(in_channel, out_channel, stride=1):
    weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    return nn.Conv2dBnAct(in_channel, out_channel, kernel_size=1, stride=stride,
                          padding=0, pad_mode='same', weight_init=weight)


def _conv3x3(in_channel, out_channel, stride=1):
    weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    return nn.Conv2dBnAct(in_channel, out_channel, kernel_size=3, stride=stride,
                          padding=0, pad_mode='same', weight_init=weight)


def _conv5x5(in_channel, out_channel, stride=1):
    weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=5)
    return nn.Conv2dBnAct(in_channel, out_channel, kernel_size=5, stride=stride,
                          padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):
    weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
    return nn.Conv2dBnAct(in_channel, out_channel, kernel_size=7, stride=stride,
                          padding=0, pad_mode='same', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
    weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
    return nn.DenseBnAct(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)



class Res_SE_Block(nn.Cell):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(Res_SE_Block, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion

        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        self.conv2_1 = _conv3x3(channel, channel//2, stride=stride)
        self.conv2_2 = _conv5x5(channel, channel//2, stride=stride)
        self.concat = ops.Concat(axis=1)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn(out_channel)

        self.relu = nn.ReLU()

        # SE
        self.se_global_pool = P.ReduceMean(keep_dims=True)
        self.se_down = _conv1x1(in_channel, channel, 1)
        self.se_up = _conv1x1(channel, out_channel, 1)
        self.se_sig = nn.Sigmoid()

        # down_sample
        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_1 = self.conv2_1(out)
        out_2 = self.conv2_2(out)
        out = self.concat((out_1, out_2))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)
            out = out + identity
        else:
            se = self.se_global_pool(x, (2, 3))
            se = self.se_down(se)
            se = self.se_up(se)
            se = self.se_sig(se)
            out = se * out + identity

        out = self.relu(out)
        return out


def _make_layer(block, layer_num, in_channel, out_channel, stride):
    layers = []

    resnet_block = block(in_channel, out_channel, stride=stride)
    layers.append(resnet_block)
    for _ in range(1, layer_num - 1):
        resnet_block = block(out_channel, out_channel, stride=1)
        layers.append(resnet_block)
    return nn.SequentialCell(layers)


class ResNet(nn.Cell):
    def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes):
        super(ResNet, self).__init__()

        self.layer_begin = nn.SequentialCell([_conv7x7(3, 64, stride=2),
                                              _bn(64),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=2, stride=2)])
        self.layer1 = _make_layer(block,
                                  layer_nums[0],
                                  in_channel=in_channels[0],
                                  out_channel=out_channels[0],
                                  stride=strides[0])
        self.layer2 = _make_layer(block,
                                  layer_nums[1],
                                  in_channel=in_channels[1],
                                  out_channel=out_channels[1],
                                  stride=strides[1])
        self.layer3 = _make_layer(block,
                                  layer_nums[2],
                                  in_channel=in_channels[2],
                                  out_channel=out_channels[2],
                                  stride=strides[2])
        self.layer4 = _make_layer(block,
                                  layer_nums[3],
                                  in_channel=in_channels[3],
                                  out_channel=out_channels[3],
                                  stride=strides[3])
        self.global_pool = P.ReduceMean(keep_dims=False)

        self.linear = nn.SequentialCell([nn.Dropout(keep_prob=0.5),
                                         _fc(2048, num_classes)])

    def construct(self, x):
        c0 = self.layer_begin(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        out = self.global_pool(c4, (2, 3))
        out = self.linear(out)

        return out


def L-SE_resnet50(class_num=2388):
    return ResNet(block=Res_SE_Block,
                  layer_nums=[3, 4, 6, 3],
                  in_channels=[64, 256, 512, 1024],
                  out_channels=[256, 512, 1024, 2048],
                  strides=[1, 2, 2, 2],
                  num_classes=class_num)

