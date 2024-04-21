import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DepthConvBlock, MaxPool2dStaticSamePadding

__all__ = ['BiFPN']

class BiFPN(nn.Module):
    def __init__(self, network_channel, num_classes, args):
        super(BiFPN, self).__init__()

        depth = args.depth
        width = args.width
        self.layers = BiFPN_layer(DepthConvBlock, network_channel, depth, width)
        self.net_channels = [x * width for x in network_channel]
        self.fc = nn.Linear(self.net_channels[-1], num_classes)

    def forward(self, feats):
        feats = self.layers(feats)
        out = F.adaptive_avg_pool2d(F.relu(feats[-1]), (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return feats, out

class BiFPN_layer(nn.Module):
    def __init__(self, block, network_channel, depth, width):
        super(BiFPN_layer, self).__init__()

        lat_depth, up_depth, down_depth = depth, depth, depth

        self.lat_conv = nn.ModuleList() # conv of F to L

        self.top_conv = nn.ModuleList()  # conv of L to P and L1 to T1 in level 1
        self.top_weight = nn.ParameterList()

        self.bottom_conv = nn.ModuleList()  # conv of P to T and L4 to T4 in level 1
        self.bottom_weight = nn.ParameterList()

        self.down_sample = nn.ModuleList()  # conv of L4 to P3, P3 to P2, and P2 to T1 in level 1
        self.up_sample = nn.ModuleList()    # conv of T1 to T2, T2 to T3, and T3 to T4 in level 1

        for i, channels in enumerate(network_channel):
            self.lat_conv.append(block(channels, channels * width, 1, 1, 0, lat_depth))

            if i != 0:
                self.bottom_conv.append(block(channels * width, channels * width, 3, 1, 1, down_depth))
                num_input = 3 if i < len(network_channel) - 1 else 2
                self.bottom_weight.append(nn.Parameter(torch.ones(num_input, dtype=torch.float32), requires_grad=True))
                self.down_sample.append(nn.Sequential(
                    MaxPool2dStaticSamePadding(3, 2),
                    block(network_channel[i - 1] * width, channels * width, 1, 1, 0, 1)))

            if i != len(network_channel) - 1:
                self.up_sample.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    block(network_channel[i + 1] * width, channels * width, 1, 1, 0, 1)))

                self.top_conv.append(block(channels * width, channels * width, 3, 1, 1, up_depth))
                self.top_weight.append(nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True))

        self.relu = nn.ReLU()
        self.epsilon = 1e-6

    def forward(self, inputs):

        # from [F1~F4] to [L1~L4] in level 1
        inputs = [self.lat_conv[i](F.relu(inputs[i])) for i in range(0, len(inputs))]

        # top-down
        up_sample = [inputs[-1]]
        out_layer = []
        for i in range(1, len(inputs)):
            w = self.relu(self.top_weight[-i])
            w = w / (torch.sum(w, dim=0) + self.epsilon)
            up_sample.insert(0,
                             self.top_conv[-i](w[0] * F.relu(inputs[-i - 1]) +
                                               w[1] * self.up_sample[-i](F.relu(up_sample[0]))))

        # up_sample = [L1, P2, P3, T1] in level 1

        out_layer.append(up_sample[0])

        # bottom-up
        for i in range(1, len(inputs)):
            w = self.relu(self.bottom_weight[i - 1])
            w = w / (torch.sum(w, dim=0) + self.epsilon)
            if i < len(inputs) - 1:
                out_layer.append(self.bottom_conv[i - 1](w[0] * F.relu(inputs[i])
                                                         + w[1] * F.relu(up_sample[i])
                                                         + w[2] * self.down_sample[i - 1](F.relu(out_layer[-1]))))
            else:
                out_layer.append(
                    self.bottom_conv[i - 1](w[0] * F.relu(inputs[i])
                                            + w[1] * self.down_sample[i - 1](F.relu(out_layer[-1]))))

        # out_layer = [T1, T2, T3, T4] in level 1

        return out_layer

