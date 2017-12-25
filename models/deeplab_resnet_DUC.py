import torch.nn as nn
import math
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return int(i)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation_
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class _DenseUpsamplingConvModule(nn.Module):
    def __init__(self, down_factor, in_dim, num_classes):
        super(_DenseUpsamplingConvModule, self).__init__()
        upsample_dim = (down_factor ** 2) * num_classes
        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(upsample_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, ASPP, NoLabels, DUC=False):
        self.inplanes = 64
        self.DUC = DUC
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__=[1,2,5,9])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__=[5,9,17])
        final_dim = NoLabels
        if DUC:
            final_dim = NoLabels * 8 * 8
        self.layer5 = self._make_pred_layer(Classifier_Module, ASPP, ASPP, final_dim)
        self.pixel_shuffle = nn.PixelShuffle(8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation__= None):
        if dilation__ is None:
            dilation__ = [1]
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ != [1]:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation__[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__[(i) % len(dilation__)]))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels):
        return block(dilation_series, padding_series, NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        if self.DUC:
            x = self.pixel_shuffle(x)

        return x

class MS_Deeplab(nn.Module):
    def __init__(self, block, NoLabels, MultiScale = False, DUC = False):
        super(MS_Deeplab, self).__init__()
        self.NoLabels = NoLabels
        self.MultiSCale = MultiScale
        self.DUC = DUC
        self.Scale = ResNet(block, [3, 4, 23, 3], [6, 12, 18, 24], NoLabels, DUC)  # changed to fix #4
        self.dense_upsample = _DenseUpsamplingConvModule(8, NoLabels, NoLabels)

    '''
    def forward(self, x):
        out = self.Scale(x)
        return F.upsample_bilinear(out, x.size()[2:])

    '''
    def forward(self, x):

        if self.MultiSCale:
            input_size1 = x.size()[2]
            input_size2 = x.size()[3]
            self.interp1 = nn.UpsamplingBilinear2d(size=(int(input_size1 * 0.75) + 1, int(input_size2 * 0.75) + 1))
            self.interp2 = nn.UpsamplingBilinear2d(size=(int(input_size1 * 0.5) + 1, int(input_size2 * 0.5) + 1))
            self.interp3 = nn.UpsamplingBilinear2d(size=(outS(input_size1), outS(input_size2)))
            out = []
            x2 = self.interp1(x)
            x3 = self.interp2(x)
            out.append(self.Scale(x))  # for original scale
            out.append(self.interp3(self.Scale(x2)))  # for 0.75x scale
            out.append(self.Scale(x3))  # for 0.5x scale

            x2Out_interp = out[1]
            x3Out_interp = self.interp3(out[2])
            temp1 = torch.max(out[0], x2Out_interp)
            out.append(torch.max(temp1, x3Out_interp))
            return F.upsample_bilinear(out[3], x.size()[2:])
        elif self.DUC:
            out = self.Scale(x)
            #out = self.dense_upsample(out)
            return out
        else:
            out = self.Scale(x)
            return F.upsample_bilinear(out, x.size()[2:])



def Res_Deeplab_DUC(NoLabels=21):
    model = MS_Deeplab(Bottleneck, NoLabels, DUC=True)
    return model