import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math


def model_parser(model, sum_mode=False, dropout_rate= 0.0, bayesian=False):
    base_model = None

    if model == 'Resnet':
        base_model = models.resnet34(pretrained=True)
        network = ShufflePoseNet(base_model, sum_mode, dropout_rate, bayesian)
    else:
        assert 'Unvalid Model'

    return network


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes// 4
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out


#currently following V2 architecture
class ShufflePoseNet(nn.Module):
    def __init__(self, base_model, sum_mode=False, dropout_rate= 0.0, bayesian=False):
        super(ShufflePoseNet, self).__init__()

        cfg = {
        'out_planes': [100,200,400],
        'num_blocks': [4,8,4],
        'groups': 1}

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        self.sum_mode = sum_mode

        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        #self.linear1 = nn.Linear(out_planes[2], 10)
        self.globalpool = nn.AvgPool2d(int(192/16)) 
        self.linear1 = nn.Linear(1600, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.relu1  = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(10, 500)
        #self.bn3 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc_trans = nn.Linear(500, 3)
        self.fc_rot = nn.Linear(500, 4)

        for module in self.modules():
          if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
              nn.init.kaiming_normal_(module.weight)
              if module.bias is not None:
                nn.init.constant_(module.bias, 0)
          elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.weight.data.fill_(1)


    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = F.avg_pool2d(out, 4)
        out = self.globalpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.bn2(out)
        out = self.relu1(out)
        out = self.linear2(out)
        #out = self.bn3(out)
        out = self.relu2 (out)
        out = F.dropout(out, p=self.dropout_rate)
        trans = self.fc_trans(out)
        rot = self.fc_rot(out)
        return trans, rot


# def ShuffleNetG2():
#     cfg = {
#         'out_planes': [200,400,800],
#         'num_blocks': [4,8,4],
#         'groups': 2
#     }
#     return ShuffleNet(cfg)

# def ShuffleNetG3():
#     cfg = {
#         'out_planes': [240,480,960],
#         'num_blocks': [4,8,4],
#         'groups': 3
#     }
#     return ShuffleNet(cfg)
