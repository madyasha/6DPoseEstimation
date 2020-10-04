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
        network = SqueezePoseNet(base_model, sum_mode, dropout_rate, bayesian)
    else:
        assert 'Unvalid Model'

    return network


class fire(nn.Module):
  def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

  def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezePoseNet(nn.Module):
  def __init__(self, base_model, sum_mode=False, dropout_rate= 0.0, bayesian=False):
    super(SqueezePoseNet, self).__init__()

    self.bayesian = bayesian
    self.dropout_rate = dropout_rate
    self.sum_mode = sum_mode

    self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2) # 32
    self.bn1 = nn.BatchNorm2d(96)
    self.relu1 = nn.ReLU(inplace=True)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2) # 16
    self.fire2 = fire(96, 16, 64)
    self.fire3 = fire(128, 16, 64)
    self.fire4 = fire(128, 32, 128)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2) # 8
    self.fire5 = fire(256, 32, 128)
    self.fire6 = fire(256, 48, 192)
    self.fire7 = fire(384, 48, 192)
    self.fire8 = fire(384, 64, 256)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2) # 4
    self.fire9 = fire(512, 64, 256)
    self.drop =  nn.Dropout(self.dropout_rate)
    self.conv2 = nn.Conv2d(512, 10, kernel_size=1, stride=1)
    self.bn2 = nn.BatchNorm2d(10)
    self.relu2 = nn.ReLU(inplace=True)
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    #added Regressor
    self.fc_dim_reduce = nn.Linear(1 * 1 * 10, 1024)
    self.relu3 = nn.ReLU(inplace=True)
    self.fc_trans = nn.Linear(1024, 3)
    self.fc_rot = nn.Linear(1024, 4)
    
    
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
          nn.init.kaiming_normal_(m.weight)
          if m.bias is not None:
            nn.init.constant_(m.bias, 0)

            # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            #m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
             


  def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu1(x)
      x = self.maxpool1(x)
      x = self.fire2(x)
      x = self.fire3(x)
      x = self.fire4(x)
      x = self.maxpool2(x)
      x = self.fire5(x)
      x = self.fire6(x)
      x = self.fire7(x)
      x = self.fire8(x)
      x = self.maxpool3(x)
      x = self.fire9(x)
      #x = self.drop(x, self.dropout_rate)
      x = self.conv2(x)
      x = self.bn2(x)
      x = self.relu2(x)
      x = self.avg_pool(x)
      x_linear = x.view(x.size(0), -1)
      x_linear = self.fc_dim_reduce(x_linear)
      x_linear = self.relu3(x_linear)
      trans = self.fc_trans(x_linear)
      rot = self.fc_rot(x_linear)

      return trans, rot

