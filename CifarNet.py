# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:38:03 2020

@author: RML
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def _cifarnet(pretrained=False, path=None):
    model = CifarNet()
    if pretrained:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    return model
