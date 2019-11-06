#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:01:10 2019

@author: alienor
"""

import torch
import torch.nn as nn
from torchvision import models
from collections import defaultdict
import torch.nn.functional as F
from romiseg.utils.loss import dice_loss
from torchvision import transforms, datasets, models
import numpy as np



def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        # Use ResNet18 as the encoder with the pretrained weights
        self.base_model = models.resnet101(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 512, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 512, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 1024, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up2 = convrelu(512 + 512, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        #print(x_original.shape)
        x_original = self.conv_original_size1(x_original)
        #print(x_original.shape)
        layer0 = self.layer0(input)
        #print(layer0.shape)
        layer1 = self.layer1(layer0)
        #print(layer1.shape)
        layer2 = self.layer2(layer1)
        #print(layer2.shape)
        layer3 = self.layer3(layer2)
        #print(layer3.shape)
        layer4 = self.layer4(layer3)
        #print(layer4.shape)
        # Upsample the last/bottom layer
        layer4 = self.layer4_1x1(layer4)
        #print(layer4.shape)
        x = self.upsample(layer4)
        #print(x.shape)
        # Create the shortcut from the encoder
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        #print(x.shape)

        x = self.conv_up3(x)
        #print(x.shape)


        x = self.upsample(x)
        #print(x.shape)

        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        #print(x.shape)

        x = self.conv_up2(x)
        #print(x.shape)
        

        x = self.upsample(x)
        #print(x.shape)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        #print(x.shape)
        x = self.conv_up1(x)
        #print(x.shape)

        x = self.upsample(x)
        #print(x.shape)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        #print(x.shape)
        x = self.conv_up0(x)
        #print(x.shape)

        x = self.upsample(x)
        #print(x.shape)
        x = torch.cat([x, x_original], dim=1)
        #print(x.shape)
        x = self.conv_original_size2(x)
        #print(x.shape)

        out = self.conv_last(x)

        return out
    
def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

class my_model_simple(nn.Module):
    def __init__(self, n_views = 72, n_class = 7):
        super(my_model_simple,self).__init__()
        self.weight = Parameter(torch.Tensor(1, 1, n_views, n_class).to(device))
        self.bias = Parameter(torch.Tensor(1,5*k,n_class).to(device))
        self.reset_parameters()

        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self,x):
        print(x.shape, self.weight.shape)
        x = x * self.weight
        print(x.shape)
        x = torch.sum(x, dim = -2, keepdim = True)
        print(x.shape)
        #x = x + self.bias
        return x[:,:,0,:]
    
# Prediction
def evaluate(inputs, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    with torch.no_grad():
        inputs.requires_grad = False
        # Get the first batch
        inputs = inputs.to(device)

        pred = model(inputs)
        # The loss functions include the sigmoid function.
        pred = F.sigmoid(pred)
        
    return pred


