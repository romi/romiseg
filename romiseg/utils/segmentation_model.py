#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:43:07 2019

@author: alienor
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import romiseg.utils.vox_to_coord as vtc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    
    
def voxel_to_pred_by_project(the_shape, torch_voxels, intrinsics, extrinsics, preds_flat, pred_pad, Sx, Sy, xinit, yinit):
        xy_coords = vtc.project_coordinates(torch_voxels, intrinsics, extrinsics, give_prod = False)
        #permute x and y coordinates
        xy_coords[:, 2, :] = xy_coords[:,0,:]
        xy_coords[:, 0, :] = xy_coords[:,1,:]
        xy_coords[:, 1, :] = xy_coords[:,2,:]
        
        coords = vtc.correct_coords_outside(xy_coords, Sx, Sy, xinit, yinit, -1) #correct the coordinates that project outside
        xy_full_flat = vtc.flatten_coordinates(coords, the_shape)
        assign_preds = preds_flat[xy_full_flat].reshape(pred_pad.shape[0], 
                                                xy_full_flat.shape[0]//pred_pad.shape[0], preds_flat.shape[-1])
        del xy_full_flat
        
        assign_preds = torch.sum(assign_preds, dim = 0)
        assign_preds[:,0] *= 0.8
        torch_voxels[:,3] = torch.argmax(assign_preds, dim = 1)
        return torch_voxels

class ResNetUNet_3D(nn.Module):

    def __init__(self, n_class, coord_file_loc):
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
        
        lin = torch.nn.Linear(n_class+1, n_class)
        lin.weight.data.fill_(0)
        lin.weight.data.fill_diagonal_(1)
        lin.bias.data.fill_(0)
        self.class_layer = nn.Sequential(lin, 
                                         nn.ReLU(inplace=True))

            
        self.coord_file_loc = coord_file_loc
        #self.xinit = xinit
        #self.yinit = yinit
        #self.Sx = Sx
        #self.Sy = Sy


    def forward(self, input):
        x_original = self.conv_original_size0(input)
        N_frames = x_original.shape[0]

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
        x = self.conv_last(x)
        
       
        
        xy_full_flat = torch.load(self.coord_file_loc + '/coords.pt').to(device)
                                
        #haut = torch.empty(N_red, label_num, (xinit-Sx)//2, yinit, requires_grad = True).to(device)   
        #bas = torch.empty(N_red, label_num, (xinit-Sx)//2, yinit, requires_grad = True).to(device)   
        #gauche = torch.empty(N_red, label_num, Sx, (yinit - Sy)//2, requires_grad = True).to(device)   
        #droite = torch.empty(N_red, label_num, Sx, (yinit - Sy)//2, requires_grad = True).to(device)   

        #pred_pad = torch.cat((gauche, x, droite), dim = 3)
        #pred_pad = torch.cat((haut, pred_pad, bas), dim = 2)
        #pred_pad = Variable(pred_pad, requires_grad = True)
        #pred_pad[:,:,(xinit-Sx)//2:(xinit+Sx)//2,
        # (yinit-Sy)//2:(yinit+Sy)//2] = x #To fit the camera parameters

        #pred_pad = pred_pad.permute(0,2,3,1)
        #print(preds.shape)
        pred_pad = F.sigmoid(torch.flip(x, dims = [0])).permute(0, 2, 3, 1)
        pred_pad = vtc.adjust_predictions(pred_pad)
        #print(preds.shape)
        pred_pad = pred_pad[xy_full_flat].reshape(N_frames, 
                               xy_full_flat.shape[0]//N_frames, pred_pad.shape[-1])
        #print(preds.shape)
        #preds[:,:,6] = 0
        #print(preds.shape)
        
        pred_pad = self.class_layer(pred_pad)
        #pred_pad = pred_pad.clamp(min=1e-8)
        #pred_pad = torch.log(pred_pad)
        #pred_pad  = torch.sum(pred_pad, dim = 0)
        pred_pad = torch.prod(pred_pad, dim = 0)
        
        #print(preds.shape)
        #print(torch.max(preds, dim = 0))
        #print(preds.shape)
        del xy_full_flat
        return [x, pred_pad]