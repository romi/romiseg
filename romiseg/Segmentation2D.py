#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:17:15 2019

@author: alienor
"""
#computer vision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from PIL import Image
from tqdm import tqdm
import requests
import os
import appdirs

#made in CSL
from romidata import io
from romiseg.utils.models import evaluate



def download_file(url, target_dir):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target_dir + '/' +local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename


    
class Dataset(Dataset): 
    """Data handling for Pytorch Dataloader"""

    def __init__(self, image_paths, transform):  

        self.image_paths = image_paths
        self.transforms = transform

    def __getitem__(self, index):

        image = Image.fromarray(io.read_image(self.image_paths[index]))
        
        t_image = self.transforms(image) #crop the images
        
        t_image = t_image[0:3, :, :] #select RGB channels
        
        return t_image

    def __len__(self):  # return count of sample
        return len(self.image_paths)


def segmentation(Sx, Sy, label_names, images_fileset, scan, model_segmentation_name):
        """Inputs a set of N_cam images of an object from different points of view and segmentes the images in N_label classes, 
        pixel per pixel.
        Outputs a matrix of size [N_cam, N_labels, xinit, yinit].
        Sx and Sy are chosen by the user to center-crop the image and lighten
        the computational cost. The neural network should be trained on RGB images of size Sx,Sy.
        """

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Select GPU
        print(device, ' used for images segmentation')
        
        trans = transforms.Compose([ #Define transform of the image
                transforms.CenterCrop((Sx, Sy)),
                transforms.ToTensor()])
        
        #PyTorch Dataloader
        image_set = Dataset(images_fileset, transform = trans) 
        batch_size = 1       
        loader = DataLoader(image_set, batch_size=batch_size, shuffle=False, num_workers=0)
        
        #Access the previously trained segmenttion network stored in db.romi-project.eu
        
        #Save folder
        weights_folder = appdirs.user_cache_dir()

        #if not already saved, download from database 
        if model_segmentation_name not in os.listdir(weights_folder):
            
            url = 'http://db.romi-project.eu/models/' + model_segmentation_name
            
            download_file(url, weights_folder)
            
        model_segmentation = torch.load(weights_folder + '/' + model_segmentation_name)[0]
        
        try: 
            model_segmentation = model_segmentation.module
        except:
            model_segmentation = model_segmentation
    
        #GET ORIGINAL IMAGE SIZE and number(could be in image metadata instead)    
        s = io.read_image(images_fileset[0]).shape
        xinit = s[0] 
        yinit = s[1]
        N_cam = len(images_fileset)
       
        
        
        with torch.no_grad():
            pred_tot = []
            count = 0
            print('Image segmentation by the CNN')
        
            for inputs in tqdm(loader):
                inputs = inputs.to(device) #input image on GPU
                outputs = evaluate(inputs, model_segmentation)  #output image
                pred_tot.append(outputs)
                count += 1
                
        pred_tot = torch.cat(pred_tot, dim = 0)
        pred_pad = torch.zeros((N_cam, len(label_names), xinit, yinit)) #reverse the crop in order to match the colmap parameters
        pred_pad[:,:,(xinit-Sx)//2:(xinit+Sx)//2,(yinit-Sy)//2:(yinit+Sy)//2] = pred_tot #To fit the camera parameters
        
        return pred_pad
