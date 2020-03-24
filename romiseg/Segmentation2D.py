#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:17:15 2019

@author: alienor
"""
#computer vision
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from romiseg.utils.train_from_dataset import ResizeCrop, ResizeFit


#made in CSL
from romidata import io
from romiseg.utils.train_from_dataset import evaluate, save_and_load_model, model_from_fileset
import logging
logger = logging.getLogger('romiscan')

class Dataset_im_id(Dataset): 
    """Data handling for Pytorch Dataloader"""

    def __init__(self, image_paths, transform):  

        self.image_paths = image_paths
        self.transforms = transform

    def __getitem__(self, index):

        db_file = self.image_paths[index]
        image = Image.fromarray(io.read_image(db_file)[:,:,:3])
        id_im = db_file.id
        
        t_image = self.transforms(image) #crop the images
        
        t_image = t_image[0:3, :, :] #select RGB channels
        print(t_image.max())
        return t_image, id_im

    def __len__(self):  # return count of sample
        return len(self.image_paths)




def segmentation(Sx, Sy, images_fileset, model_file, resize=False):
        """Inputs a set of N_cam images of an object from different points of view and segmentes the images in N_label classes, 
        pixel per pixel.
        Outputs a matrix of size [N_cam, N_labels, xinit, yinit].
        Sx and Sy are chosen by the user to center-crop the image and lighten
        the computational cost. The neural network should be trained on RGB images of size Sx,Sy.
        """

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Select GPU
        logger.debug(str(device) + ' used for images segmentation')

        trans = transforms.Compose([ResizeCrop((Sx, Sy)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]) #imagenet
        
        #PyTorch Dataloader
        image_set = Dataset_im_id(images_fileset, transform = trans) 
        # image_test, channels = init_set('', path_train)
        batch_size = 1       
        loader = DataLoader(image_set, batch_size=batch_size, shuffle=False, num_workers=0)
        #Access the previously trained segmenttion network stored in db.romi-project.eu
        
        #Save folder
        #directory_weights = appdirs.user_cache_dir()
        
        #directory_weights = '/home/alienor/Documents/database/WEIGHTS'
        #if directory_weights == 'complete here':
        #   directory_weights = filedialog.askdirectory(initialdir="/home/", title='create folder to save fine-tuning weights')
        #   create_folder_if(directory_weights)
        logger.debug('Model name:' + str(model_file.get_metadata('model_id')))
        #model_segmentation = save_and_load_model(directory_weights, model_segmentation_name)
        model_segmentation, label_names = model_from_fileset(model_file)
    
        #GET ORIGINAL IMAGE SIZE and number(could be in image metadata instead)    
        s = io.read_image(images_fileset[0]).shape
        xinit = s[0] 
        yinit = s[1]
        N_cam = len(images_fileset)
       
        
        
        with torch.no_grad():
            pred_tot = []
            id_list = []
            count = 0
            logger.debug('Image segmentation by the CNN')
            im = Image.new("RGB", (Sx, Sy))
            new_size, padding = ResizeFit((xinit, yinit)).padding(im)       
            for inputs, id_im in tqdm(loader):
                inputs = inputs.to(device) #input image on GPU
                outputs = evaluate(inputs, model_segmentation)  #output image
                outputs = F.interpolate(outputs, new_size, mode = 'bilinear')
                pred_tot.append(outputs)
                id_list.append(id_im)
                count += 1
        pred_tot = torch.cat(pred_tot, dim = 0)
        pred_pad = torch.zeros((N_cam, len(label_names), xinit, yinit))
        pred_pad[:,:,padding[0]:pred_pad.size(2)-padding[2],padding[1]:pred_pad.size(3)-padding[3]] = pred_tot

#        if resize:
#                pass
#        else:
#     
#            pred_pad = torch.zeros((N_cam, len(label_names), xinit, yinit)) #reverse the crop in order to match the colmap parameters
#            pred_pad[:,:,(xinit-Sx)//2:(xinit+Sx)//2,(yinit-Sy)//2:(yinit+Sy)//2] = pred_tot #To fit the camera parameters
        
        return pred_pad, id_list
