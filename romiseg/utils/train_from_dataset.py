#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:56:38 2019

@author: alienor
"""


import torch
from torch.optim import lr_scheduler
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MaxPool2d
from torch.autograd import Variable
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import appdirs

from tqdm import tqdm


import romiseg.utils.alienlab as alien

from romidata import io
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

from romidata import fsdb
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from torchvision import transforms

import matplotlib.pyplot as plt
import os
import requests
import copy
import math
import random


import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##################LOAD PRE-TRAINED WEIGHTS############

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

def save_and_load_model(weights_folder, model_segmentation_name):


    #if not already saved, download from database 
    if model_segmentation_name not in os.listdir(weights_folder):
        
        url = 'http://db.romi-project.eu/models/' + model_segmentation_name 
        
        download_file(url, weights_folder)
   
    model_segmentation = torch.load(weights_folder + '/' + model_segmentation_name)
    try:
        model_segmentation = model_segmentation[0]
    except:
        model_segmentation = model_segmentation
    
    try: 
        model_segmentation = model_segmentation.module
    except:
        model_segmentation = model_segmentation
            
    return model_segmentation

def model_from_fileset(model_file):
    model_segmentation = io.read_torch(model_file)
    try:
        model_segmentation = model_segmentation[0]
    except:
        model_segmentation = model_segmentation
    
    try: 
        model_segmentation = model_segmentation.module
    except:
        model_segmentation = model_segmentation
    label_names = model_file.get_metadata('label_names')
    return model_segmentation, np.sort(label_names)

    

def gaussian(ins, is_training, mean, stddev, dyn = 1):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return torch.clamp(ins + noise, 0, 1)
    return torch.clamp(ins,0, dyn)

class ResizeFit(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def padding(self, img):
        aspect_ratio = self.size[0] / self.size[1]
        old_aspect_ratio = img.size[0] / img.size[1]

        if aspect_ratio < old_aspect_ratio:
            new_size = (self.size[0], int(1/old_aspect_ratio * self.size[0]))
        else:
            new_size = (int(old_aspect_ratio * self.size[1]), self.size[1])

        diff = [self.size[i] - new_size[i] for i in range(2)]
        padding =  diff[0]//2, diff[1]//2, (diff[0] + 1) //2, (diff[1] + 1)//2
        return new_size, padding

    def __call__(self, img):
        from PIL import ImageOps
        new_size, padding = self.padding(img)
        new_img = img.resize(new_size, resample=self.interpolation)
        new_img = ImageOps.expand(new_img, padding)
        return new_img

class ResizeCrop(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def padding(self, img):
        aspect_ratio = self.size[0] / self.size[1]
        old_aspect_ratio = img.size[0] / img.size[1]

        if aspect_ratio > old_aspect_ratio:
            new_size = (self.size[0], int(1/old_aspect_ratio * self.size[0]))
        else:
            new_size = (int(old_aspect_ratio * self.size[1]), self.size[1])

        diff = [- self.size[i] + new_size[i] for i in range(2)]
        padding =  diff[0]//2, diff[1]//2, (diff[0] + 1) //2, (diff[1] + 1)//2


        return new_size, padding

    def __call__(self, img):
        from PIL import ImageOps
        new_size, padding = self.padding(img)
        new_img = img.resize(new_size, resample=self.interpolation)
        new_img = ImageOps.crop(new_img, padding)
        return new_img
    
def init_set(mode, path):
    db = fsdb.FSDB(path)
    db.connect()
    scans = db.get_scans()
    shots = []
    for s in scans:
        f = s.get_fileset('images')
        list_files = f.get_files( query = {'channel':'rgb'})
        #for i in range(len(list_files)):
         #   f0 = list_files[i]
            #print(f0.metadata.keys(), f0.metadata['channel'], f0.metadata['shot_id'])
        shots += [{"scan": s.id, "shot_id": list_files[i].metadata['shot_id']} for i in range(len(list_files))]

    channels = f.get_metadata('channels')
    channels = copy.copy(channels)
    channels.remove('rgb')
    db.disconnect()
    return shots, np.sort(channels)

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle, fill):
        self.angle = angle
        self.fill = fill

    def __call__(self, x):
        return TF.rotate(x, self.angle, fill=self.fill)

class Dataset_im_label(Dataset):
    """Data handling for Pytorch Dataloader"""

    def __init__(self, shots, channels, size, path):

        self.shots = shots
        self.channels = channels
        self.size = size
        self.path = path


    def __getitem__(self, index):
        db = fsdb.FSDB(self.path)
        db.connect()
        db_file_meta = self.shots[index]
        s = db.get_scan(db_file_meta['scan'])
        image_file = s.get_fileset('images').get_files(query = {'channel':'rgb', 'shot_id':db_file_meta['shot_id']})[0]
        angle = random.randint(-90, 90)
        scale = 1 + np.random.rand()


        image = Image.fromarray(io.read_image(image_file))
        padding = image.size
        resize = ResizeCrop(self.size)
        pad = transforms.Pad(padding, padding_mode='reflect')
        crop = transforms.CenterCrop(self.size)
        scale = transforms.Resize(np.asarray((np.array(self.size) * scale),dtype=int).tolist())
        #id_im = db_file.id
        rot = MyRotationTransform(angle, fill=(0,0,0))
        trans = transforms.Compose([resize, scale, pad, rot, crop, transforms.ToTensor()])
        
        
        t_image = trans(image)
        t_image = t_image[0:3, :, :] #select RGB channels
        t_image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])(t_image)
        dyn = t_image.max()
        t_image = gaussian(t_image, is_training = True,
                           mean = 0, stddev =  np.random.rand()*1/100, dyn = dyn)
        torch_labels = []

        
        for i, c in enumerate(self.channels):
            
            labels = s.get_fileset('images').get_files(query = {'channel':c, 'shot_id':db_file_meta['shot_id']})[0]
            if c != 'background':
                t_label = Image.fromarray(1.0 * (io.read_image(labels) > 0))
            else:
                t_label = Image.fromarray(1.0 * (io.read_image(labels) == 255))

            num_bands = len(t_label.getbands())

            rot.fill = 0.
            t_label = trans(t_label)
            torch_labels.append(t_label)
        torch_labels = torch.cat(torch_labels, dim = 0)
        db.disconnect()

        return t_image, torch_labels

    def __len__(self):  # return count of sample
        return len(self.shots)

    def read_label(self, labels):
        somme = labels.sum(axis = 0)
        background = somme == 0
        background = background.astype(somme.dtype)
        background = background*255
        dimx, dimy = background.shape
        background = np.expand_dims(background, axis = 0)
        if labels.shape[0] == 5:
            labels = np.concatenate((background*0, labels), axis = 0)

        labels = np.concatenate((background, labels), axis = 0)


        return labels




def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def my_metric(outputs: torch.Tensor, labels: torch.Tensor):
    inds  = labels != 0
    bools = outputs[inds] == labels[inds]
    
    return torch.mean(bools.float())  # Or thresholded.mean() if you are interested in average across the batch



def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(f_weights, dataloaders, model, optimizer, scheduler, writer, num_epochs=25, viz = False, label_names = []):
    L = {'bce':[], 'dice':[]}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    loss_test = []
    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        print('Running epoch %d/%d'%(epoch, num_epochs), end="\r")

        #since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                #for param_group in optimizer.param_groups:
                #    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    #print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #print(model.conv_last.weight.grad, model.conv_last.bias.grad)
                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            L['bce'].append(metrics['bce']/epoch_samples)
            L['dice'].append(metrics['dice']/epoch_samples)
            writer.add_scalar('train/crossentropy', epoch_loss, epoch)
        
            if phase == 'val':
                inputs, labels = next(iter(dataloaders[phase]))
                inputs = inputs.to(device)
                labels = labels.to(device)
                lab = torch.argmax(labels, dim = 1)
                # forward
                # track history if only in train
                outputs = model(inputs)
                out = torch.argmax(outputs, dim = 1)
                #loss_test.append(my_metric(out, lab))
                
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                #print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
            #plot 4 images to visualize the data
        if viz == True:
 
            plt.ioff()
            fig = plt.figure(figsize = (14, 6))
        
            col = len(label_names)
            for i in range(col):
                plt.subplot(2, col, 2*i + 1)
                plt.axis('off')
                plt.grid(False)
                img = inputs[0]
                img = torchvision.transforms.ToPILImage()(img.detach().cpu())
                plt.imshow(img)
                plt.title('image')
                img = F.sigmoid(outputs[0,i,:,:])
                img = torchvision.transforms.ToPILImage()(img.detach().cpu())
                plt.subplot(2, col, 2*i + 2)
                plt.axis('off')
                plt.grid(False)
                plt.imshow(img)
                plt.title(label_names[i])
            
            
            
            writer.add_figure('Segmented images', fig, epoch)
        if epoch%10==0:
            model_name =  'tmp_epoch%d'%epoch
        
            file = f_weights.create_file(model_name)
            io.write_torch(file, model)
            file.set_metadata({'model_id':model_name, 'label_names':label_names.tolist()})


            
        
        #time_elapsed = time.time() - since
        #print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights 
    model.load_state_dict(best_model_wts)
    return model, L#, loss_test




def fine_tune_train(path_train, path_val, weights_folder, label_names, tsboard_name,
                    model_segmentation_name, Sx, Sy, num_epochs, scan):
    num_classes = len(label_names)
    
    trans = transforms.Compose([
    transforms.CenterCrop((Sx, Sy)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), #imagenet
    transforms.ToTensor()
])
    
    image_train, target_train = init_set('', path_train)
    image_val, target_val = init_set('', path_val)

    train_dataset = Dataset_im_label(image_train, target_train, transform = trans)
    val_dataset = Dataset_im_label(image_val, target_val, transform = trans) 
    
        
    batch_size = min(num_classes, len(image_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    
    fig = plot_dataset(train_loader, label_names, batch_size) #display training set
    plt.show(block=True)
    print('Now the network will train on the data you annotated')

    
    batch_size = 2
       
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        }
    
    
    model = save_and_load_model(weights_folder, model_segmentation_name)

    
    writer = SummaryWriter('test')#tsboard_name)


    a = list(model.children())
    for child in  a[0].children():
        for param in child.parameters():
            param.requires_grad = False
    
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    
    
    model = train_model(dataloaders, model, optimizer_ft, exp_lr_scheduler, writer,  num_epochs = num_epochs)
    ext_name = '_finetune_' + scan + '_epoch%d.pt'%num_epochs
    new_model_name = model_segmentation_name[:-3] + ext_name

    torch.save(model, weights_folder + '/' + new_model_name)
    
    
    return model, new_model_name



def plot_dataset(train_loader, label_names, batch_size, showit=False):
    all_data = next(iter(train_loader))
    images = all_data[0]
    label = all_data[1]
    #plot 4 images to visualize the data
    images_tot = []
    titles_tot = []
    for j in range(batch_size):
        if j * len(label_names) >= 14*14:
            break
        img = images[j]
        img = img.permute(1, 2, 0)
        images_tot.append(img)
        titles_tot.append('image')
        for i in range(len(label_names)):
            img = label[j,i,:,:]*255#.int()
            images_tot.append(img)
            titles_tot.append(label_names[i])
    g = alien.showclass()
    g.save_im = False

    g.col_num = 3
    g.figsize = ((14, 14))
    g.title_list = titles_tot
    if showit == False:
        fig = g.saving(images_tot)
    else: 
        fig = g.showing(images_tot)
    
    return fig

    
# Prediction
def evaluate(inputs, model):

    with torch.no_grad():
        inputs.requires_grad = False
        # Get the first batch
        inputs = inputs.to(device)

        pred = model(inputs)
        # The loss functions include the sigmoid function.
        #for i in range(pred.shape[1]):
        #    pred[:,1,:,:] = F.sigmoid(pred[:,1,:,:])

        pred = F.sigmoid(pred)
        
        
    return pred
    
def test(inputs, labels, model):
    metrics = defaultdict(float)

    with torch.no_grad():
        inputs.requires_grad = False
        # Get the first batch
        inputs = inputs.to(device)

        pred = model(inputs)
        # The loss functions include the sigmoid function.
        #for i in range(pred.shape[1]):
        #    pred[:,1,:,:] = F.sigmoid(pred[:,1,:,:])
        loss = calc_loss(pred, labels, metrics)

        pred = F.sigmoid(pred)
        
        
    return pred, metrics
