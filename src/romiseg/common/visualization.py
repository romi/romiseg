#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Visualization Utilities.

This module contains utilities for visualizing data and results.
"""

import datetime
import matplotlib.pyplot as plt
import torch


class plotclass(object):
    """
    This class contains functions that can plot graphs and several curves on a graph (and save the plot).
    
    Parameters
    ----------
    xval : array or list/tuple of arrays, optional
        x axis values, either one array or multiple arrays.
    yval : array or list/tuple of arrays, optional
        y axis values, either one array or multiple arrays.
        
    Returns
    -------
    matplotlib.figure.Figure
        Plot of f(x) = y, or overlayed curves f(xi) = yi.
    """
        
    def __init__(self, xval=None, yval=None):
        # plot parameters
        self.figsize = (9, 6)
        self.fontsize = 13
        self.color = 'steelblue'
        self.marker = 'o-'
        self.linewidth = 2
        self.title = 'My Title'
        self.xlabel = 'x label (unit)'
        self.ylabel = 'y label (unit)'
        
        # saving parameters
        self.date = True
        self.save_name = 'Figure'
        self.extension = '.tiff'
        
        # multiplot parameters
        self.label_item = ['MyLabel']
        self.label_list = self.label_item * 100
        self.color_list = [self.color] + ['indianred', 'seagreen', 'mediumslateblue', 'maroon', 'palevioletred'
                          'orange', 'lightseagreen', 'dimgrey', 'slateblue']

        self.xval = xval
        self.yval = yval
    
    def plotting(self):
        if type(self.xval) != tuple and type(self.xval) != list:  # converts to a list if there is only one element
            self.xval = [self.xval]
        
        if type(self.yval) != tuple and type(self.yval) != list:  # converts to a list if there is only one element
            self.yval = [self.yval]
            
        NX = len(self.xval)
        NY = len(self.yval)
        if NX != NY:
            if NX != 1:
                print('OooOouups! X should be a list or tuple containing either 1 array or the same number of arrays as Y')
                return False
            else: 
                self.xval = self.xval * NY  # extends the X list to match the size of the Y list
        
        f = plt.figure(figsize=self.figsize)
   
        for i in range(NY):
            plt.title(self.title, fontsize=self.fontsize + 2)
            plt.xlabel(self.xlabel, fontsize=self.fontsize)
            plt.ylabel(self.ylabel, fontsize=self.fontsize)
            plt.plot(self.xval[i], self.yval[i], self.marker, color=self.color_list[i],
                    linewidth=self.linewidth, label=self.label_list[i])  # overlays new curve on the plot
        if NY > 1:
            plt.legend()
        if self.date:            
            plt.savefig(str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')) + self.save_name + self.extension)
        else: 
            plt.savefig(self.save_name + self.extension)
        return f


class showclass(object):
    """
    This class contains functions that can show images and subplot several images (and save the plot).
    
    Parameters
    ----------
    None
        
    Returns
    -------
    matplotlib.figure.Figure
        Plot of the image x or subplots of images xi.
    """
        
    def __init__(self):
        # imshow parameters
        self.figsize = (9, 6)  # figure size
        self.fontsize = 13  # font size
        self.title = 'My Title'  # figure title
        self.cmap = 'inferno'

        # multiple image imshow        
        self.title_item = ['MyLabel']
        self.title_list = self.title_item * 100
        self.col_num = 3
       
        # figure save parameters
        self.date = True  # write date and time before figure name
        self.save_name = 'Figure'  # setting the format!
        self.save_folder = 'alienlab_images/'
        self.extension = '.tiff'
        self.save_im = True
        self.spacing = 0.2

    def multi(self, x=None):
        if type(x) != tuple and type(x) != list:  # When there is only one image, convert it in a list element
            x = [x]
            
        N = len(x)

        COLS = self.col_num
        if N == 1:  # when there is only one image
            ROWS, COLS = 1, 1
        elif N % COLS == 0:  # when its a multiple of the number of columns expected, no extra row should be added
            ROWS = N // COLS
        else: 
            ROWS = N // COLS + 1  # extra row for remaining figures otherwise    
            
        f = plt.figure(figsize=self.figsize)
        
        for i in range(N):
            plt.subplot(ROWS, COLS, i+1)
            plt.imshow(x[i], cmap=self.cmap)
            plt.axis('off')
            plt.grid(False)
            plt.subplots_adjust(wspace=self.spacing, hspace=self.spacing)
            if self.title_list is not None:
                plt.title(self.title_list[i], fontsize=self.fontsize)  # update subfigure title
        
        return f
    
    def saving(self, x=None):
        f = self.multi(x)
        if self.date:            
            f.savefig(self.save_folder + str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_')) + self.save_name + self.extension,
                    bbox_inches='tight', frameon=False)  # save with the date and time befor the figure name
        else: 
            f.savefig(self.save_folder + self.save_name + self.extension, bbox_inches='tight', frameon=False)
        return f
    
    def showing(self, x=None):
        plt.ion()
        f = self.multi(x)
        f.show()    
        plt.pause(0.01)
        input("Press [enter] to continue.")
        return f


def plot_dataset(train_loader, label_names, batch_size, showit=False):
    """
    Plot a batch of images and their labels from a data loader.
    
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Data loader containing images and labels.
    label_names : list
        List of label names.
    batch_size : int
        Batch size.
    showit : bool, optional
        Whether to show the plot interactively. Default is False.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plotted images and labels.
    """
    all_data = next(iter(train_loader))
    images = all_data[0]
    label = all_data[1]
    # plot 4 images to visualize the data
    images_tot = []
    titles_tot = []
    for j in range(batch_size):
        if j * len(label_names) >= 14 * 14:
            break
        img = images[j]
        img = img.permute(1, 2, 0)
        images_tot.append(img)
        titles_tot.append('image')
        for i in range(len(label_names)):
            img = label[j, i, :, :] * 255  # .int()
            images_tot.append(img)
            titles_tot.append(label_names[i])
    g = showclass()
    g.save_im = False

    g.col_num = 3
    g.figsize = ((14, 14))
    g.title_list = titles_tot
    if not showit:
        fig = g.saving(images_tot)
    else:
        fig = g.showing(images_tot)

    return fig