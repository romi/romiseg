### Training a network from a database.


We suggest the following organisation of folder for training and database
```
.  
├── weights #empty, to save model weights 
├── board #empty, for tensorboard visualisation  
└── data  
    └── dataset  
            ├── train  
            │       ├── images #training images (png or jpg) RGB
            │       └── labels #training labels (specific format, read the doc) 
            ├── val  
            │       ├── images  
            │       └── labels  
            └── test  
                    ├── images  
                    └── labels  
```
The parent directory should be sepcified in the file [parameters_train.toml](https://github.com/romi/Segmentation/blob/master/romiseg/parameters_train.toml)


## Generate a dataset with blender
To be completed

## Parameters
the training parameters are to be filled in parameters_train.toml.   
-size of the training images (center crop from images in train/images)  
-type of segmentation network (for now they are sourced in [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch)    
-batch size
-number of epochs  
-learning rate  

## Launch training

simply launch   
```  
python train_cnn.py  
```

you can also import the train_cnn function for multiple tests
