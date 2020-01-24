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
The parent directory should be specified in the config file
```
[TrainingDirectory]
path = "/home/tim/shared/"
directory_weights = "weights/"
tsboard = "board/"
directory_dataset = "dataset_stem_green"

[Segmentation2D]
upstream_task = "Scan"
query = "{\"channel\":\"rgb\"}"
labels = "background,flower,peduncle,stem,bud,leaf,fruit"
model_name = "Resnet101"
model_segmentation_name = "Resnet101_896_896_epoch51.pt"
Sx = 896
Sy = 896
epochs = 5
batch = 1
learning_rate = 0.0001


``` 


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
train_cnn.py --config [path/to/config/file]  
```

you can also import the train_cnn function for multiple tests
