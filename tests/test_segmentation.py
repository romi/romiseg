#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import torch
from packaging import version
from plantdb.commons import io
from plantdb.commons.test_database import test_database

from romiseg.models.unet import ResNetUNet
from romiseg.predict.segmentation import file_segmentation
from romiseg.predict.segmentation import fileset_segmentation
from romiseg.predict.segmentation import segmentation


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        """Set up any state specific to the execution of the given class (called before every test in the class)."""
        # Define image dimensions for segmentation
        self.Sx = 256
        self.Sy = 256
        # Use CPU for test environment
        self.device = torch.device("cpu")
        # Set dataset and model names for testing
        self.dataset_name = 'real_plant'
        self.model_name = 'Resnet_896_896_epoch50.pt'
        # Initialize test database with models
        db = test_database(self.dataset_name, with_models=True)
        self.tmp_dir = db.path()
        # Connect to database with unsafe mode (allows write operations)
        db.connect(unsafe=True)
        # Get scan from database and extract image file paths
        scan = db.get_scan(self.dataset_name)
        self.images_fileset = [f.path() for f in scan.get_fileset('images').get_files()]
        # Get models fileset and specific model file
        self.models_fileset = db.get_scan('models').get_fileset('models')
        self.model_file = self.models_fileset.get_file(self.model_name.split('.')[0])
        # Extract label names from model metadata
        self.label_names = self.model_file.get_metadata('label_names')

        # Handle PyTorch version compatibility for models >= 2.6.0
        if version.parse(torch.__version__) >= version.parse('2.6.0'):
            # Convert model to state dictionary format for newer PyTorch versions
            print('Converting model to state dict')
            from romiseg.cli.convert_models import setup_module_alias
            from romiseg.cli.convert_models import convert_model_to_state_dict
            setup_module_alias()
            # Convert model and save to new file
            state_model_file = convert_model_to_state_dict(self.model_file.path())
            self.model_name = self.model_name.replace('.pt', '_state')
            self.model_file = self.models_fileset.create_file(self.model_name)
            self.model_file.import_file(state_model_file)
            # Preserve label names in new model file
            self.model_file.set_metadata('label_names', self.label_names)

    def tearDown(self):
        """Clean up after the test."""
        import shutil
        shutil.rmtree(self.tmp_dir)

    def test_segmentation(self):
        """Test segmentation function."""
        result, id_list = segmentation(self.Sx, self.Sy, self.images_fileset, self.model_file, self.device)

        # Verify result is a tensor and id_list matches input image count
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(len(id_list), len(self.images_fileset))

    def test_fileset_segmentation(self):
        """Test fileset_segmentation function."""
        pred_images = fileset_segmentation(self.Sx, self.Sy, self.images_fileset, self.model_file, self.device)

        # Verify output contains tensor predictions
        self.assertIsInstance(pred_images[0], torch.Tensor)

    def test_file_segmentation(self):
        """Test file_segmentation function."""
        image_path = self.images_fileset[0]
        # Initialize model with appropriate number of output classes
        model = ResNetUNet(len(self.label_names))
        # Load model weights from state dictionary
        model.load_state_dict(io.read_torch(self.model_file))
        # Perform segmentation on single file
        result = file_segmentation(self.Sx, self.Sy, image_path, model, self.label_names, self.device)
        # Verify output is a tensor
        self.assertIsInstance(result, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
