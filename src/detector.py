import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pydicom import dcmread
from utils import get_filenames

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms, models
from PIL import Image

# create custom dataset class for dataloader to ingest
class DicomDataSet(Dataset):
    def __init__(self, data_dir, image_transforms):
        self.paths = get_filenames(data_dir)
        self.transforms = image_transforms
        
    def __len__(self):
        return len(self.paths)

    @staticmethod
    def rescale_hu(dicom_ds):
        '''
        Rescale Dicom file image to the Hounsfield scale
        using the following linear transformation:
        rescaled pixel = pixel * RescaleSlope + RescaleIntercept
        '''
        image = dicom_ds.pixel_array
        return image * dicom_ds.RescaleSlope + dicom_ds.RescaleIntercept
    
    def transform_image(self, ds):
        image = self.rescale_hu(ds)
        PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
        tensor_image = self.transforms(PIL_image)
        return tensor_image

    def __getitem__(self, idx):
            img_loc = self.paths[idx]
            ds = dcmread(img_loc)
            return self.transform_image(ds)

if __name__ == "__main__":
    dicom_filepath = '../data\\sample_scans_2\\e6af016d16fa05a0\\01cd156160af4a7d\\568c70068d300648'
    dicom_file = dcmread(dicom_filepath)
    
    data_dir = '../data'
    
    # Define image transforms for ImageNet
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
    imagenet_transforms = transforms.Compose([transforms.Resize(256), 
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              imagenet_normalize]
                                              )
    
    db = DicomDataSet(data_dir, imagenet_transforms)
    
    trainloader = DataLoader(db, batch_size=32, shuffle=True)
    images = next(iter(trainloader))
    print(images.shape)
    
    
    