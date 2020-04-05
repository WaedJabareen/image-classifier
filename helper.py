# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:08:34 2020

@author: waed Jabareen
"""
from torchvision import datasets, transforms
import torch
from torchvision import models
import numpy as np
from PIL import Image
# Function train_transformer(train_dir) performs training transformations on a dataset
def create_data_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    
    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    return trainloader,validloader,testloader,train_data
# Function load_checkpoint(checkpoint_path) loads our saved deep learning model from checkpoint
def load_checkpoint(checkpoint_path):
    # Load the saved file
    checkpoint = torch.load(checkpoint_path)
    
    # Load Defaults if none specified
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Function process_image(image_path) performs cropping, scaling of image for our model
def process_image(image_path):
    img = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transformed_img = transform(img)
    return transformed_img
 # Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
def check_gpu(gpu_arg):
       # If gpu_arg is false then simply return the cpu device
        if not gpu_arg:
           return torch.device("cpu")
        
        # If gpu_arg then make sure to check for CUDA before assigning it
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Print result
        if device == "cpu":
            print("CUDA was not found on device, using CPU instead.")
       
        return device
   