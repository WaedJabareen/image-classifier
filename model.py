# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:30:34 2020

@author: waed Jabareen
"""

import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torchvision import models
class FlowerClassifier():
    def __init__(self):
       
        self.criterion = None
    # ------------------------------------------------------------------------------- #
    # Define Functions
    # ------------------------------------------------------------------------------- #
    
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
    
    # primaryloader_model(architecture="vgg16") downloads model (primary) from torchvision
    def primaryloader_model(architecture="vgg16"):
        # Load Defaults if none specified
        if type(architecture) == type(None): 
            model = models.vgg16(pretrained=True)
            model.name = "vgg16"
            print("Network architecture specified as vgg16.")
        else: 
            exec("model = models.{}(pretrained=True)".format(architecture))
            model.name = architecture
        
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False 
        return model
    
    # Function initial_classifier(model, hidden_units) creates a classifier with the corect number of input layers
    def initial_classifier(model, hidden_units):
        # Check that hidden layers has been input
        if type(hidden_units) == type(None): 
            hidden_units = 4096 #hyperparamters
            print("Number of Hidden Layers specificed as 4096.")
        
        # Find Input Layers
        input_features = model.classifier[0].in_features
        
        # Define Classifier
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        return classifier
    
    # Function validation(model, testloader, criterion, device) validates training against testloader to return loss and accuracy
    def validation(model, test_loader, criterion, device):
        test_loss = 0
        test_accuracy = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            
            test_loss += loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += equals.type(torch.FloatTensor).mean()
        
    
        return test_loss/len(test_loader), test_accuracy/len(test_loader)
    
    # Function network_trainer represents the training of the network model
    def network_trainer(self,model, trainloader, testloader,validloader, device, 
                      criterion, optimizer, epochs, print_every, steps):
        # Check Model Kwarg
        if type(epochs) == type(None):
            epochs = 5
            print("Number of Epochs specificed as 5.")    
     
        print("Training process initializing .....\n")
    
        # Train Model
        for e in range(epochs):
            running_loss = 0
            model.train() # Technically not necessary, setting this for good measure
            
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
            
                if steps % print_every == 0:
                    model.eval()
    
                    with torch.no_grad():
                        valid_loss, accuracy = self.validation(model, validloader, criterion, device)
                
                    print("Epoch: {}/{} | ".format(e+1, epochs),
                         "Training Loss: {:.4f} | ".format(running_loss/print_every),
                         "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                         "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
                
                    running_loss = 0
                    model.train()
    
        return model
    
    #Function validate_model(Model, Testloader, Device) validate the above model on test data images
    def validate_model(self,model, testloader, criterion,device):
       # Do validation on the test set
       model.eval()
       with torch.no_grad():
           test_loss, test_accuracy = self.validation(model, testloader, criterion,device)
    
       print(f"Test loss: {test_loss:.3f}.. "
          f"Test accuracy: {100 * test_accuracy:.2f}%..")
    
    # Function initial_checkpoint(model, save_dir, train_data) saves the model at a defined checkpoint
    def initial_checkpoint(model, save_dir, train_data):
           
        # Save model at checkpoint
        if type(save_dir) == type(None):
            print("Model checkpoint directory not specified, model will not be saved.")
        else:
            if isdir(save_dir):
                # Create `class_to_idx` attribute in model
                model.class_to_idx = train_data.class_to_idx
                
                # Create checkpoint dictionary
                checkpoint = {'architecture': model.name,
                              'classifier': model.classifier,
                              'class_to_idx': model.class_to_idx,
                              'state_dict': model.state_dict()}
                
                # Save checkpoint
                torch.save(checkpoint, 'model_checkpoint.pth')
    
            else: 
                print("Directory not found.")
    
    
    
    
