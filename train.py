# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:30:34 2020

@author: waed Jabareen
"""
import argparse
from  model import FlowerClassifier as fc
from helper import create_data_loaders
from torch import optim
from torch import nn

# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Image Classifier Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    return args


# =============================================================================
# Main Function
# =============================================================================

# Function main() is where all functions are called and executed 
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'ImageClassifier/flowers'

    # Pass transforms in, then create data loaders
    trainloader,validloader,testloader,train_data = create_data_loaders(data_dir)

    # Load Model
    model = fc.primaryloader_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = fc.initial_classifier(model, 
                                         hidden_units=args.hidden_units)
     
    # Check for GPU
    device = fc.check_gpu(gpu_arg=args.gpu);
    print(device)
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 30
    steps = 0
    
    # Train the classifier layers using backpropogation
    trained_model = fc.network_trainer(fc,model, trainloader, testloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    
    # Validate the model
    fc.validate_model(fc,trained_model, testloader,criterion, device)
    
    # Save the model
    fc.initial_checkpoint(trained_model, args.save_dir, train_data)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()