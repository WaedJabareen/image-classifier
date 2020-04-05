# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:08:34 2020

@author: waed Jabareen
"""
import argparse
import json
import torch
import numpy as np

from math import ceil
from helper import check_gpu
from helper import load_checkpoint,process_image
# ------------------------------------------------------------------------------- #
# Function Definitions
# ------------------------------------------------------------------------------- #
# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", action="store")
    parser.add_argument("checkpoint", action="store")
    parser.add_argument("--top_k", action="store", default=1, type=int)
    parser.add_argument("--category_names", action="store",
                        default="ImageClassifier/cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", default=False)
    return parser.parse_args()

   


def predict(image_obj, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    tensor_image = torch.from_numpy(np.asarray(image_obj)).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze_(0)

    tensor_image.to('cpu')
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        outputs = model(tensor_image)

        probs = torch.exp(outputs)
        top_p, top_class = probs.topk(topk, dim=1)

        top_p = top_p.numpy()[0]
        top_class = top_class.numpy()[0]
    idx_to_class = {val: key for key, val in
                        model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]
    return top_p, top_class
    


def print_probability(top_class, cat_to_name,top_ps):
    """
    Converts two lists into a dictionary to print on screen
    """
    
    for i, c in enumerate(top_class):
        print(f"Prediction {i+1}: "
              f"{cat_to_name[c]} .. "
              f"({100.0 * top_ps[i]:.3f}%)")
    

# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    print(model)
    # Process Image
    image = process_image(args.image_path)
    
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Use `processed_image` to predict the top K most likely classes
    top_ps, top_class = predict(image, model, args.top_k)
    
    # Print out probabilities
    print_probability(top_class,cat_to_name, top_ps)

# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()