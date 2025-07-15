import torch
import numpy as np
import os
import argparse
from copy import deepcopy

import matplotlib.pyplot as plt

def load_model_weights(model_path):
    """Load weights from a PyTorch model file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    return state_dict

def extract_rnn_weights(state_dict):
    """Extract RNN weights from state dictionary."""
    # Try different possible weight naming conventions
    if 'rnn_cell.weight_hh' in state_dict:
        return state_dict['rnn_cell.weight_hh'].numpy()
    elif 'rnn.weight_hh_l0' in state_dict:
        return state_dict['rnn.weight_hh_l0'].numpy()
    else:
        # Print available keys to help debug
        print("Available keys in state_dict:")
        for key in state_dict.keys():
            if 'weight' in key and ('rnn' in key.lower() or 'lstm' in key.lower() or 'gru' in key.lower()):
                print(f"  {key}")
        
        raise KeyError("Could not find RNN weight matrices in the model state dict")
    
def extract_wih_weights(state_dict):
    """Extract input-to-hidden weights from state dictionary."""
    # Try different possible weight naming conventions
    if 'rnn_cell.weight_ih' in state_dict:
        return state_dict['rnn_cell.weight_ih'].numpy()
    elif 'rnn.weight_ih_l0' in state_dict:
        return state_dict['rnn.weight_ih_l0'].numpy()
    else:
        # Print available keys to help debug
        print("Available keys in state_dict:")
        for key in state_dict.keys():
            if 'weight' in key and ('rnn' in key.lower() or 'lstm' in key.lower() or 'gru' in key.lower()):
                print(f"  {key}")
        
        raise KeyError("Could not find input-to-hidden weight matrices in the model state dict")

def compare_weights(weights1, weights2):
    """Compare two weight matrices and return their difference."""
    
    # Calculate the difference
    if weights1.shape[1] > weights2.shape[1]:
        difference = weights1[:, :weights2.shape[1]] - weights2
    elif weights1.shape[1] < weights2.shape[1]:
        difference = weights1 - weights2[:, :weights1.shape[1]]
    else:
        difference = weights1 - weights2

    return difference

def visualize_weights(weights1, weights2, difference, title1="Model 1 Weights", 
                     title2="Model 2 Weights", title_diff="Weight Difference"):
    """Visualize the weights and their difference using matshow."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot first weight matrix
    im1 = ax1.matshow(weights1, cmap='viridis')
    ax1.set_title(title1)
    fig.colorbar(im1, ax=ax1)
    
    # Plot second weight matrix
    im2 = ax2.matshow(weights2, cmap='viridis')
    ax2.set_title(title2)
    fig.colorbar(im2, ax=ax2)
    
    # Plot difference
    im3 = ax3.matshow(difference, cmap='coolwarm')
    ax3.set_title(title_diff)
    fig.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare RNN weights from PyTorch models")
    parser.add_argument("--model1", type=str, default=None, help="First model filename")
    parser.add_argument("--model2", type=str, default=None, help="Second model filename (optional)")
    parser.add_argument("--weights", type=str, default='Whh', help="Path to weights file (optional, if not using model files)")

    args = parser.parse_args()
    
    # Load first model
    model1_path = os.path.join(args.model1,'best_model.pth') if os.path.isdir(args.model1) else args.model1
    state_dict1 = load_model_weights(model1_path)
    if args.weights=='Whh':
        weights1 = extract_rnn_weights(state_dict1)
    elif args.weights=='Wih':
        weights1 = extract_wih_weights(state_dict1)
    else:
        raise ValueError("Invalid weights type specified. Use 'Whh' for RNN hidden weights or 'Wih' for input-to-hidden weights.")
    
    # If second model is provided, compare them
    if args.model2:
        model2_path = os.path.join(args.model2,'best_model.pth') if os.path.isdir(args.model2) else args.model2
        state_dict2 = load_model_weights(model2_path)
        if args.weights=='Whh':
            weights2 = extract_rnn_weights(state_dict2)
        elif args.weights=='Wih':
            weights2 = extract_wih_weights(state_dict2)
        else:
            raise ValueError("Invalid weights type specified. Use 'Whh' for RNN hidden weights or 'Wih' for input-to-hidden weights.")

        # Compare weights
        diff = compare_weights(weights1, weights2)
        
        # Visualize
        visualize_weights(weights1, weights2, diff, 
                        title1=f"Weights from {args.model1}", 
                        title2=f"Weights from {args.model2}")
        
        # Print some statistics
        print(f"Mean absolute difference: {np.abs(diff).mean()}")
        print(f"Max absolute difference: {np.abs(diff).max()}")
    else:
        # If only one model is provided, just visualize its weights
        plt.figure(figsize=(10, 8))
        im = plt.matshow(weights1, cmap='viridis', fignum=1)
        plt.colorbar(im)
        plt.title(f"RNN Weights from {args.model1}")
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"Shape of weights: {weights1.shape}")
        print(f"Mean weight value: {weights1.mean()}")
        print(f"Weight std deviation: {weights1.std()}")

if __name__ == "__main__":
    main()