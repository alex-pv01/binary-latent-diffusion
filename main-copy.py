import sys
import os
from tqdm import tqdm
from os.path import join as pjoin

import torch
from torchvision import transforms
import numpy as np

from bld.autoencoder.bvae import BVAEModel, train_b_vae2

import matplotlib.pyplot as plt

from datasets import load_dataset

def plot_tensor_image(tensor, name):
    # Convert the tensor to a NumPy array
    image_array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    # Ensure values are in the range [0, 1] (assuming your tensor values are normalized)
    image_array = image_array.clip(0, 1)
    
    # Display the image
    plt.imshow(image_array)
    plt.axis('off')  # Turn off the axis labels and ticks
    plt.savefig(name)

def main():
    # Instantiate and train the B-VAE
    src_dir = './dataset/HumanML3D/new_joints/'

    npy_files = os.listdir(src_dir)
    npy_files = sorted(npy_files)

    dataset = []

    for npy_file in tqdm(npy_files):
        dataset.append(np.load(pjoin(src_dir, npy_file)))

    # Check if a GPU is available
    if torch.cuda.is_available():
        # Set the device to the first available GPU
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        # If no GPU is available, use the CPU
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")

    resolution = 16

    model = BVAEModel(device, resolution)
    num_epochs = 20
    learning_rate = 1e-3

    train_b_vae2(model, dataset, num_epochs, learning_rate, resolution)

    image = dataset[0]

    trans = transforms.ToTensor()

    img = image.copy()
    img = img.resize((resolution,resolution))

    tensor_image = trans(img).unsqueeze(0).to(device)

    plot_tensor_image(tensor_image, 'original.png')

    r_tensor_image = model.forward(tensor_image) 

    plot_tensor_image(r_tensor_image, 'predicted.png')

    return 0

if __name__== '__main__':
    sys.exit(main())