import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms

from datasets import load_dataset

from models.modules import Encoder, Decoder, BinaryQuantizer
from models.losses import TVLoss, VGGLoss, WeightedLoss


class BVAEModel(nn.Module):
    def __init__(self, 
                 device
                 ):
        super(BVAEModel, self).__init__()
        self.encoder = Encoder(ch = 32, 
                         out_ch = 3, 
                         num_res_blocks = 2,
                         attn_resolutions = [16], 
                         ch_mult = (2,4),
                         in_channels = 3,
                         resolution = 32, 
                         z_channels = 32,
                         double_z = False)
        self.quantizer = BinaryQuantizer()
        self.decoder = Decoder(ch = 32, 
                         out_ch = 3, 
                         num_res_blocks = 2,
                         attn_resolutions = [16], 
                         ch_mult = (2,4),
                         in_channels = 3,
                         resolution = 32, 
                         z_channels = 32)
        self.device = device

        self.loss = WeightedLoss([VGGLoss(shift=2),
                             nn.MSELoss(),
                             TVLoss(p=1)],
                             [1, 40, 10])#.to(self.device)

    def preprocess(self, x):
        return x
    
    def postprocess(self, dec):
        return dec

    def encode(self, x):
        h = self.preprocess(x)
        h = self.encoder.forward(h)
        quant = self.quantizer.forward(h)
        return quant
    
    def decode(self, quant):
        dec = self.decoder.forward(quant)
        dec = self.postprocess(dec)
        return dec

    def forward(self, input):
        quant = self.encode(input)
        dec = self.decode(quant)
        return dec
        
    def training_step(self, x):
        xrec = self.forward(x)
        loss = self.loss(x, xrec)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []


def train_b_vae(bvae, dataset, num_epochs, lr):
    opt = optim.Adam(bvae.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for image in dataset:
            trans = transforms.ToTensor()

            img = image.copy()
            img = img.resize((32,32))

            tensor_image = trans(img).unsqueeze(0)

            opt.zero_grad()
            input_data = tensor_image
            loss = bvae.training_step(input_data)

            loss.backward()
            opt.step()
        print("Epoch ", epoch)


def main():
    # Instantiate and train the B-VAE
    dataset  = load_dataset("beans", split="train")[:500]['image']


    # Check if a GPU is available
    if torch.cuda.is_available():
        # Set the device to the first available GPU
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        # If no GPU is available, use the CPU
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")

    model = BVAEModel(device)
    num_epochs = 10
    learning_rate = 1e-3

    train_b_vae(model, dataset, num_epochs, learning_rate)

if __name__== '__main__':
    main()