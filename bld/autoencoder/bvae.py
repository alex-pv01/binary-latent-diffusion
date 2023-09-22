import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms

from bld.modules.modules import Encoder, Decoder, BinaryQuantizer
from bld.modules.losses import TVLoss, VGGLoss, WeightedLoss


class BVAEModel(nn.Module):
    def __init__(self, 
                 device,
                 resolution
                 ):
        super(BVAEModel, self).__init__()
        self.device = device
        self.encoder = Encoder(ch = resolution, 
                         out_ch = 3, 
                         num_res_blocks = 2,
                         attn_resolutions = [16], 
                         ch_mult = (2,4),
                         in_channels = 3,
                         resolution = resolution, 
                         z_channels = resolution,
                         double_z = False).to(self.device)
        self.quantizer = BinaryQuantizer().to(self.device)
        self.decoder = Decoder(ch = resolution, 
                         out_ch = 3, 
                         num_res_blocks = 2,
                         attn_resolutions = [16], 
                         ch_mult = (2,4),
                         in_channels = 3,
                         resolution = resolution, 
                         z_channels = resolution).to(self.device)
        self.loss = WeightedLoss([VGGLoss(shift=2),
                             nn.MSELoss(),
                             TVLoss(p=1)],
                             [1, 40, 10]).to(self.device)

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
                                  lr=lr, betas=(0.5, 0.9)).to(self.device)
        return [opt_ae], []


def train_b_vae(bvae, dataset, num_epochs, lr, resolution = 32):
    opt = optim.Adam(bvae.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for image in dataset:
            trans = transforms.ToTensor()

            img = image.copy()
            img = img.resize((resolution,resolution))

            tensor_image = trans(img).unsqueeze(0).to(bvae.device)

            opt.zero_grad()
            input_data = tensor_image
            loss = bvae.training_step(input_data)

            loss.backward()
            opt.step()
        print("Epoch ", epoch)

