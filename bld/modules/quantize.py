import torch 
import torch.nn as nn


class BinaryQuantizer(nn.Module):
    def forward(self, h):
        sigma_h = torch.sigmoid(h)
        binary = torch.bernoulli(sigma_h)
        aux_binary = binary.detach() + sigma_h - sigma_h.detach()
        return aux_binary