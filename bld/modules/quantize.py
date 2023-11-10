import torch 
import torch.nn as nn


class BinaryQuantizer(nn.Module):
    def forward(self, h):
        sigma_h = torch.sigmoid(h)
        binary = torch.bernoulli(sigma_h)
        aux_binary = binary.detach() + sigma_h - sigma_h.detach()
        #return aux_binary, torch.tensor(0.0, device=self.device).detach()
        return aux_binary


class BinaryVectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, use_tanh=False):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        act = nn.Sigmoid
        if use_tanh:
            act = nn.Tanh
        self.proj = nn.Sequential(
            nn.Conv2d(num_hiddens, codebook_size, 1),  # projects last encoder layer to quantized logits
            act(),
            )
        self.embed = nn.Embedding(codebook_size, emb_dim)
        self.use_tanh = use_tanh

    def quantizer(self, x, deterministic=False):
        if self.use_tanh:
            x = x * 0.5 + 0.5
            if deterministic:
                x = (x > 0.5) * 1.0 
            else:
                x = torch.bernoulli(x)
            x = (x - 0.5) * 2.0
            return x
            
        else:
            if deterministic:
                x = (x > 0.5) * 1.0 
                return x
            else:
                return torch.bernoulli(x)

    def forward(self, h, deterministic=False):

        z = self.proj(h)

        # code_book_loss = F.binary_cross_entropy_with_logits(z, (torch.sigmoid(z.detach())>0.5)*1.0)
        code_book_loss = (torch.sigmoid(z) * (1 - torch.sigmoid(z))).mean()

        z_b = self.quantizer(z, deterministic=deterministic)

        z_flow = z_b.detach() + z - z.detach()

        z_q = torch.einsum("b n h w, n d -> b d h w", z_flow, self.embed.weight)

        # return z_q,  code_book_loss, {
        #     "binary_code": z_b.detach()
        # }, z_b.detach()

        return z_flow, code_book_loss