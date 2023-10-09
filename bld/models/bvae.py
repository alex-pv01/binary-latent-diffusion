import torch 
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from bld.modules.model import Encoder, Decoder
from bld.modules.quantize import BinaryQuantizer


class BVAEModel(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=[], image_key="image", colorize_nlabels=None, monitor=None, remap=None, sane_index_shape=False):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = BinaryQuantizer()
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None: 
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant = self.quantize(h)
        return quant
    
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def forward(self, input):
        quant = self.encode(input)
        dec = self.decode(quant)
        return dec
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)

        if optimizer_idx == 0:
            # autoencoder
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            return aeloss

        if optimizer_idx == 0:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)

        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]

        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parametets())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec = self(x)
        if x.shape[1] > 3:
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
    
    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x-x.max()) - 1.
        return x