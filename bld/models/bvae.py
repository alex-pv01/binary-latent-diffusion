import torch 
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from bld.modules.model import Encoder, Decoder
from bld.modules.quantize import BinaryQuantizer, BinaryVectorQuantizer


class BVAEModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 codebook_size=None,
                 emb_dim=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 quantize="binary",
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        if quantize == "vector":
            assert codebook_size is not None and emb_dim is not None, "Need to specify codebook_size and emb_dim for vector quantization."
            raise NotImplementedError("Vector quantization not implemented yet.")
            #TODO: Fix compatibility errors with pytorch lightning
            self.quantize = BinaryVectorQuantizer(codebook_size=codebook_size, emb_dim=emb_dim, num_hiddens=ddconfig["z_channels"])
        elif quantize == "binary":
            self.quantize = BinaryQuantizer()
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], emb_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(emb_dim, ddconfig["z_channels"], 1)
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
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
        #quant, emb_loss, info = self.quantize(h)
        #return quant, emb_loss, info
        quant = self.quantize(h)
        return quant

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    #def decode_code(self, code_b):
    #    quant_b = self.quantize.embed_code(code_b)
    #    dec = self.decode(quant_b)
    #    return dec

    def forward(self, input):
        #quant, diff, _ = self.encode(input)
        quant = self.encode(input)
        dec = self.decode(quant)
        #return dec, diff
        return dec

    def get_input(self, batch, k):
        #print("Keys: ", batch.keys())
        #print(type(batch))
        #print(type(batch[k]))
        #print(k)
        #print(batch[k].shape)
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        #xrec, codebook_loss = self(x)
        xrec = self(x)

        qloss = 0

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            #self.log("train/codebook_loss", codebook_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            #return aeloss + codebook_loss
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        #xrec, codebook_loss = self(x)
        xrec = self(x)

        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        #rec_loss = log_dict_ae["val/rec_loss"]
        #self.log("val/rec_loss", rec_loss,
        #           prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/codebook_loss", codebook_loss,
        #            prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/discloss", discloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
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
        xrec  = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
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
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x