from typing import Callable
import torch 
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from functools import partial
from contextlib import contextmanager
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

import pdb

from bld.utils import count_params, exists, instantiate_from_config
from bld.modules.ema import LitEma
from bld.modules.diffusionmodules.utils import make_beta_schedule
from bld.modules.distributions.distributions import BernoulliDistribution

__conditioning_keys__ = {"concat": "c_concat",
                         "crossattn": "c_crossattn",
                         "adm": "y"}


def disabled_train(self, mode=True):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore.
    """
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device = device) + r2


# Binary DDPM with Bernoulli Diffusion
class BDDPM(pl.LightningModule):
    def __init__(self, 
                 unet_config,
                 aux=0,
                 gamma=1.,
                 timesteps=100,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=5,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="zt+z0",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 device=None,
                 ):
        super().__init__()
        #assert parameterization in ['prob', 'z0', 'zt+z0']
        assert parameterization in ['zt+z0']
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.device = device

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        # hyperparameter for balancing the sum of the loss terms
        self.aux = aux
        # hyperparameter for balancing the residual loss
        self.gamma = gamma


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        bs = [0.5 * betas[0]]
        for i in range(1, len(betas)):
            bs.append(alphas[i] * bs[i - 1] + 0.5 * betas[i])
        bs = np.array(bs)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        assert bs.shape[0] == self.num_timesteps, 'bs have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('bs', to_torch(bs))

        #if self.parameterization == "prob":
            # create a torch tensor of ones with the same shape as self.betas
        lvlb_weights = torch.ones_like(self.betas + 1, dtype=torch.float32)
        #elif self.parameterization == "x0":
        #    lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        #else:
        #    raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = -1.
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")


    def init_from_ckpt(self, ckpt_path, ignore_keys=[], only_model=False):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if only_model:
            ckpt = ckpt["state_dict"]
        if exists(ignore_keys):
            for k in ignore_keys:
                del ckpt[k]
        missing, unexpected = self.load_state_dict(ckpt, strict=False) if not only_model else self.model.load_state_dict(ckpt, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def sample_time(self, batch_size, device=None):
        """
        Sample a time step from the diffusion process.
        :param batch_size: the number of samples to generate.
        :param device: the torch device to use.
        :return: A tensor of shape [batch_size] of sampled time steps.
        """
        if device is None:
            device = self.device
        return torch.randint(1, self.num_timesteps, (batch_size,), device=device).long()


    def q_post(self, x_start, t):
        """
        Get the posterior probability of the Bernoulli distribution q(x_t | x_0, x_T).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A probability tensor of x_start's shape.
        """
        assert t >= 0 and t < self.num_timesteps
        # torch tensor of shape as x_start but with all elements being self.alphas_cumprod[t]
        kt = torch.full_like(x_start, self.alphas_cumprod[t])
        # torch tensor of shape as x_start but with all elements being self.bs[t]
        bt = torch.full_like(x_start, self.bs[t])
        return kt * x_start + bt
    

    def q_sample(self, x_start, t):
        """
        Sample from the posterior distribution q(x_t | x_0, x_T).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A sample tensor of x_start's shape.
        """
        return torch.bernoulli(self.q_post(x_start, t))


    def apply_model(self, x_start, t, cond=None):
        """
        Apply the model to a noiseless input x_start.
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param cond: the [N x C x ...] tensor of conditioning inputs.
        :return: A tensor of x_start's shape.
        """
        #TODO add conditioning
        return self.model(x_start, t)
    

    def p_losses(self, x_start, t, cond=None):
        """
        Compute the loss of the model.
        """
        # Sample x_t
        x_t = self.q_sample(x_start, t)
        # Predict using the model
        pred = self.apply_model(x_t, t, cond)
        # Compute the loss
        loss_dict = dict()
        #if self.parameterization == "prob":
        #    target = 
        #elif self.parameterization == "z0":
        #    target =
        if self.parameterization == "zt+z0":
            # Ensure pred is in [0, 1]
            p_flip = torch.sigmoid(pred)
            # Compute zt xor z0
            xor = torch.logical_xor(x_start, x_t) * 1.0
            # Compute BCE loss
            bce_loss = nn.functional.binary_cross_entropy_with_logits(p_flip, xor, reduction="none")
            loss_dict["bce_loss"] = bce_loss.mean()
            # Compute L_residual loss
            p_residual = p_flip * xor + (1 - p_flip) * (1 - xor) # probability for computing expectation of the BCE loss
            L_residual = bce_loss * (p_residual ** self.gamma)
            L_residual = L_residual.mean()
            loss_dict["L_residual"] = L_residual
            # Compute predicted x_start, it is a samplig of p_0
            p_0 = (1 - x_t) * p_flip + x_t * (1 - p_flip)
            p_0 = torch.clamp(p_0, 1e-7, 1 - 1e-7)
            x_start_pred = torch.bernoulli(p_0)
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        # Compute accuracy
        acc = (x_start_pred == x_start).float().mean()
        loss_dict["acc"] = acc
        
        # Compute L_vlb loss
        if self.aux > 0:
            ftr = (((t-1) == 0) * 1.0).view(-1, 1, 1)

            x_0_logits = torch.cat([x_start_pred.unsqueeze(-1), (1-x_start_pred).unsqueeze(-1)], dim=-1)
            x_t_logits = torch.cat([x_t.unsqueeze(-1), (1-x_t).unsqueeze(-1)], dim=-1)

            p_EV_qxtmin_x0 = self.scheduler(x_0_logits, t-1)

            q_one_step = self.scheduler.one_step(x_t_logits, t)
            unnormed_probs = p_EV_qxtmin_x0 * q_one_step
            unnormed_probs = unnormed_probs / (unnormed_probs.sum(-1, keepdims=True)+1e-6)
            unnormed_probs = unnormed_probs[...,0]
            
            x_tm1_logits = unnormed_probs * (1-ftr) + x_start * ftr
            x_0_gt = torch.cat([x_start.unsqueeze(-1), (1-x_start).unsqueeze(-1)], dim=-1)
            p_EV_qxtmin_x0_gt = self.scheduler(x_0_gt, t-1)
            unnormed_gt = p_EV_qxtmin_x0_gt * q_one_step
            unnormed_gt = unnormed_gt / (unnormed_gt.sum(-1, keepdims=True)+1e-6)
            unnormed_gt = unnormed_gt[...,0]

            x_tm1_gt = unnormed_gt

            if torch.isinf(x_tm1_logits).max() or torch.isnan(x_tm1_logits).max():
                pdb.set_trace()
            aux_loss = nn.functional.binary_cross_entropy(x_tm1_logits.clamp(min=1e-6, max=(1.0-1e-6)), x_tm1_gt.clamp(min=0.0, max=1.0), reduction='none')

            L_vlb = aux_loss.mean()
            L_total = self.aux * L_vlb + L_residual

            loss_dict["L_vlb"] = L_vlb
            loss_dict["L_total"] = L_total

            return L_total, loss_dict
        
        else:
            return L_residual, loss_dict


    def forward(self, x, *args, **kwargs):
        t = self.sample_time(x.shape[0], device=self.device)
        return self.p_losses(x, t, *args, **kwargs)


    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x


    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict
    

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)


    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    


class BinaryLatentDiffusion(BDDPM):
    """
    Main class
    """
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="caption",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = num_timesteps_cond
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.restarted_from_ckpt = True


    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids


    def register_schedule(self, 
                          given_betas=None, 
                          beta_schedule="linear", 
                          timesteps=1000, 
                          linear_start=0.0001, 
                          linear_end=0.02, 
                          cosine_s=0.008):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()


    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False


    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage as cond stage")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print("Using unconditional model")
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_unconditional__"
            assert config != "__is_first_stage__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model


    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # Only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1.0, "Rather not use custom rescaling and std-rescaling simultaneously"
            # Set rescale weigth to 1./std of encodings
            print("Rescaling by std of encodings")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1./z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")


    def get_input(self, batch, k,
                  return_first_stage_outputs = False,
                  force_c_encode = False, 
                  cond_key = None, 
                  return_original_cond = False, 
                  bs = None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach

        #TODO: add conditioning
        if self.model.conditioning_key is not None:
            raise NotImplementedError("Conditioning not yet supported")
            pass
        else:
            c = None
            xc = None

        out = [z, c]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        
        return out


    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, BernoulliDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError("Only Bernoulli and Tensor encodings supported")
        return self.scale_factor * z


    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)
    

    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)
    

    def get_learned_conditioning(self, c):
        #TODO: add conditioning
        raise NotImplementedError("Conditioning not yet supported")
        return None
            

    def shared_step(self, batch):
        """
        Compute the loss of the model.
        """
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss


    def forward(self, x, c, *args, **kwargs):
        t = self.sample_time(x.shape[0], device=self.device)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc)
        return self.p_losses(x, t, c, *args, **kwargs)


    def sample(self, cond,
               batch_size=16,
               return_intermediates=False,
               x_T=None,
               verbose=True,
               timesteps=None,
               quantized_denoised=True,
               mask=None,
               x0=None,
               shape=None, 
               **kwargs):
        """
        Sample from the model.
        """
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            #TODO: add conditioning
            raise NotImplementedError("Conditioning not yet supported")
            pass
        return self.p_sample_loop(cond, shape, x_T, timesteps, quantized_denoised, mask, x0, return_intermediates, verbose, **kwargs)

    
    def p_sample_loop(self, cond, shape, x_T, timesteps, quantized_denoised, mask, x0, return_intermediates, verbose,
                      img_callback=None,
                      start_T=None,
                      log_every_t=None,
                      **kwargs):
        """
        Loop to sample from the model.
        """
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            # Tensor of shape as shape of 0.5s
            x_T = torch.full(shape, 0.5, device=device)
        else:
            img = x_T
        
        intermediates = [img]

        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3] # Spatial dimensions must match

        for i in iterator:
            ts = torch.full((b,), i, device=device).long()
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc)

            img = self.p_sample(img, ts, cond)
        
        

    


        





class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'adm', 'hybrid']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            return self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            assert exists(c_concat)
            xc = torch.cat([x] + c_concat, dim=1)
            return self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            assert exists(c_crossattn)
            cc = torch.cat(c_crossattn, 1)
            return self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "adm":
            assert exists(c_crossattn)
            cc = c_crossattn[0]
            return self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == "hybrid":
            assert exists(c_concat) and exists(c_crossattn)
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            return self.diffusion_model(xc, t, context=cc)
        else:
            raise NotImplementedError(f"Conditioning key {self.conditioning_key} not supported")
            return None