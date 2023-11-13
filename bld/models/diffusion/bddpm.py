import os
import torch 
import torch.nn as nn
import torchvision
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from functools import partial
from contextlib import contextmanager
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torchvision import transforms
from pytorch_lightning.utilities.distributed import rank_zero_only
from PIL import Image

import pdb

from main import instantiate_from_config
from bld.utils import count_params, exists, log_txt_as_img, isimage, ismap
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
                 alpha=0.5,
                 timesteps=100,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=False,
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
                 parameterization="p_flip",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 device=None,
                 ):
        super().__init__()
        #assert parameterization in ['prob', 'z0', 'zt+z0']
        assert parameterization in ['p_flip']
        self.parameterization = parameterization
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
        # Make sure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
        self.total_params = count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            raise NotImplementedError("EMA not yet supported")
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        #self.device = device

        if monitor is not None:
            self.monitor = monitor
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        #self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
        #                       linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps)

        self.loss_type = loss_type

        # hyperparameter for balancing the sum of the loss terms
        self.aux = aux
        # hyperparameter for balancing the residual loss
        self.gamma = gamma
        # hyperparameter for balancing the residual loss
        self.alpha = alpha


    def register_schedule_2(self, beta_schedule, timesteps):
        if beta_schedule == 'linear':

            beta = 1 - 1 / (timesteps - np.arange(1, timesteps+1) + 1) 

            k_final = [1.0]
            b_final = [0.0]

            for i in range(timesteps):
                k_final.append(k_final[-1]*beta[i])
                b_final.append(beta[i] * b_final[-1] + 0.5 * (1-beta[i]))

            k_final = k_final[1:]
            b_final = b_final[1:]


        elif beta_schedule == 'cos':

            k_final = np.linspace(0.0, 1.0, timesteps+1)

            k_final = k_final * np.pi
            k_final = 0.5 + 0.5 * np.cos(k_final)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(timesteps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = np.array(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]
        
        elif beta_schedule == 'sigmoid':
            
            def sigmoid(x):
                z = 1/(1 + np.exp(-x))
                return z

            def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=0.0):
                # A gamma function based on sigmoid function.
                v_start = sigmoid(start / tau)
                v_end = sigmoid(end / tau)
                output = sigmoid((t * (end - start) + start) / tau)
                output = (v_end - output) / (v_end - v_start)
                return np.clip(output, clip_min, 1.)
            
            k_final = np.linspace(0.0, 1.0, timesteps+1)
            k_final = sigmoid_schedule(k_final, 0, 3, 0.8)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(timesteps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = np.array(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]


        else:
            raise NotImplementedError
        
        k_final = np.hstack([1, k_final])
        b_final = np.hstack([0, b_final])
        beta = np.hstack([1, beta])
        self.register_buffer('k_final', torch.Tensor(k_final))
        self.register_buffer('b_final', torch.Tensor(b_final))
        self.register_buffer('betas', torch.Tensor(beta))  
        self.register_buffer('cumbeta', torch.cumprod(self.betas, 0))  
        self.num_timesteps = int(timesteps)
        # pdb.set_trace()

        print(f'Noise scheduler with {beta_schedule}:')

        print(f'Diffusion 1.0 -> 0.5:')
        data = (1.0 * self.k_final + self.b_final).data.numpy()
        print(' '.join([f'{d:0.4f}' for d in data]))

        print(f'Diffusion 0.0 -> 0.5:')
        data = (0.0 * self.k_final + self.b_final).data.numpy()
        print(' '.join([f'{d:0.4f}' for d in data]))

        print(f'Beta:')
        print(' '.join([f'{d:0.4f}' for d in self.betas.data.numpy()]))



    def register_schedule(self, beta_schedule="linear", timesteps=100,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, given_betas=None):
        """
        Deprecated
        """
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(schedule=beta_schedule, n_timestep=timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        bs = [0.5 * betas[0]]
        for i in range(1, len(betas)):
            bs.append(alphas[i] * bs[i - 1] + 0.5 * betas[i])
        bs = np.array(bs)

        beta_timesteps, = betas.shape
        assert beta_timesteps==timesteps, f"beta_timesteps ({beta_timesteps}) != timesteps ({timesteps})"
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


    # @contextmanager
    # def ema_scope(self, context=None):
    #     if self.use_ema:
    #         self.model_ema.store(self.model.parameters())
    #         self.model_ema.copy_to(self.model)
    #         if context is not None:
    #             print(f"{context}: Switched to EMA weights")
    #     try:
    #         yield None
    #     finally:
    #         if self.use_ema:
    #             self.model_ema.restore(self.model.parameters())
    #             if context is not None:
    #                 print(f"{context}: Restored training weights")


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


    def sample_time(self, batch_size, device):
        """
        Sample a time step from the diffusion process.
        :param batch_size: the number of samples to generate.
        :param device: the torch device to use.
        :return: A tensor of shape [batch_size] of sampled time steps.
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()


    def q_post_2(self, x_start, t):
        """
        Get the posterior probability of the Bernoulli distribution q(x_t | x_0, x_T).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A probability tensor of x_start's shape.
        """
        dim = x_start.ndim - 1
        kt = self.k_final[t].view(-1, *([1]*dim))
        bt = self.b_final[t].view(-1, *([1]*dim))
        return kt * x_start + bt

    def q_post(self, x_start, t):
        """
        Deprecated
        """
        # torch tensor of shape as x_start but with all elements being self.alphas_cumprod[t]
        # kt = torch.full_like(x_start, self.alphas_cumprod[t].cpu().numpy()[0])
        # torch tensor of shape as x_start but with all elements being self.bs[t]
        # bt = torch.full_like(x_start, self.bs[t].cpu().numpy()[0])
        #kt = self.alphas_cumprod[t]
        #bt = torch.full(x_start.shape, self.bs[t])
        bs = []
        ks = []
        sh = x_start[0].shape
        for b, k in zip(self.bs[t], self.alphas_cumprod[t]):
            b_t = torch.full(sh, b)
            k_t = torch.full(sh, k)
            bs.append(b_t)
            ks.append(k_t)
        bt = torch.stack(bs).to(x_start.device)
        kt = torch.stack(ks).to(x_start.device)
        # bt = self.bs[t]
        return kt * x_start + bt
    

    def q_sample(self, x_start, t, return_logits=False):
        """
        Sample from the posterior distribution q(x_t | x_0, x_T).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A sample tensor of x_start's shape.
        """
        logits = self.q_post(x_start, t)
        sample = torch.bernoulli(logits)
        if return_logits:
            return sample, logits
        return sample


    def one_step(self, x, t):
        # dim = x.ndim - 1
        # beta = self.betas[t].view(-1, *([1]*dim))
        # print(beta)
        #print("beta: ", beta.shape)
        # x = x * (1 - beta) + 0.5 * beta
        # print("x: ", x.shape)
        # print("t: ", t)
        # print("self.betas[t]: ", self.betas[t])
        # print("self.betas[t]: ", self.betas[t].shape)
        # print("self.betas: ", self.betas)
        # #beta = self.betas[t]
        # Multiply by beta of size [4] with x of size [4, 3, 256, 256]
        #beta = beta.view(-1, *([1]*3))
        betass = []
        sh = x[0].shape
        for beta in self.betas[t]:
            beta_t = torch.full(sh, beta)
            betass.append(beta_t)
        betas_ = torch.stack(betass).to(x.device)
        # print(betas_.shape)
        # print("beta: ", betas_.shape)
        x = x * (1 - betas_) + 0.5 * betas_
        # print("x: ", x.shape)
        #x = x * (1 - self.betas[t]) + 0.5 * self.betas[t]
        return x

    
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
        log_prefix = 'train' if self.training else 'val'
        # Sample x_t
        x_t, logits = self.q_sample(x_start, t, return_logits=True)
        # print("x_t: ", x_t.shape)
        # print("x_t: ", x_t)
        # print("logits: ", logits)
        # Predict using the model
        pred = self.apply_model(x_t, t-1, cond)
        #print("pred: ", pred.shape)
        # Compute the loss
        loss_dict = dict()
        #if self.parameterization == "prob":
        #    target = 
        #elif self.parameterization == "z0":
        #    target =
        if self.parameterization == "p_flip":
            # Ensure pred is in [0, 1]
            p_flip = torch.sigmoid(pred)
            #print("p_flip: ", p_flip.shape)

            # Compute zt xor z0
            xor = torch.logical_xor(x_start, x_t) * 1.0
            # print("xor: ", xor.shape)
            # print("pred: ", pred)
            # print("xor: ", xor)
            # print("t", t)
            # Compute BCE loss (use pred instead of p_flip to avoid numerical instabilities)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, xor, reduction="none")
            # Compute MSE loss
            mse_loss = nn.functional.mse_loss(pred, xor, reduction="none")
            #print("bce_loss: ", bce_loss.shape)
            #print("bce_loss: ", bce_loss.mean().shape)
            loss_dict.update({f'{log_prefix}/bce_loss': bce_loss.mean()})
            loss_dict.update({f'{log_prefix}/mse_loss': mse_loss.mean()})
            # Compute L_residual loss
            p_residual = p_flip * xor + (1 - p_flip) * (1 - xor) # probability for computing expectation of the BCE loss
            p_residual = 1 - p_residual
            #print("p_residual: ", p_residual.shape)
            L_residual = bce_loss * (p_residual ** self.gamma)
            if self.alpha == -1:
                neg_weight = xor.sum((-1,-2))
                neg_weight = neg_weight / xor[0].numel()
                neg_weight = neg_weight.view(-1, 1, 1)
                alpha_t = (1 - neg_weight) * xor + neg_weight * (1 - xor)
                L_residual = alpha_t * L_residual
            elif self.alpha > 0:
                alpha_t = self.alpha * xor + (1 - self.alpha) * (1 - xor)
                L_residual = alpha_t * L_residual
            #print("L_residual: ", L_residual.shape)
            L_residual = L_residual.mean()
            #print("L_residual: ", L_residual)
            loss_dict.update({f'{log_prefix}/L_residual': L_residual})
            # Compute predicted x_start, it is a samplig of p_0
            p_0 = (1 - x_t) * p_flip + x_t * (1 - p_flip)
            #print("p_0: ", p_0.shape)
            #p_0 = torch.clamp(p_0, 1e-7, 1 - 1e-7)
            #print("p_0: ", p_0.shape)
            x_start_pred = torch.bernoulli(p_0)
            #print("x_start_pred: ", x_start_pred.shape)

            L_total = L_residual #+ mse_loss.mean() * 0.1
            #L_total = mse_loss.mean()
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        # Compute accuracy
        acc = (x_start_pred == x_start).float().mean()
        #print("acc: ", acc.shape)
        loss_dict["acc"] = acc
        
        # Compute L_vlb loss
        if self.aux > 0:
            ftr = (((t-1) == 0) * 1.0).view(-1, 1, 1, 1)
            #print("ftr: ", ftr.shape)

            x_0_logits = torch.cat([x_start_pred.unsqueeze(-1), (1-x_start_pred).unsqueeze(-1)], dim=-1)
            #print("x_0_logits: ", x_0_logits.shape)
            x_t_logits = torch.cat([x_t.unsqueeze(-1), (1-x_t).unsqueeze(-1)], dim=-1)
            #print("x_t_logits: ", x_t_logits.shape)

            p_EV_qxtmin_x0 = self.q_post(x_0_logits, t-1)
            #print("p_EV_qxtmin_x0: ", p_EV_qxtmin_x0.shape)

            q_one_step = self.one_step(x_t_logits, t)
            #print("q_one_step: ", q_one_step.shape)
            unnormed_probs = p_EV_qxtmin_x0 * q_one_step
            #print("unnormed_probs: ", unnormed_probs.shape)
            unnormed_probs = unnormed_probs / (unnormed_probs.sum(-1, keepdims=True)+1e-6)
            #print("unnormed_probs: ", unnormed_probs.shape)
            unnormed_probs = unnormed_probs[...,0]
            #print("unnormed_probs: ", unnormed_probs.shape)
            
            x_tm1_logits = unnormed_probs * (1-ftr) + x_start * ftr
            #print("x_tm1_logits: ", x_tm1_logits.shape)

            x_0_gt = torch.cat([x_start.unsqueeze(-1), (1-x_start).unsqueeze(-1)], dim=-1)
            #print("x_0_gt: ", x_0_gt.shape)
            p_EV_qxtmin_x0_gt = self.q_post(x_0_gt, t-1)
            #print("p_EV_qxtmin_x0_gt: ", p_EV_qxtmin_x0_gt.shape)
            unnormed_gt = p_EV_qxtmin_x0_gt * q_one_step
            #print("unnormed_gt: ", unnormed_gt.shape)
            unnormed_gt = unnormed_gt / (unnormed_gt.sum(-1, keepdims=True)+1e-6)
            #print("unnormed_gt: ", unnormed_gt.shape)
            unnormed_gt = unnormed_gt[...,0]
            #print("unnormed_gt: ", unnormed_gt.shape)

            x_tm1_gt = unnormed_gt
            #print("x_tm1_gt: ", x_tm1_gt.shape)

            if torch.isinf(x_tm1_logits).max() or torch.isnan(x_tm1_logits).max():
                pdb.set_trace()
            aux_loss = nn.functional.binary_cross_entropy(x_tm1_logits.clamp(min=1e-6, max=(1.0-1e-6)), x_tm1_gt.clamp(min=0.0, max=1.0), reduction='none')
            #print("aux_loss: ", aux_loss.shape)
            L_vlb = aux_loss.mean()
            #print("L_vlb: ", L_vlb.shape)
            L_total = self.aux * L_vlb + L_residual #+ mse_loss.mean() * 0.1
            #print("L_total: ", L_total.shape)
            
            loss_dict.update({f'{log_prefix}/L_vlb': L_vlb})

        loss_dict.update({f'{log_prefix}/L_total': L_total})

        acc = (x_start_pred == x_start).float().mean()
        image = self.decode_first_stage(x_start_pred)
        #print("acc: ", acc)
        grid = torchvision.utils.make_grid(image, nrow=4)
        grid = (grid + 1.0) / 2.0
        grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
        grid = grid.cpu().detach().numpy()
        grid = (grid * 255).astype(np.uint8)
        # make sure values are in range 0-255
        grid = np.clip(grid, 0, 255)
        filename = "image4.png"
        path = os.path.join("/home/apujol/binary-latent-diffusion", filename)
        Image.fromarray(grid).save(path)
        
        return L_total, loss_dict


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
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    # def on_train_batch_end(self, *args, **kwargs):
    #     if self.use_ema:
    #         self.model_ema(self.model)


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
                 num_sample_steps=None,
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
        self.first_stage_config = first_stage_config
        try:
            self.num_downs = len(self.first_stage_config.params.ddconfig.ch_mult) - 1
            latent_res = int(self.first_stage_config.params.ddconfig.resolution / (2**self.num_downs))
            self.latent_shape = tuple([self.first_stage_config.params.ddconfig.z_channels, latent_res, latent_res])
        except:
            self.num_downs = 0
            self.latent_shape = None
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.num_sample_steps = num_sample_steps
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
        #super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        super().register_schedule(beta_schedule, timesteps)
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
        x = x.to(self.device)
        #encoder_posterior = self.encode_first_stage(x)
        #z = self.get_first_stage_encoding(encoder_posterior).detach()
        z = self.encode_first_stage(x).detach()

        #TODO: add conditioning
        if self.model.conditioning_key is not None:
            print("Conditioning key: ", self.model.conditioning_key)
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
        loss, loss_dict = self(x, c)
        return loss, loss_dict


    def forward(self, x, c, *args, **kwargs):
        t = self.sample_time(x.shape[0], device=self.device)
        #print("t: ", t)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc)
        return self.p_losses(x, t, c, *args, **kwargs)

    @torch.no_grad()
    def sample(self,
               batch_size=16,
               z_T=None,
               shape=None, 
               return_intermediates=False,
               temp=1.0,
               log_every_t=None,
               cond=None,
               verbose=True,
               **kwargs):
        """
        Sample from the model.
        """
        if shape is None:
            if self.latent_shape is not None:
                # add batch_size to the tuple self.latent_shape
                shape = (batch_size, *self.latent_shape)
            else:
                shape = (batch_size, self.channels, self.image_size, self.image_size)

        if cond is not None:
            #TODO: add conditioning
            raise NotImplementedError("Conditioning not yet supported")
            pass
        
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device

        if z_T is None:
            # Tensor of shape as shape of 0.5s
            p_T = torch.full(shape, 0.5, device=device)
            z_T = torch.bernoulli(p_T)
        else:
            z_T = torch.bernoulli(z_T).to(device)

        intermediates = [z_T]

        if self.num_sample_steps is None:
            self.num_sample_steps = self.num_timesteps

        sampling_steps = np.array(range(1, self.num_timesteps+1))

        if self.num_sample_steps != self.num_timesteps:
            idx = np.linspace(0.0, 1.0, self.num_sample_steps)
            idx = np.array(idx * (self.num_timesteps-1), int)
            sampling_steps = sampling_steps[idx]

        iterator = tqdm(reversed(range(1, self.num_sample_steps)), desc='Sampling t', total=self.num_sample_steps) if verbose else reversed(range(self.num_sample_steps))

        z_t = z_T
        for i in iterator:
            ts = torch.full((batch_size,), sampling_steps[i], device=device, dtype=torch.long)
            #print("ts: ", ts)
            pred = self.apply_model(z_t, ts, cond)
            #print("pred: ", pred)
            #print("pred: ", pred.shape)
            p_flip = torch.sigmoid(pred)
            #print("p_flip: ", p_flip)
            #print("p_flip: ", p_flip.shape)
            # Control with temperature
            p_flip = p_flip / temp
            #print("p_flip: ", p_flip.shape)
            # Compute p_0 given p_flip and z_t
            p_0 = (1 - z_t) * p_flip + z_t * (1 - p_flip)
            #print("p_0: ", p_0.shape)   
            z_0 = torch.bernoulli(p_0)
            #print("z_0: ", z_0.shape)
            
            #print("sampling_steps[i]: ", sampling_steps[i])

            # Compute z_t-1
            if not sampling_steps[i] == 1:
                t_p = torch.full((batch_size,), sampling_steps[i-1], device=device, dtype=torch.long)
                z_0_logits = torch.cat([p_0.unsqueeze(-1), (1-p_0).unsqueeze(-1)], dim=-1)
                z_t_logits = torch.cat([z_t.unsqueeze(-1), (1-z_t).unsqueeze(-1)], dim=-1)
                # print("z_0_logits: ", z_0_logits.shape)
                # print("z_t_logits: ", z_t_logits.shape)
                # print("t_p: ", t_p.shape)
                # print("t_p: ", t_p)
                p_EV_qztmin_z0 = self.q_post(z_0_logits, t_p)
                # print("p_EV_qztmin_z0: ", p_EV_qztmin_z0.shape)
                q_one_step = z_t_logits

                for mns in range(1, sampling_steps[i] - sampling_steps[i-1]):
                    #print("mns: ", mns)
                    #print("sampling_steps[i] - mns: ", sampling_steps[i] - mns)
                    t_s = torch.full((batch_size,), sampling_steps[i] - mns, device=device, dtype=torch.long)
                    q_one_step = self.one_step(q_one_step, t_s)

                unnormed_probs = p_EV_qztmin_z0 * q_one_step
                unnormed_probs = unnormed_probs / unnormed_probs.sum(-1, keepdims=True)
                unnormed_probs = unnormed_probs[...,0]
                
                z_tm1_logits = unnormed_probs
                z_tm1 = torch.bernoulli(z_tm1_logits)

                aux = z_t - z_tm1
                #print("aux: ", aux)
                #print("z_t: ", z_t)
            else: 
                z_tm1 = ()
            
            z_t = z_tm1
            #img = self.decode_first_stage(z_t)

            if i % log_every_t == 0 or i == self.num_sample_steps - 1:
                #intermediates.append(img)
                #intermediates.append(z_0)
                intermediates.append(z_0)
                
        if return_intermediates:
            #return img, intermediates
            return z_t, intermediates 
        else:
            #return img
            return z_t
    
    @torch.no_grad()
    def sample_log(self,cond,batch_size, ddim=False, ddim_steps=None,**kwargs):

        if ddim:
            #TODO: implement ddim
            #ddim_sampler = DDIMSampler(self)
            #shape = (self.channels, self.image_size, self.image_size)
            #samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
            #                                            shape,cond,verbose=False,**kwargs)
            raise NotImplementedError("DDIM not yet supported")
            pass
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=8,
                   n_row=4, 
                   sample=True, 
                   ddim_steps=None, 
                   ddim_eta=1., 
                   return_keys=None,
                   latent_images=False, 
                   inpaint=False, 
                   plot_denoise_rows=True, 
                   plot_progressive_rows=False,
                   plot_diffusion_rows=False, 
                   **kwargs):
        """
        
        """
        use_ddim = ddim_steps is not None

        log = dict()

        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)

        # print("z: ", z)
        # print("c: ", c)
        # print("x: ", x)
        # print("xrec: ", xrec)
        # print("xc: ", xc)

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        #log["inputs"] = x
        #log["reconstruction"] = xrec

        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            percentage = list()
            logits_list = list()
            acc = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    z_noisy, logits = self.q_sample(x_start=z_start, t=t, return_logits=True)
                    percentage.append(z_noisy.mean().item())
                    logits_list.append(logits.mean().item())
                    diffusion_row.append(self.decode_first_stage(z_noisy))
                
                    acc.append((z_noisy == z_start).float().mean())

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid
            # print("")
            # print("Diffusion row")
            # print("Percentage: ", percentage)
            # print("")
            # print("Logits: ", logits_list)
            # print("")
            # print("Last logits: ", logits)
            # print("")
            # print("Last z_noisy: ", z_noisy)
            # print("")
            # print("Last z_start: ", z_start)
            # print("")
            # print("Acc: ", acc)
            # print("")


        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta)
            # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            # print("samples: ", samples.shape)
            # print("samples: ", samples)
            # print("z_denoise_row: ", len(z_denoise_row))
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if plot_denoise_rows:
                #raise NotImplementedError("Denoise rows not yet supported")
                denoise_grid = self._get_denoise_row_from_list(samples=z_denoise_row)
                log["denoise_row"] = denoise_grid

            if latent_images:
                # plot the latent images
                raise NotImplementedError("Latent images not yet supported")

            if inpaint:
                raise NotImplementedError("Inpainting not yet supported")
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            raise NotImplementedError("Progressive rows not yet supported")
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    
    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            #print("zd: ", zd.shape)
            denoise_row.append(self.decode_first_stage(zd))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
    

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt





class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'adm', 'hybrid']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            return self.diffusion_model(x, t)
        #elif self.conditioning_key == "concat":
        #    assert exists(c_concat)
        #    xc = torch.cat([x] + c_concat, dim=1)
        #    return self.diffusion_model(xc, t)
        #elif self.conditioning_key == "crossattn":
        #    assert exists(c_crossattn)
        #    cc = torch.cat(c_crossattn, 1)
        #    return self.diffusion_model(x, t, context=cc)
        #elif self.conditioning_key == "adm":
        #    assert exists(c_crossattn)
        #    cc = c_crossattn[0]
        #    return self.diffusion_model(x, t, y=cc)
        #elif self.conditioning_key == "hybrid":
        #    assert exists(c_concat) and exists(c_crossattn)
        #    xc = torch.cat([x] + c_concat, dim=1)
        #    cc = torch.cat(c_crossattn, 1)
        #    return self.diffusion_model(xc, t, context=cc)
        else:
            raise NotImplementedError(f"Conditioning key {self.conditioning_key} not supported")
            return None