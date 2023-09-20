import numpy as np
import torch
from torch import nn

class BernoulliDiffusion(nn.Module):
    """
    Utilities for training an sampling diffusion models based on a Bernoulli diffusion process.

    Inspired from 
    https://github.com/BarqueroGerman/BeLFusion/blob/main/models/diffusion/gaussian_diffusion.py#L190

    """

    def __init__(
            self, 
            noise_schedule,
            steps,
            predict="start_x",
            losses="mse",
            losses_multipliers=1.,
            rescale_timesteps=False,
            **kwargs
    ):
        # Use float64 for accuracy
        betas = get_named_beta_schedule(noise_schedule, steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        assert self.alphas_cumprod_next.shape == (self.num_timesteps,)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        gammas = [0.5*betas[0]]
        for t in range(self.num_timesteps-1):
            gammas.append(alphas[t+1]*gammas[t] + 0.5 * betas[t+1])
        self.gammas = np.asarray(gammas)
        assert self.gammas.shape == (self.num_timesteps,)
        
    def q_dist(
            self,
            x_start,
            t
    ):
        """
        Get the distribution:
            q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: the probability of x_start
        """
        pass

    def q_sample(
            self,
            x_start,
            t,
            noise=None
    ):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from:
            q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A noisy verion of x_start.
        """
        pass

    def q_posterior_dist(
            self,
            x_start,
            t
    ):
        """
        Compute the probability parameter of the Bernoulli diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        
        """
        prob = self.alphas_cumprod[t] * x_start + self.gammas[t]

        return prob

    def p_dist(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_Kargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps as input.
        :oaram x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'prob': the model probability output.
                 - 'pred_xstart': the prediction for x_0.
        """
        pass


    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_wargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        pass


    def training(
            self,
            dataset,
            encoder,
            diffusion_steps,
            noise_scheduler,
            epochs,
            lmbda=0.01,
    ):
        """
        Training procedure for the Bernoulli DDPM model.

        :param dataset: the raw images to train on.
        :param encoder: a pretrained model that encodes each image into a binary representation.
        :param diffusion_steps: the number of diffusion steps.
        :param noise_scheduler: 1-D numpy array with the beta value for each time step to regulate noise.
        :param epochs: number of epochs to train the model.
        """