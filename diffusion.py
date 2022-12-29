"""
Diffusion model

https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils.py
https://github.com/rosinality/denoising-diffusion-pytorch/blob/master/diffusion.py
"""
from typing import Tuple, List, Union, Callable, Optional
import tensorflow as tf
import numpy as np


def noise_like(
    shape: Union[List, Tuple],
    noise_fn: Callable = tf.random_normal,
    repeat: bool = False,
    dtype: tf.DType = tf.float32
):
    # repeat_noise = lambda: tf.repeat(noise_fn(shape=(1, *shape[1:]), dtype=dtype), repeats=shape[0], axis=0)
    # noise = lambda: noise_fn(shape=shape, dtype=dtype)
    # return repeat_noise() if repeat else noise()
    return tf.cond(
        repeat,
        lambda: tf.repeat(
            noise_fn(shape=(1, *shape[1:]), dtype=dtype), repeats=shape[0], axis=0
        ),
        lambda: noise_fn(shape=shape, dtype=dtype)
    )


def warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract_value_at_t(alphas, t, input_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    bs, = t.shape
    assert input_shape[0] == bs
    alphas = tf.gather(alphas, t)
    assert alphas.shape == [bs]
    return tf.reshape(alphas, [bs] + ((len(input_shape) - 1) * [1]))


class GaussianDiffusion:
    
    def __init__(self, betas, dtype=tf.float32) -> None:
        betas = tf.cast(betas, tf.float64)  # computations here in float64 for accuracy
        self.num_timesteps = int(betas.shape[0])
        
        alphas = 1. - betas
        alphas_cumprod = tf.math.cumprod(alphas)  # alpha_bar
        # alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_prev = tf.concat([[1.0], alphas_cumprod[:-1]], axis=0)
        assert alphas_cumprod_prev.shape == (self.num_timesteps,)
        
        # casting
        self.betas = tf.cast(betas, dtype)
        self.alphas_cumprod = tf.cast(alphas_cumprod, dtype)
        self.alphas_cumprod_prev = tf.cast(alphas_cumprod_prev, dtype)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.math.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.math.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = tf.math.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.math.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.math.sqrt(1. / alphas_cumprod - 1.)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)  # beta_tilde
        self.posterior_log_variance_clipped = tf.cast(
            tf.math.maximum(self.posterior_variance, 1e-20),
            dtype
        )
        
        self.posterior_mean_coef1 = tf.cast(
            betas * tf.math.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
            dtype
        )
        self.posterior_mean_coef2 = tf.cast(
            (1. - alphas_cumprod_prev) * tf.math.sqrt(alphas) / (1. - alphas_cumprod)
        )
    
    def q_posterior(self, x_0, x_t, t):
        """
        Compute the mean and variance 
        of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract_value_at_t(self.posterior_mean_coef1, t, x_0.shape) * x_0 +
            extract_value_at_t(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_value_at_t(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = extract_value_at_t(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_0, t, noise=None):
        """compute `x_t(x_0, noise)`
        
        t시점에서의 noised x를 구하는 메소드
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = tf.random_normal(shape=x_0.shape)
        assert noise.shape == x_0.shape
        return (
            extract_value_at_t(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract_value_at_t(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        """q_sample과 반대로 계산
        
        q_sample: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        => x_0 = (1 / sqrt(alpha_bar_t)) * x_t - (sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t)) * noise
        """
        assert x_t.shape == noise.shape
        return (
            extract_value_at_t(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_value_at_t(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def p_mean_variance(self, x, t, denoise_fn, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, -1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        assert model_mean.shape == x_recon.shape == x.shape
        assert posterior_variance.shape == posterior_log_variance.shape == [x.shape[0], 1, 1, 1]
        
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(self, x, t, noise_fn, denoise_fn, clip_denoised: bool = True, repeat_noise: bool = False):
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, noise_fn, repeat_noise)
        assert noise.shape == x.shape
        # no noise when t == 0
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [x.shape[0]] + [1] * (len(x.shape) - 1))
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
    
    def p_sample_loop(self, shape, denoise_fn, noise_fn=tf.random_normal):
        """
        Generate samples
        """
        i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
        img_0 = noise_fn(shape=shape, dtype=tf.float32)
        _, img_final = tf.while_loop(
        cond=lambda i_, _: tf.greater_equal(i_, 0),
        body=lambda i_, img_: [
            i_ - 1,
            self.p_sample(denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), noise_fn=noise_fn)
        ],
        loop_vars=[i_0, img_0],
        shape_invariants=[i_0.shape, img_0.shape],
        back_prop=False
        )
        assert img_final.shape == shape
        return img_final
    
    def p_losses(self, model, x_0, t, noise=None):
        """Training loss calculation
        noised_x를 가지고 model에 넣어 추가된 noise를 예측

        Args:
            model (tf.keras.Model): denoising function(UNet) and returning noises
            x_0 (tf.Tensor): input image tensor
            t (tf.Tensor): timestep; (B,)
            noise (Callable, optional): function making noise. Defaults to None.

        Returns:
            tf.Tensor: L2 loss
        """
        B, H, W, C = x_0.shape.as_list()
        assert t.shape == [B]

        if noise is None:
            noise = tf.random_normal(shape=x_0.shape, dtype=x_0.dtype)
        assert noise.shape == x_0.shape and noise.dtype == x_0.dtype
        x_noisy = self.q_sample(x_start=x_0, t=t, noise=noise)

        # denoise-func returns noises to be subtracted
        # denoise-func is model(UNet)
        x_recon = model(x_noisy, t)  
        assert x_noisy.shape == x_0.shape
        assert x_recon.shape[:3] == [B, H, W] and len(x_recon.shape) == 4

        # predict the noise instead of x_start. seems to be weighted naturally like SNR
        assert x_recon.shape == x_0.shape
        losses = tf.squared_difference(noise, x_recon)
        losses = tf.math.reduce_mean(losses, axis=list(range(1, len(losses.shape))))

        assert losses.shape == [B]
        return losses
