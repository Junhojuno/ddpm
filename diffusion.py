"""
Diffusion model

"""
import tensorflow as tf
import numpy as np


class GaussianDiffusion:
    
    def __init__(self, betas, loss_type, dtype=tf.float32) -> None:
        self.loss_type = loss_type
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
    
    def _extract_t_value(self, alphas, t, input_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert input_shape[0] == bs
        alphas = tf.gather(alphas, t)
        assert alphas.shape == [bs]
        return tf.reshape(alphas, [bs] + ((len(input_shape) - 1) * [1]))
    
    def q_posterior(self, x_0, x_t, t):
        """
        Compute the mean and variance 
        of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            self._extract_t_value(self.posterior_mean_coef1, t, x_0.shape) * x_0 +
            self._extract_t_value(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_t_value(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = self._extract_t_value(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_mean_variance(self, x_0, t):
        pass
    
    def q_sample(self, x_0, t, noise=None):
        pass
    
    def predict_start_from_noise(self, x_t, t, noise):
        pass
    
