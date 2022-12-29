import tensorflow as tf

from model import UNet
from diffusion import GaussianDiffusion, get_beta_schedule


def train(
    data_loader: tf.data.Dataset,
    model: tf.keras.Model,
    diffusion: GaussianDiffusion,
    n_epochs: int,
    num_timesteps: int = 1000
):
    for epoch in range(n_epochs):
        for image in data_loader:
            t = tf.random_uniform([image.shape[0]], 0, num_timesteps, dtype=tf.int32)
            loss = diffusion.p_losses(model, x_0=image, t=t)


def main():
    input_shape = [256, 256, 3]
    channels = 128
    channel_multiplier = [1, 1, 2, 2, 4, 4]
    num_res_blocks = 2
    attn_resolutions = (16,)
    dropout = 0.0
    
    model = UNet(
        input_shape,
        channels,
        ch_mult=channel_multiplier,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
        name='denoising_unet'
    )
    
    betas = get_beta_schedule('linear', 0.0001, 0.02, 1000)
    diffusion = GaussianDiffusion(betas)
    


if __name__ == '__main__':
    main()
