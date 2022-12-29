"""
UNet
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
from typing import Tuple, Union, Optional, List
import tensorflow as tf
import tensorflow_addons as tfa


def swish(x):
    return tf.keras.activations.swish(x)


def group_norm(x, name='group_norm'):
    return tfa.layers.GroupNormalization(32, name=name)(x)


def default_initializer(scale=1.):
    scale = 1e-10 if scale == 0 else scale
    return tf.keras.initializers.VarianceScaling(scale, mode='fan_avg', distribution='uniform')


def Conv2D(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = 'same',
    init_scale: float = 1.0,
    use_bias: bool = True,
    name: str = 'conv'
):
    return tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides,
        padding,
        kernel_initializer=default_initializer(init_scale),
        bias_initializer=tf.constant_initializer(0.),
        use_bias=use_bias,
        name=name
    )


def Dense(units, init_scale: float = 1.0, use_bias: bool = True, name: str = 'dense'):
    return tf.keras.layers.Dense(
        units,
        kernel_initializer=default_initializer(init_scale),
        bias_initializer=tf.constant_initializer(0.),
        use_bias=use_bias,
        name=name
    )


def get_timestep_embedding(timesteps, embedding_dim: int, default_dtype=tf.float32):
    half_dim = embedding_dim // 2
    emb = tf.math.log(10000.) / (half_dim - 1)
    emb = tf.math.exp(tf.range(half_dim, dtype=default_dtype) * -emb)
    emb = tf.cast(timesteps, dtype=default_dtype)[:, None] * emb[None, :]
    emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == [timesteps.shape[0], embedding_dim]
    return emb


def res_block(
    inputs,
    time_embedding,
    out_ch: Optional[int] = None,
    conv_shortcut: bool = False,
    dropout: float = 0.,
    name='res_block'
):
    """residual block"""
    prefix = name
    B, _, _, C = inputs.shape
    if out_ch is None:
        out_ch = C
    
    x = inputs
    x = group_norm(x, name=f'{prefix}/norm1')
    x = swish(x)
    x = Conv2D(out_ch, 3, 1, 'same', name=f'{prefix}/conv1')(x)
    
    # add timestep embedding
    x += Dense(out_ch, name=f'{prefix}/t_emb_proj')(swish(time_embedding))[:, None, None, :]
    
    x = group_norm(x, name=f'{prefix}/norm2')
    x = swish(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = Conv2D(out_ch, 3, 1, 'same', init_scale=0., name=f'{prefix}/conv2')(x)
    
    if C != out_ch:
        if conv_shortcut:
            inputs = Conv2D(out_ch, 3, 1, 'same', name=f'{prefix}/conv_shortcut')(inputs)
        else: # use dense layer instead of nin
            inputs = Dense(out_ch, name=f'{prefix}/nin_shortcut')(inputs)
    assert inputs.shape == x.shape, f'shortcut connection is unavailable, inputs={inputs.shape}, x={x.shape}'
    # return inputs + x
    return tf.keras.layers.Add()([inputs, x])


def attention_block(inputs, name='att_block'):
    _, H, W, C = inputs.shape
    prefix = name
    h = group_norm(inputs, name=f'{prefix}/norm')
    q = Dense(C, name=f'{prefix}/query')(h)
    k = Dense(C, name=f'{prefix}/key')(h)
    v = Dense(C, name=f'{prefix}/value')(h)
    
    w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = tf.reshape(w, [-1, H, W, H * W])
    w = tf.nn.softmax(w, -1)
    w = tf.reshape(w, [-1, H, W, H, W])
    
    h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
    h = Dense(C, init_scale=0.0, name=f'{prefix}/proj_out')(h)
    
    assert h.shape == inputs.shape
    # return inputs + h
    return tf.keras.layers.Add()([inputs, h])


def downsample(inputs, with_conv: bool = True, name: str = 'downsample'):
    prefix = name
    B, H, W, C = inputs.shape
    if with_conv:
        x = Conv2D(C, 3, 2, 'same', name=f'{prefix}/conv_s2')(inputs)
    else:
        x = tf.keras.layers.AveragePooling2D(2, 2, 'same', name=f'{prefix}/avg_pool')(inputs)
    assert x.shape == [B, H // 2, W // 2, C]
    return x


def upsample(inputs, with_conv: bool = True, name: str = 'upsample'):
    prefix = name
    B, H, W, C = inputs.shape
    x = tf.image.resize(inputs, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    assert x.shape == [B, H * 2, W * 2, C]
    if with_conv:
        x = Conv2D(C, 3, 1, 'same', name=f'{prefix}/conv')(x)
        assert x.shape == [B, H * 2, W * 2, C]
    return x


def UNet(
    input_shape,
    ch: int = 128,
    out_ch: int = 3,
    ch_mult: Union[List, Tuple]=(1, 2, 4, 8),
    num_res_blocks: int = 2,
    attn_resolutions: Tuple = (16,),
    dropout: float = 0.,
    resamp_with_conv: bool = True,
    name: str = 'unet'
):
    num_resolutions = len(ch_mult)
    inputs = tf.keras.Input(input_shape)
    timestep = tf.keras.Input([])
    
    prefix = 't_emb'
    t_emb = get_timestep_embedding(timestep, embedding_dim=ch)
    t_emb = Dense(ch * 4, name=f'{prefix}/dense0')(t_emb)
    t_emb = swish(t_emb)
    t_emb = Dense(ch * 4, name=f'{prefix}/dense1')(t_emb)
    assert t_emb.shape == [inputs.shape[0], ch * 4]
    
    # down-sampling
    xs = [Conv2D(ch, 3, 1, 'same', name='conv_in')(inputs)]
    for i_level in tf.range(num_resolutions):
        prefix = f'down_{i_level}'
        for i_block in tf.range(num_res_blocks):
            x = res_block(
                inputs=xs[-1],
                time_embedding=t_emb,
                out_ch=ch * ch_mult[i_level],
                conv_shortcut=False,
                dropout=dropout,
                name=f'{prefix}/block_{i_block}'
            )
            if x.shape[1] in attn_resolutions:
                x = attention_block(x, name=f'{prefix}/attn_{i_block}')
            xs.append(x)
        # downsample
        if i_level != num_resolutions - 1:
            xs.append(downsample(xs[-1], with_conv=resamp_with_conv, name=f'{prefix}/downsample'))
    
    # middle
    prefix = 'mid'
    x = xs[-1]
    x = res_block(x, t_emb, None, False, dropout, name=f'{prefix}/block1')
    x = attention_block(x, name=f'{prefix}/attn1')
    x = res_block(x, t_emb, None, False, dropout, name=f'{prefix}/block2')
    
    # up-sampling
    for i_level in reversed(tf.range(num_resolutions)):
        prefix = f'up_{i_level}'
        for i_block in tf.range(num_res_blocks + 1):
            x = res_block(
                inputs=tf.concat([x, xs.pop()], axis=-1),
                time_embedding=t_emb,
                out_ch=ch * ch_mult[i_level],
                conv_shortcut=False,
                dropout=dropout,
                name=f'{prefix}/block_{i_block}'
            )
            if x.shape[1] in attn_resolutions:
                x = attention_block(x, name=f'{prefix}/attn_{i_block}')
        # upsample
        if i_level != 0:
            x = upsample(x, with_conv=resamp_with_conv, name=f'{prefix}/upsample')
    assert not xs
        
    # end
    x = group_norm(x, name='norm_out')
    x = swish(x)
    x = Conv2D(out_ch, 3, 1, 'same', init_scale=0., name='conv_out')(x)
    assert x.shape == inputs.shape[:3] + [out_ch]
    return tf.keras.Model([inputs, timestep], x, name=name)
