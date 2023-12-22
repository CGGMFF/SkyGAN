import numpy as np
from torch_utils import training_stats
import dnnlib

# expects shape in CHW
def circular_mask(shape):
    assert len(shape) == 3
    x_coords = np.linspace(-1., 1., shape[2])
    y_coords = np.linspace(-1., 1., shape[1])
    x,y = np.meshgrid(x_coords, y_coords)

    # circular black/white mask for each image
    mask = np.ones(shape, dtype=np.uint8)
    mask[:, (x*x+y*y) > 1.0] = 0
    return mask

def inject_bottleneck_into_ws(bottleneck, ws):
    #return bottleneck.cuda() # this would replace the ws with bottleneck completely
    # print('bottleneck.shape', bottleneck.shape)
    # print('ws.shape', ws.shape)
    #print('bottleneck mean', bottleneck.mean())
    assert bottleneck.shape[-1] == 10
    assert ws.shape[-1] == 512
    
    #training_stats.report('Extra/utils/inject_bottleneck_min', bottleneck.min())
    #training_stats.report('Extra/utils/inject_bottleneck_mean', bottleneck.mean())
    #training_stats.report('Extra/utils/inject_bottleneck_max', bottleneck.max())
    
    bottleneck_norm = (bottleneck - bottleneck.mean()) / bottleneck.std() # normalise: zero mean, std == 1
    
    #training_stats.report('Extra/utils/inject_bottleneck_norm_min', bottleneck_norm.min())
    #training_stats.report('Extra/utils/inject_bottleneck_norm_mean', bottleneck_norm.mean())
    #training_stats.report('Extra/utils/inject_bottleneck_norm_max', bottleneck_norm.max())

    #training_stats.report('Extra/utils/inject_ws_min', ws.min())
    #training_stats.report('Extra/utils/inject_ws_mean', ws.mean())
    #training_stats.report('Extra/utils/inject_ws_max', ws.max())

    #print('bottleneck_norm mean', bottleneck_norm.mean())

    #print('ws pre mean', ws.mean())
    ws[..., 0:bottleneck.shape[1]] = bottleneck_norm.unsqueeze(1)
    #print('ws post mean', ws.mean())
    return ws

def generator_output_extract_fake(img):
    assert img.shape[1] == 6 # 6 channels
    out_img = img[:, 0:3, :, :]
    return out_img

def generator_output_extract_clear(img):
    assert img.shape[1] == 6 # 6 channels
    return img[:, 3:, :, :]

def clear_extract_rgb(img): # discard guiding channel(s)
    assert img.shape[1] == 4 # channels: RGB + polar distance
    out_img = img[:, 0:3, :, :]
    return out_img

def clear_extract_extra(img): # only extra (guiding) channels
    assert img.shape[1] == 4 # channels: RGB + polar distance
    out_img = img[:, 3:, :, :]
    return out_img


## *********************
## Image transformations
## *********************

transform_cfg = dnnlib.EasyDict(
    # Multiplier value applied before the log transform
    input_mul = 1, # e.g. 2**8 would shift it 8 EV steps up
    # The epsilon constant for log-mapping input HDR images.
    log_epsilon = 1e-3,
    # separate the value space around 1.0 into two separate mapping functions
    # - x < 1: log(x) => expansion of the value space
    # - x >= 1: pow(x, 1./log_pow) => adjustable compression of the value space
    log_split_around1 = False,
    log_pow = 7.5,
    # Shift the value up/down (after the log transform) - can be used to ensure zero-mean
    output_bias = 2.5,
    output_scale = 1/2.2/2,
)

# Applies the log-transformation that converts linear HDR images to a form suitable for a
# neural network.
def log_transform(x):
    if transform_cfg.input_mul != 1.0:
        x = x * transform_cfg.input_mul
    x = x + transform_cfg.log_epsilon

    log_x = np.log(x)
    pow_x = np.power(x, 1./transform_cfg.log_pow) - 1.0

    if transform_cfg.log_split_around1:
        x = np.where(x < 1.0, log_x, pow_x)
    else:
        x = log_x
    return (x + transform_cfg.output_bias) * transform_cfg.output_scale

# Inverse of log_transform(x)
def invert_log_transform(y):
    y = y / transform_cfg.output_scale - transform_cfg.output_bias

    exp_y = np.exp(y)
    pow_y = np.power(y + 1.0, transform_cfg.log_pow)

    if transform_cfg.log_split_around1:
        y = np.where(y < 0.0/transform_cfg.input_mul, exp_y, pow_y)
    else:
        y = exp_y
    y = y - transform_cfg.log_epsilon
    
    if transform_cfg.input_mul != 1.0:
        y = y / transform_cfg.input_mul
    return y

# Inverse of log_transform(x)
def invert_log_transform_torch(y):
    import torch
    y = y / transform_cfg.output_scale - transform_cfg.output_bias

    exp_y = torch.exp(y)
    pow_y = torch.pow(y + 1.0, transform_cfg.log_pow)

    if transform_cfg.log_split_around1:
        y = torch.where(y < 0.0/transform_cfg.input_mul, exp_y, pow_y)
    else:
        y = exp_y
    y = y - transform_cfg.log_epsilon
    
    if transform_cfg.input_mul != 1.0:
        y = y / transform_cfg.input_mul
    return y


def fix_gamma(x):
    return np.power(x, 1/2.2)

# Converts linear HDR in [0, 1] into gamma-corrected LDR in [0, 255].
def hdr_to_ldr(x):
    return fix_gamma(x) * 255
