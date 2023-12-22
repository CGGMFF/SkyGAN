# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
from secondary_channels import SecondaryChannels
import torch
import training.utils
from training.training_loop import save_image_grid, stretch
import training.dataset
import training.loss
import legacy
from training.dataset import ImageFolderDataset

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    if s == 'elevations+1000': return []
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def load_nets(network_pkl, device):
    with dnnlib.util.open_url(network_pkl) as f:
        p = legacy.load_network_pkl(f)
        E = p['E_ema'].to(device) # type: ignore
        G = p['G_ema'].to(device) # type: ignore
        D = p['D'].to(device) # type: ignore
    return E, G, D

#----------------------------------------------------------------------------

def generate_clear_img(elevation, azimuth, device, sc, resolution=1024):
    clear_img = torch.from_numpy(
            training.dataset.generate_clear_sky_image_and_secondary_channels(
                resolution=resolution, secondary_channels=sc, azimuth=azimuth, elevation=elevation)
                )
    clear_img = clear_img[None, ...] # add the "batch" dimension
    print("clear_img.shape", clear_img.shape)
    return clear_img

#----------------------------------------------------------------------------

def save_img(img, outdir, seed, prefix='', drange=[0,1], ldr=True, hdr=False):
    #img_ldr = (torch.Tensor(img).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #PIL.Image.fromarray(img_ldr[0].cpu().numpy(), 'RGB').save(f'{outdir}/{prefix}_ldrseed{seed:04d}.png')
    fname = f'{outdir}/{prefix}_seed{seed:04d}.png'
    grid_size = [1,1]
    if ldr:
        save_image_grid(
            img,
            fname,
            drange=drange, grid_size=grid_size
        )
    if hdr:
        save_image_grid(
            img,
            fname.replace('.png', '.exr'),
            drange=[-1,1], grid_size=grid_size, hdr=True
        )

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
#@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--elevations', help='Elevation angles in degrees', type=parse_range, required=True)
@click.option('--azimuth', help='Azimuth angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--device', help='Device for inference', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
@click.option('--normalize-azimuth', 'normalize_azimuth', help='Force azimuths to 180 deg', metavar='BOOL', type=bool, default=False, show_default=True)
#@click.option('--use-circular-mask', 'use_circular_mask', help='Mask the clear', metavar='BOOL', type=bool, default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    #noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    elevations: List[int],
    azimuth: float,
    device: str,
    normalize_azimuth: bool,
    #use_circular_mask: bool,
    class_idx: Optional[int],
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device(device)
    E, G, D = load_nets(network_pkl, device)

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    #import pdb; pdb.set_trace()
    resolution = G.img_resolution
    print('will use the resolution', resolution)
    sc = SecondaryChannels(resolution)

    for elevation in elevations:
        clear_img = generate_clear_img(elevation, azimuth if not normalize_azimuth else 180, device, sc, resolution=resolution).to(device)
        if True: #use_circular_mask:
            mask = torch.from_numpy(
                training.utils.circular_mask(np.array((1, resolution, resolution)))
                ).to(device)
            clear_img *= mask
        clear_img = stretch(clear_img.to(device).to(torch.float32))
        save_img(clear_img.cpu().numpy()[:, :3, :, :], outdir, -1, 'clear', drange=[-1,1]) # limit to RGB (in NCHW)
        #exit(123)

        loss = training.loss.StyleGAN2Loss(device, E, G, D)

        # Generate images.
        for seed_idx, seed in enumerate(seeds if len(seeds) > 0 else [elevation+1000, elevation+2000, elevation+3000, elevation+4000, elevation+5000, elevation+6000, elevation+7000]): # fallback: use the elevation+1000 as seed
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            # Construct an inverse rotation/translation matrix and pass to the generator.  The
            # generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(G.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            img_fake, img_clear_rec, _gen_ws = loss.run_G(
                z, c=None, clear_img=clear_img, update_emas=False)

            img_fake = training.utils.invert_log_transform(img_fake.cpu()).numpy()
            img_clear_rec = img_clear_rec.cpu().numpy()

            if normalize_azimuth:
                def rot(img, azimuth):
                    #print('rotating', type(img), img.shape)
                    img = img[0]
                    img = img.transpose(1, 2, 0) # CHW -> HWC
                    img = ImageFolderDataset.rotate_image(img, azimuth - 180)
                    img = img.transpose(2, 0, 1) # HWC -> CHW
                    img = np.expand_dims(img, 0) # re-add batch dimension
                    #print('rotated', type(img), img.shape)
                    return img
                
                img_fake = rot(img_fake, azimuth)
                img_clear_rec = rot(img_clear_rec, azimuth)

            save_img(img_fake, outdir, seed, 'fake', hdr=True)
            save_img(img_clear_rec, outdir, seed, 'clear_rec', drange=[-1,1])

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
