# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import training.utils
import cv2

import legacy
from metrics import metric_main

dump_images = False

#----------------------------------------------------------------------------

def stretch(x): # [0, 1] -> [-1, 1]
    return x * 2 - 1

def unstretch(x): # [-1, 1] -> [0, 1]
    return (x + 1) / 2

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    
    images = stretch(np.stack(images))
    
    return (gw, gh), images, np.stack(labels)

#----------------------------------------------------------------------------

def linear2srgb(x):
    # adapted to numpy from http://www.cyril-richon.com/blog/2019/1/23/python-srgb-to-linear-linear-to-srgb, based on https://stackoverflow.com/questions/34472375/linear-to-srgb-conversion
    return np.where(x > 0.0031308, 1.055 * (x**(1.0 / 2.4)) - 0.055, 12.92 * x)

def linear2srgb_torch(x):
    # adapted to numpy from http://www.cyril-richon.com/blog/2019/1/23/python-srgb-to-linear-linear-to-srgb, based on https://stackoverflow.com/questions/34472375/linear-to-srgb-conversion
    return torch.where(x > 0.0031308, 1.055 * (x**(1.0 / 2.4)) - 0.055, 12.92 * x)

def save_image_grid(img, fname, drange, grid_size=[1,1], hdr=False):
    print('save_image_grid: img.shape', img.shape, fname)
    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2) # gh, H, gw, W, C
    img = img.reshape([gh * H, gw * W, C])
    
    if hdr:
        saved_successfully = cv2.imwrite(fname, img[:1024, :1024, ::-1]) # crop and RGB -> BGR
        assert saved_successfully
    else:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) / (hi - lo) # fix range
        img = np.rint(linear2srgb(img)*255).clip(0, 255).astype(np.uint8) # to LDR

        assert C in [1, 3]
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def run_E_and_G(E_ema, G_ema, clear_images, clear_labels, batch_gpu, device, grid_z, grid_c, use_encoder=True): # TODO refactoring: merge with loss.py: run_EG
    #training_stats.report('Extra/run_E_and_G/clear_images_min', clear_images.min())
    #training_stats.report('Extra/run_E_and_G/clear_images_mean', clear_images.mean())
    #training_stats.report('Extra/run_E_and_G/clear_images_max', clear_images.max())
    
    if use_encoder:
        if dump_images:
            ci = torch.Tensor(clear_images).to(device).split(batch_gpu)[0][:1].cpu().numpy()

            print('(going to E) ci.shape', ci.shape)

            ci_RGB = training.utils.clear_extract_rgb(ci)
            ci_extra = training.utils.clear_extract_extra(ci)

            run_dir = './'
            save_image_grid(ci_RGB, os.path.join(run_dir, 'tmp_training_clears_to_encode.png'), drange=[-1,1])
            save_image_grid(ci_extra, os.path.join(run_dir, 'tmp_training_clears_to_encode_extra.png'), drange=[-1,1])

        bottlenecks = [E_ema(im, lab).cpu() for im, lab in zip(
            torch.Tensor(clear_images).to(device).split(batch_gpu),
            torch.Tensor(clear_labels).to(device).split(batch_gpu)
            )]

        if dump_images:
            print('(from E) bottlenecks[0]', bottlenecks[0])
    else:
        #bottlenecks = (([None] * grid_z[0].shape[0]) for i in range(len(grid_z)))
        bottlenecks = (None for i in range(len(grid_z)))

    images = torch.cat([G_ema(z=z, c=c, injected_bottlenecks=b, noise_mode='const').cpu() for z, c, b in zip(grid_z, grid_c, bottlenecks)]).numpy()
    
    if dump_images:
        save_image_grid(training.utils.generator_output_extract_clear(images[:1]), os.path.join(run_dir, 'tmp_training_clears_rec_init.png'), drange=[-1,1])

    return images

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    clear_training_set_kwargs = {},     # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    E_kwargs                = {},       # Options for encoder network.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    E_opt_kwargs            = {},       # Options for encoder optimizer.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    EG_reg_interval         = None,     # How often to perform regularization for E and G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    use_encoder             = True,
    resume_augment_pipe    = False,    # Restore the saved parameters of the augmentation
):
    fid_fname = resume_pkl+'.fid'
    assert os.path.exists(resume_pkl)

    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    clear_training_set = dnnlib.util.construct_class_by_name(**clear_training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    clear_training_set_iterator = iter(torch.utils.data.DataLoader(dataset=clear_training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))

    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution)

    E_common_kwargs = dict(img_resolution=training_set.resolution)
    E_kwargs.num_outputs = 10 # 10 outputs (bottleneck size)
    #E_kwargs.img_channels = 3 # RGB
    E_kwargs.img_channels = 4 # RGB + polar distance to Sun direction

    assert training_set.num_channels == 3
    G_kwargs.img_channels = 3 + 3 # RGB cloudy sky + RGB clear sky

    D_kwargs.img_channels = training_set.num_channels

    E = dnnlib.util.construct_class_by_name(**E_kwargs, **E_common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    E_ema = copy.deepcopy(E).eval()
    G_ema = copy.deepcopy(G).eval()

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        modules_to_restore = [('E', E), ('G', G), ('D', D), ('E_ema', E_ema), ('G_ema', G_ema)]
        if resume_augment_pipe:
            modules_to_restore.append(('augment_pipe', augment_pipe))
        for name, module in modules_to_restore:
            if name not in resume_data.keys():
                print('WARNING: skipping', name, '- not present in resume_data')
                continue
            print('Copying', name)
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        del resume_data

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [E, G, D, E_ema, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, E=E, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, modules, opts_kwargs, run_interval, reg_interval in [
        ('EG', [E, G], [E_opt_kwargs, G_opt_kwargs], 1, EG_reg_interval),
        ('D', [D], [D_opt_kwargs], 1, D_reg_interval)
        ]:
    
        if reg_interval is None:
            print(name, 'will merge "main" and "reg" into "both"')
            #opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            opts = [dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) for module, opt_kwargs in zip(modules, opts_kwargs)] # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', modules=modules, opts=opts, interval=1)]
        else: # Lazy regularization.
            print(name, 'will use lazy regularization')
            opts = []
            for module, opt_kwargs in zip(modules, opts_kwargs):
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opts += [dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)] # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', modules=modules, opts=opts, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', modules=modules, opts=opts, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)

        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        done = (cur_nimg >= total_kimg * 1000)
        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(E=E, G=G, D=D, E_ema=E_ema, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    print('DEBUG: skipping pickle dump')
                    #pickle.dump(snapshot_data, f)
        
        # Evaluate metrics.
        snapshot_data='something not None'
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(
                    metric=metric,
                    E=E,
                    G=G,
                    dataset_kwargs=training_set_kwargs, clear_dataset_kwargs=clear_training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    print('result_dict', result_dict)
                    with open(fid_fname, 'w') as f_out:
                        f_out.write(str(result_dict['results']['fid50k_full']))
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        print('DEBUG: early stop after computing the metrics')
        break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
