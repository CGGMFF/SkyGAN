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

def run_E_and_G(E_ema, G_ema, clear_images, clear_labels, batch_gpu, device, grid_z, grid_c, use_encoder=True):
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

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        print('SKIPPING Encoder SUMMARY')
        #misc.print_module_summary(E, [img, c]) # this will actually use a different img: the clear sky
        misc.print_module_summary(D, [training.utils.generator_output_extract_fake(img), c])


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
        print('images', images.min(), images.mean(), images.max())
        save_image_grid(training.utils.invert_log_transform(images),
            os.path.join(run_dir, 'reals.png'),
            drange=[0,1], grid_size=grid_size
        )
        save_image_grid(
            training.utils.invert_log_transform(images),
            os.path.join(run_dir, 'reals.exr'),
            drange=[-1,1], grid_size=grid_size, hdr=True
        )

        clear_grid_size, clear_images, clear_labels = setup_snapshot_image_grid(training_set=clear_training_set)
        print('clear_images.shape', clear_images.shape)
        clear_images_RGB = training.utils.clear_extract_rgb(clear_images)
        clear_images_extra = training.utils.clear_extract_extra(clear_images)

        save_image_grid(clear_images_RGB, os.path.join(run_dir, 'clears.png'), drange=[-1,1], grid_size=grid_size)
        save_image_grid(clear_images_extra, os.path.join(run_dir, 'clears_extra.png'), drange=[-1,1], grid_size=grid_size)
        
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)

        images = run_E_and_G(E_ema, G_ema, clear_images, clear_labels, batch_gpu, device, grid_z, grid_c, use_encoder)
        save_image_grid(
            training.utils.invert_log_transform(training.utils.generator_output_extract_fake(images)),
            os.path.join(run_dir, 'fakes_init.png'),
            drange=[0,1], grid_size=grid_size
        )
        save_image_grid(
            training.utils.invert_log_transform(training.utils.generator_output_extract_fake(images)),
            os.path.join(run_dir, 'fakes_init.exr'),
            drange=[-1,1], grid_size=grid_size, hdr=True
        )
        save_image_grid(training.utils.generator_output_extract_clear(images), os.path.join(run_dir, 'clears_rec_init.png'), drange=[-1,1], grid_size=grid_size)


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

        # Fetch training data. (and shift the images range from [0, 1+] to [-1, 1+])
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = stretch(phase_real_img.to(device).to(torch.float32)).split(batch_gpu)
            phase_clear_img, phase_clear_c = next(clear_training_set_iterator)
            phase_clear_img = stretch(phase_clear_img.to(device).to(torch.float32)).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        #print('Execute training phases.')
        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            #print('phase', phase.name)
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            #print('Accumulate gradients.')
            # Accumulate gradients.
            for opt in phase.opts:
                opt.zero_grad(set_to_none=True)
            for module in phase.modules:
                module.requires_grad_(True)
            
            for real_img, clear_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_clear_img, phase_real_c, phase_gen_z, phase_gen_c):
                #print('Accumulate gradients...')
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, clear_img=clear_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            
            for module in phase.modules:
                module.requires_grad_(False)

            #print('Update weights.')
            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for module in phase.modules:
                    named_params = [(name, param) for name, param in module.named_parameters() if param.grad is not None]
                    param_names = [name for name, param in named_params]
                    params = [param for name, param in named_params]
                    #print('len(params)', len(params))
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                        #print('flat isnan', flat.isnan().any())
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for name, param, grad in zip(param_names, params, grads):
                            #print(f'param {param.min():.5f} {param.mean():.5f} {param.max():.5f} grad {grad.min():.5f} {grad.mean():.5f} {grad.max():.5f}')

                            
                            #training_stats.report('Extra/'+phase.name+'/param_min', param.min())
                            #training_stats.report('Extra/'+phase.name+'/param_mean', param.mean())
                            training_stats.report('Extra/'+phase.name+'/param_abs_mean_'+name, param.abs().mean())
                            #training_stats.report('Extra/'+phase.name+'/param_max', param.max())
                            
                            #training_stats.report('Extra/'+phase.name+'/grad_min', grad.min())
                            #training_stats.report('Extra/'+phase.name+'/grad_mean', grad.mean())
                            training_stats.report('Extra/'+phase.name+'/grad_abs_mean', grad.abs().mean())
                            #training_stats.report('Extra/'+phase.name+'/grad_abs_mean_'+name, grad.abs().mean())
                            #training_stats.report('Extra/'+phase.name+'/grad_max', grad.max())
                            

                            param.grad = grad.reshape(param.shape)
                for opt in phase.opts:
                    opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        #print('Update G_ema.')
        # Update G_ema. (exponential moving average)
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
        
        #print('Update E_ema.')
        # Update E_ema. (exponential moving average)
        with torch.autograd.profiler.record_function('Eema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(E_ema.parameters(), E.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(E_ema.buffers(), E.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = run_E_and_G(E_ema, G_ema, clear_images, clear_labels, batch_gpu, device, grid_z, grid_c, use_encoder)
            save_image_grid(
                training.utils.invert_log_transform(training.utils.generator_output_extract_fake(images)),
                os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'),
                drange=[0,1], grid_size=grid_size
            )
            save_image_grid(
                training.utils.invert_log_transform(training.utils.generator_output_extract_fake(images)),
                os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.exr'),
                drange=[-1,1], grid_size=grid_size, hdr=True
            )
            save_image_grid(training.utils.generator_output_extract_clear(images), os.path.join(run_dir, f'clears_rec{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

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
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric,
                    G=training.networks_stylegan3.OutputFilterWrapper(snapshot_data['G_ema'], training.utils.generator_output_extract_fake), # keep just RGB channels corresponding to the cloudy/fake images
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
