# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import training.training_loop
import training.utils

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

def run_EG_mapping(z, c, clear_img, use_encoder, E, G, style_mixing_prob, truncation_psi=1, truncation_cutoff=None, update_emas=False):
    if use_encoder:
        bottleneck = E(clear_img, c, update_emas=update_emas)

    ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
    if style_mixing_prob > 0:
        with torch.autograd.profiler.record_function('style_mixing'):
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
    if use_encoder:
        ws = training.utils.inject_bottleneck_into_ws(bottleneck, ws)
    return z, ws

def run_G_synthesis(ws, G, mask, update_emas=False):
    img = G.synthesis(ws, update_emas=update_emas)
    #print('img.shape, mask.shape', img.shape, mask.shape)
    img = (mask.to(img.device) * (img + 1.0) - 1.0) # img is in [-1,1], mask is {0,1}
    return training.utils.generator_output_extract_fake(img), training.utils.generator_output_extract_clear(img), ws

def run_EG(z, c, clear_img, use_encoder, E, G, style_mixing_prob, mask, update_emas=False, truncation_psi=1, truncation_cutoff=None): # (truncation parameters are used only from visualizer)
    #grid_size=[2,2]
    #training.training_loop.save_image_grid(training.utils.clear_extract_rgb(clear_img).cpu().numpy(), f'{i:04d}run_G_clear.png', [-1, 1], grid_size=grid_size)

    z, ws = run_EG_mapping(z, c, clear_img, use_encoder, E, G, style_mixing_prob, truncation_psi, truncation_cutoff, update_emas)
    img_fake, img_clear_rec, _gen_ws = run_G_synthesis(ws, G, mask, update_emas)

    #training.training_loop.save_image_grid(img_clear_rec.cpu().detach().numpy(), f'{i:04d}run_clear_rec{i}.png', [-1, 1], grid_size=grid_size)

    #img_fake_transformed = training.utils.invert_log_transform(img_fake.cpu().detach().numpy())
    #training.training_loop.save_image_grid(
    #    img_fake_transformed,
    #    f'{i:04d}run_G_fake.png', [0, 1], grid_size=grid_size)
    #print('run_EG: img_fake_transformed', img_fake_transformed.min(), img_fake_transformed.mean(), img_fake_transformed.max())

    return img_fake, img_clear_rec, _gen_ws

class StyleGAN2Loss(Loss):
    def __init__(self, device, E, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, use_encoder=True, reg_encoder=True):
        super().__init__()
        self.device             = device
        self.E                  = E
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.use_encoder        = use_encoder
        self.reg_encoder        = reg_encoder
        self.mask = torch.from_numpy(training.utils.circular_mask((G.img_channels, G.img_resolution, G.img_resolution)))

    def run_G_mapping(self, z, c, clear_img, truncation_psi=1, truncation_cutoff=None, update_emas=False): # TODO refactor (only visualizer uses this -> remove and use the functions above)
        return run_EG_mapping(
            z, c, clear_img,
            use_encoder=self.use_encoder, E=self.E, G=self.G, style_mixing_prob=self.style_mixing_prob,
            truncation_psi=1, truncation_cutoff=None, update_emas=False
        )

    def run_G_synthesis(self, ws, update_emas=False): # TODO refactor (only visualizer uses this -> remove and use the functions above)
        return run_G_synthesis(ws, self.G, self.mask, update_emas)

    def run_G(self, z, c, clear_img, update_emas=False):
        return run_EG(
            z=z, c=c, clear_img=clear_img, use_encoder=self.use_encoder,
            E=self.E, G=self.G,
            style_mixing_prob=self.style_mixing_prob, mask=self.mask, update_emas=update_emas
            )

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, clear_img, real_c, gen_z, gen_c, gain, cur_nimg):
        #print('accumulate_gradients phase:', phase, end=' ')
        assert phase in ['EGmain', 'EGreg', 'EGboth', 'Dmain', 'Dreg', 'Dboth']
        losses = 0
        
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['EGmain', 'EGboth']:
            with torch.autograd.profiler.record_function('EGmain_forward'):
                gen_z = torch.clone(gen_z) # clone: avoid "RuntimeError: A view was created in no_grad mode and is being modified inplace with grad mode enabled. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one.""

                gen_img, regen_img_clear, _gen_ws = self.run_G(gen_z, gen_c, clear_img)

                #training_stats.report('Extra/E/real_img_min', real_img.min())
                #training_stats.report('Extra/E/real_img_mean', real_img.mean())
                #training_stats.report('Extra/E/real_img_max', real_img.max())
                
                #training_stats.report('Extra/E/gen_img_min', gen_img.min())
                training_stats.report('Extra/E/gen_img_mean', gen_img.mean())
                training_stats.report('Extra/E/gen_img_max', gen_img.max())

                #training_stats.report('Extra/E/clear_img_min', clear_img.min())
                #training_stats.report('Extra/E/clear_img_mean', clear_img.mean())
                #training_stats.report('Extra/E/clear_img_max', clear_img.max())

                #training_stats.report('Extra/E/regen_img_clear_min', regen_img_clear.min())
                #training_stats.report('Extra/E/regen_img_clear_mean', regen_img_clear.mean())
                #training_stats.report('Extra/E/regen_img_clear_max', regen_img_clear.max())

                # Reconstruction loss:
                target = training.utils.clear_extract_rgb(clear_img)
                #target = torch.zeros_like(regen_img_clear)
                
                #training_stats.report('Extra/E/target_min', target.min())
                #training_stats.report('Extra/E/target_mean', target.mean())
                #training_stats.report('Extra/E/target_max', target.max())

                if self.use_encoder:
                    loss_clear_reconstruction = torch.nn.functional.mse_loss(input=regen_img_clear, target=target, reduction='mean')
                    training_stats.report('Loss/E/loss', loss_clear_reconstruction)

                # Adversarial loss:
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('EGmain_backward'):
                if self.use_encoder:
                    losses += loss_clear_reconstruction.mul(gain*10000)
                losses += loss_Gmain.mean().mul(gain)

        # Gpl: Apply path length regularization.
        if (phase in ['EGreg', 'EGboth']) and (self.pl_weight != 0):
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, regen_img_clear, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], clear_img)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay) # exponential moving average
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                losses += loss_Gpl.mean().mul(gain)
        
        # Er1: E gradient regularization.
        if (phase in ['EGreg', 'EGboth']) and self.reg_encoder and (self.r1_gamma != 0):
            with torch.autograd.profiler.record_function('Er1_forward'):
                clear_img_tmp = clear_img.detach().requires_grad_(True)
                bottleneck_tmp = self.E(clear_img_tmp, real_c, update_emas=False)
                with torch.autograd.profiler.record_function('Er1_grads'), conv2d_gradfix.no_weight_gradients():
                    Er1_grads = torch.autograd.grad(outputs=[bottleneck_tmp.sum()], inputs=[clear_img_tmp], create_graph=True, only_inputs=True)[0] # TODO is sum correct for this shape (not just one output per input image like for the discriminator, but a 10-value bottleneck)? (or should we reduce/flatten/mean... it first?)
                Er1_penalty = Er1_grads.square().sum([1,2,3])
                loss_Er1 = Er1_penalty * (self.r1_gamma / 2)
                training_stats.report('Loss/Er1_penalty', Er1_penalty)
                training_stats.report('Loss/E/reg', loss_Er1)

            with torch.autograd.profiler.record_function(phase + '_backward'):
                losses += loss_Er1.mean().mul(gain)

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, regen_img_clear, _gen_ws = self.run_G(gen_z, gen_c, clear_img, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                losses += loss_Dgen.mean().mul(gain)

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            if (phase == 'Dmain') or (self.r1_gamma == 0):
                name = 'Dreal'
            elif phase == 'Dreg':
                name = 'Dr1'
            else:
                name = 'Dreal_Dr1'
           
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0):
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                losses += (loss_Dreal + loss_Dr1).mean().mul(gain)
            
        if type(losses) == type(0):
            # unchanged since the first assignment
            print('WARNING: phase', phase, 'did not add any loss')
            return

        with torch.autograd.profiler.record_function('phase_' + phase + '_backward'):
            losses.backward()

#----------------------------------------------------------------------------
