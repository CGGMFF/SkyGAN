// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "bias_act.h"
#include <ipex.h>

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };

//------------------------------------------------------------------------
// CUDA kernel.

template <class T, int A>
/*
DPCT1110:0: The total declared local variable size in device function
bias_act_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void bias_act_kernel(bias_act_kernel_params p, const sycl::nd_item<3> &item_ct1)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    int G                 = p.grad;
    scalar_t alpha        = (scalar_t)p.alpha;
    scalar_t gain         = (scalar_t)p.gain;
    scalar_t clamp        = (scalar_t)p.clamp;
    scalar_t one          = (scalar_t)1;
    scalar_t two          = (scalar_t)2;
    scalar_t expRange     = (scalar_t)80;
    scalar_t halfExpRange = (scalar_t)40;
    scalar_t seluScale    = (scalar_t)1.0507009873554804934193349852946;
    scalar_t seluAlpha    = (scalar_t)1.6732632423543772848170429916717;

    // Loop over elements.
    int xi = item_ct1.get_group(2) * p.loopX * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    for (int loopIdx = 0; loopIdx < p.loopX && xi < p.sizeX;
         loopIdx++, xi += item_ct1.get_local_range(2))
    {
        // Load.
        scalar_t x = (scalar_t)((const T*)p.x)[xi];
        scalar_t b = (p.b) ? (scalar_t)((const T*)p.b)[(xi / p.stepB) % p.sizeB] : 0;
        scalar_t xref = (p.xref) ? (scalar_t)((const T*)p.xref)[xi] : 0;
        scalar_t yref = (p.yref) ? (scalar_t)((const T*)p.yref)[xi] : 0;
        scalar_t dy = (p.dy) ? (scalar_t)((const T*)p.dy)[xi] : one;
        scalar_t yy = (gain != 0) ? yref / gain : 0;
        scalar_t y = 0;

        // Apply bias.
        ((G == 0) ? x : xref) += b;

        // linear
        if (A == 1)
        {
            if (G == 0) y = x;
            if (G == 1) y = x;
        }

        // relu
        if (A == 2)
        {
            if (G == 0) y = (x > 0) ? x : 0;
            if (G == 1) y = (yy > 0) ? x : 0;
        }

        // lrelu
        if (A == 3)
        {
            if (G == 0) y = (x > 0) ? x : x * alpha;
            if (G == 1) y = (yy > 0) ? x : x * alpha;
        }

        // tanh
        if (A == 4)
        {
            if (G == 0) {
                scalar_t c = sycl::exp(x); scalar_t d = one / c;
                y = (x < -expRange)  ? -one
                    : (x > expRange) ? one
                                     : (c - d) / (c + d);
            }
            if (G == 1) y = x * (one - yy * yy);
            if (G == 2) y = x * (one - yy * yy) * (-two * yy);
        }

        // sigmoid
        if (A == 5)
        {
            if (G == 0) y = (x < -expRange) ? 0 : one / (sycl::exp(-x) + one);
            if (G == 1) y = x * yy * (one - yy);
            if (G == 2) y = x * yy * (one - yy) * (one - two * yy);
        }

        // elu
        if (A == 6)
        {
            if (G == 0) y = (x >= 0) ? x : sycl::exp(x) - one;
            if (G == 1) y = (yy >= 0) ? x : x * (yy + one);
            if (G == 2) y = (yy >= 0) ? 0 : x * (yy + one);
        }

        // selu
        if (A == 7)
        {
            if (G == 0) y =
                (x >= 0) ? seluScale * x
                         : (seluScale * seluAlpha) * (sycl::exp(x) - one);
            if (G == 1) y = (yy >= 0) ? x * seluScale : x * (yy + seluScale * seluAlpha);
            if (G == 2) y = (yy >= 0) ? 0 : x * (yy + seluScale * seluAlpha);
        }

        // softplus
        if (A == 8)
        {
            if (G == 0) y = (x > expRange) ? x : sycl::log(sycl::exp(x) + one);
            if (G == 1) y = x * (one - sycl::exp(-yy));
            if (G == 2) { scalar_t c = sycl::exp(-yy); y = x * c * (one - c); }
        }

        // swish
        if (A == 9)
        {
            if (G == 0)
                y = (x < -expRange) ? 0 : x / (sycl::exp(-x) + one);
            else
            {
                scalar_t c = sycl::exp(xref);
                scalar_t d = c + one;
                if (G == 1)
                    y = (xref > halfExpRange) ? x : x * c * (xref + d) / (d * d);
                else
                    y = (xref > halfExpRange) ? 0 : x * c * (xref * (two - d) + two * d) / (d * d * d);
                yref = (xref < -expRange)
                           ? 0
                           : xref / (sycl::exp(-xref) + one) * gain;
            }
        }

        // Apply gain.
        y *= gain * dy;

        // Clamp.
        if (clamp >= 0)
        {
            if (G == 0)
                y = (y > -clamp & y < clamp) ? y : (y >= 0) ? clamp : -clamp;
            else
                y = (yref > -clamp & yref < clamp) ? y : 0;
        }

        // Store.
        ((T*)p.y)[xi] = (T)y;
    }
}

template <class T>
void bias_act_kernel(bias_act_kernel_params p, const sycl::nd_item<3> &item_ct1) {
    if (p.act == 1) bias_act_kernel<T, 1>(p, item_ct1);
    else if (p.act == 2) bias_act_kernel<T, 2>(p, item_ct1);
    else if (p.act == 3) bias_act_kernel<T, 3>(p, item_ct1);
    else if (p.act == 4) bias_act_kernel<T, 4>(p, item_ct1);
    else if (p.act == 5) bias_act_kernel<T, 5>(p, item_ct1);
    else if (p.act == 6) bias_act_kernel<T, 6>(p, item_ct1);
    else if (p.act == 7) bias_act_kernel<T, 7>(p, item_ct1);
    else if (p.act == 8) bias_act_kernel<T, 8>(p, item_ct1);
    else if (p.act == 9) bias_act_kernel<T, 9>(p, item_ct1);
    //else TODO fail
}

void bias_act_kernel_launch(bias_act_kernel_params p) {
    int blockSize = 4 * 32; // TODO tune, or rather remove and let the runtime choose its favorite work unit size
    int gridSize = (p.sizeX - 1) / (p.loopX * blockSize) + 1;
    
    auto device_type = c10::DeviceType::XPU;
    c10::impl::VirtualGuardImpl impl(device_type);
    c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
    auto& queue = xpu::get_queue_from_stream(c10_stream);
    
    queue.submit([&] (sycl::handler& cgh) {
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.dtype, "bias_act_xpu", [&]
        {
            cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, gridSize) * sycl::range<3>(1, 1, blockSize),
                        sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        bias_act_kernel<scalar_t>(p, item_ct1);
                    });
        });
    }).wait();
}

//------------------------------------------------------------------------
// CUDA kernel selection.

// template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p)
// {
//     if (p.act == 1) return (void*)bias_act_kernel<T, 1>;
//     if (p.act == 2) return (void*)bias_act_kernel<T, 2>;
//     if (p.act == 3) return (void*)bias_act_kernel<T, 3>;
//     if (p.act == 4) return (void*)bias_act_kernel<T, 4>;
//     if (p.act == 5) return (void*)bias_act_kernel<T, 5>;
//     if (p.act == 6) return (void*)bias_act_kernel<T, 6>;
//     if (p.act == 7) return (void*)bias_act_kernel<T, 7>;
//     if (p.act == 8) return (void*)bias_act_kernel<T, 8>;
//     if (p.act == 9) return (void*)bias_act_kernel<T, 9>;
//     return NULL;
// }

//------------------------------------------------------------------------
// Template specializations.

// template void* choose_bias_act_kernel<double>       (const bias_act_kernel_params& p);
// template void* choose_bias_act_kernel<float>        (const bias_act_kernel_params& p);
// template void* choose_bias_act_kernel<c10::Half>    (const bias_act_kernel_params& p);
/*
#define SPECIALIZE_FOR_1to9(kernel_name, T) \
template __global__ void kernel_name<T, 1> (bias_act_kernel_params p); \
template __global__ void kernel_name<T, 2> (bias_act_kernel_params p); \
template __global__ void kernel_name<T, 3> (bias_act_kernel_params p); \
template __global__ void kernel_name<T, 4> (bias_act_kernel_params p); \
template __global__ void kernel_name<T, 5> (bias_act_kernel_params p); \
template __global__ void kernel_name<T, 6> (bias_act_kernel_params p); \
template __global__ void kernel_name<T, 7> (bias_act_kernel_params p); \
template __global__ void kernel_name<T, 8> (bias_act_kernel_params p); \
template __global__ void kernel_name<T, 9> (bias_act_kernel_params p);

SPECIALIZE_FOR_1to9(bias_act_kernel, c10::Half)
SPECIALIZE_FOR_1to9(bias_act_kernel, float)
SPECIALIZE_FOR_1to9(bias_act_kernel, double)
*/
//template __global__ void bias_act_kernel<c10::Half> (bias_act_kernel_params p);
//template __global__ void bias_act_kernel<float> (bias_act_kernel_params p);
//template __global__ void bias_act_kernel<double> (bias_act_kernel_params p);

//------------------------------------------------------------------------
