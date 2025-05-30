// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "upfirdn2d.h"
#include <exception>

//------------------------------------------------------------------------

template <class T> void choose_and_run_upfirdn2d_kernel(const upfirdn2d_kernel_params& p)
{
    // The original CUDA implementation initially sets a default kernel pointer (+ other parameters, together called 'spec') to a general "large" kernel, and then goes through a list of (nested) `if`s handling special cases, each of which could overwrite the 'spec' -> the last one matching (the most specific match) is used.
    // Since we can't use kernel pointers (DPCT/c2s can't convert them to SYCL; it supports only a direct call for cudaLaunchKernel), we need to change/invert the order of `if`s and make them "disjunt" (so the first matching one is used/called, and none other).
    // Refactoring from "last overwritten value is used" to "call the first matching kernel" was done like this:
    //  - invert the order (from the most specific to the general)
    //  - "if" -> "else if", except for the first if in group
    //  - add "else throw ..." after each if-else group (safeguard, TODO: is it needed?)
    //  - change from the old format (setting "spec = {kernel_ptr, other_params...}") to the new format (just calling the kernel):
    //      sed -ibak 's/ spec = {(void\*)upfirdn2d_kernel_small<\(.*\)>,.*};/ run_upfirdn2d_kernel_small<\1>(p);/;s/ spec = {(void\*)upfirdn2d_kernel_large<T>,\(.*\)};/ run_upfirdn2d_kernel_large<T>(p,\1);/' torch_utils/ops/upfirdn2d.cpp
    //    (older:)  sed -ibak 's/ spec = {(void\*)\(upfirdn2d_kernel_small\)<\(.*\)>,.*};/ \1<\2>(p);/' torch_utils/ops/upfirdn2d.cpp
    int s = p.inStride.z, fx = p.filterSize.x, fy = p.filterSize.y;

    // handle special cases with a "small" kernel

    // 4x downsampling (inefficient).
    if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 4)
    {
        // channels_last
        if (s == 1 && fx <= 1  && fy <= 32) run_upfirdn2d_kernel_small<T, 1,1, 1,4, 1,32, 1,32,8>(p);
        else if (s == 1 && fx <= 1  && fy <= 48) run_upfirdn2d_kernel_small<T, 1,1, 1,4, 1,48, 1,32,8>(p);
        // contiguous
        else if (s != 1 && fx <= 1 && fy <= 32) run_upfirdn2d_kernel_small<T, 1,1, 1,4, 1,32, 32,8,1>(p);
        else if (s != 1 && fx <= 1 && fy <= 48) run_upfirdn2d_kernel_small<T, 1,1, 1,4, 1,48, 32,8,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }
    else if (p.up.x == 1 && p.up.y == 1 && p.down.x == 4 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 32 && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 4,1, 32,1, 32,1,8>(p);
        else if (s == 1 && fx <= 48 && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 4,1, 48,1, 32,1,8>(p);
        // contiguous
        else if (s != 1 && fx <= 32 && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 4,1, 32,1, 32,8,1>(p);
        else if (s != 1 && fx <= 48 && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 4,1, 48,1, 32,8,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }

    // 4x upsampling.
    else if (p.up.x == 1 && p.up.y == 4 && p.down.x == 1 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 1 && fy <= 32) run_upfirdn2d_kernel_small<T, 1,4, 1,1, 1,32, 1,128,16>(p);
        else if (s == 1 && fx <= 1 && fy <= 48) run_upfirdn2d_kernel_small<T, 1,4, 1,1, 1,48, 1,128,16>(p);
        // contiguous
        else if (s != 1 && fx <= 1 && fy <= 32) run_upfirdn2d_kernel_small<T, 1,4, 1,1, 1,32, 32,32,1>(p);
        else if (s != 1 && fx <= 1 && fy <= 48) run_upfirdn2d_kernel_small<T, 1,4, 1,1, 1,48, 32,32,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }
    else if (p.up.x == 4 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 32 && fy <= 1) run_upfirdn2d_kernel_small<T, 4,1, 1,1, 32,1, 128,1,16>(p);
        else if (s == 1 && fx <= 48 && fy <= 1) run_upfirdn2d_kernel_small<T, 4,1, 1,1, 48,1, 128,1,16>(p);
        // contiguous
        else if (s != 1 && fx <= 32 && fy <= 1) run_upfirdn2d_kernel_small<T, 4,1, 1,1, 32,1, 128,8,1>(p);
        else if (s != 1 && fx <= 48 && fy <= 1) run_upfirdn2d_kernel_small<T, 4,1, 1,1, 48,1, 128,8,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }
    else if (p.up.x == 4 && p.up.y == 4 && p.down.x == 1 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 32 && fy <= 32) run_upfirdn2d_kernel_small<T, 4,4, 1,1, 32,32, 32,32,1>(p);
        else if (s == 1 && fx <= 48 && fy <= 48) run_upfirdn2d_kernel_small<T, 4,4, 1,1, 48,48, 32,32,1>(p);
        // contiguous
        else if (s != 1 && fx <= 32 && fy <= 32) run_upfirdn2d_kernel_small<T, 4,4, 1,1, 32,32, 64,32,1>(p);
        else if (s != 1 && fx <= 48 && fy <= 48) run_upfirdn2d_kernel_small<T, 4,4, 1,1, 48,48, 64,32,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }

    // 2x downsampling.
    else if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 2)
    {
        // channels_last
        if (s == 1 && fx <= 1  && fy <= 8 ) run_upfirdn2d_kernel_small<T, 1,1, 1,2, 1,8,  1,64,8>(p);
        else if (s == 1 && fx <= 1  && fy <= 16) run_upfirdn2d_kernel_small<T, 1,1, 1,2, 1,16, 1,64,8>(p);
        else if (s == 1 && fx <= 1  && fy <= 24) run_upfirdn2d_kernel_small<T, 1,1, 1,2, 1,24, 1,64,8>(p);
        // contiguous
        else if (s != 1 && fx <= 1 && fy <= 8 ) run_upfirdn2d_kernel_small<T, 1,1, 1,2, 1,8,  32,16,1>(p);
        else if (s != 1 && fx <= 1 && fy <= 16) run_upfirdn2d_kernel_small<T, 1,1, 1,2, 1,16, 32,16,1>(p);
        else if (s != 1 && fx <= 1 && fy <= 24) run_upfirdn2d_kernel_small<T, 1,1, 1,2, 1,24, 32,16,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }
    else if (p.up.x == 1 && p.up.y == 1 && p.down.x == 2 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 8  && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 2,1, 8,1,  64,1,8>(p);
        else if (s == 1 && fx <= 16 && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 2,1, 16,1, 64,1,8>(p);
        else if (s == 1 && fx <= 24 && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 2,1, 24,1, 64,1,8>(p);
        // contiguous
        else if (s != 1 && fx <= 8  && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 2,1, 8,1,  64,8,1>(p);
        else if (s != 1 && fx <= 16 && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 2,1, 16,1, 64,8,1>(p);
        else if (s != 1 && fx <= 24 && fy <= 1) run_upfirdn2d_kernel_small<T, 1,1, 2,1, 24,1, 64,8,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }
    else if (p.up.x == 1 && p.up.y == 1 && p.down.x == 2 && p.down.y == 2)
    {
        // channels_last
        if (s == 1 && fx <= 2  && fy <= 2 ) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 2,2,   8,8,8>(p);
        else if (s == 1 && fx <= 4  && fy <= 4 ) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 4,4,   8,8,8>(p);
        else if (s == 1 && fx <= 6  && fy <= 6 ) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 6,6,   8,8,8>(p);
        else if (s == 1 && fx <= 8  && fy <= 8 ) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 8,8,   8,8,8>(p);
        else if (s == 1 && fx <= 16 && fy <= 16) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 16,16, 16,16,1>(p);
        else if (s == 1 && fx <= 24 && fy <= 24) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 24,24, 16,16,1>(p);
        // contiguous
        else if (s != 1 && fx <= 2  && fy <= 2 ) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 2,2,   32,8,1>(p);
        else if (s != 1 && fx <= 4  && fy <= 4 ) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 4,4,   32,8,1>(p);
        else if (s != 1 && fx <= 6  && fy <= 6 ) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 6,6,   32,8,1>(p);
        else if (s != 1 && fx <= 8  && fy <= 8 ) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 8,8,   32,8,1>(p);
        else if (s != 1 && fx <= 16 && fy <= 16) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 16,16, 32,16,1>(p);
        else if (s != 1 && fx <= 24 && fy <= 24) run_upfirdn2d_kernel_small<T, 1,1, 2,2, 24,24, 32,16,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }

    // 2x upsampling.
    else if (p.up.x == 1 && p.up.y == 2 && p.down.x == 1 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 1 && fy <= 8 ) run_upfirdn2d_kernel_small<T, 1,2, 1,1, 1,8,  1,128,16>(p);
        else if (s == 1 && fx <= 1 && fy <= 16) run_upfirdn2d_kernel_small<T, 1,2, 1,1, 1,16, 1,128,16>(p);
        else if (s == 1 && fx <= 1 && fy <= 24) run_upfirdn2d_kernel_small<T, 1,2, 1,1, 1,24, 1,128,16>(p);
        // contiguous
        else if (s != 1 && fx <= 1 && fy <= 8 ) run_upfirdn2d_kernel_small<T, 1,2, 1,1, 1,8,  32,32,1>(p);
        else if (s != 1 && fx <= 1 && fy <= 16) run_upfirdn2d_kernel_small<T, 1,2, 1,1, 1,16, 32,32,1>(p);
        else if (s != 1 && fx <= 1 && fy <= 24) run_upfirdn2d_kernel_small<T, 1,2, 1,1, 1,24, 32,32,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }
    else if (p.up.x == 2 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 8  && fy <= 1) run_upfirdn2d_kernel_small<T, 2,1, 1,1, 8,1,  128,1,16>(p);
        else if (s == 1 && fx <= 16 && fy <= 1) run_upfirdn2d_kernel_small<T, 2,1, 1,1, 16,1, 128,1,16>(p);
        else if (s == 1 && fx <= 24 && fy <= 1) run_upfirdn2d_kernel_small<T, 2,1, 1,1, 24,1, 128,1,16>(p);
        // contiguous
        else if (s != 1 && fx <= 8  && fy <= 1) run_upfirdn2d_kernel_small<T, 2,1, 1,1, 8,1,  128,8,1>(p);
        else if (s != 1 && fx <= 16 && fy <= 1) run_upfirdn2d_kernel_small<T, 2,1, 1,1, 16,1, 128,8,1>(p);
        else if (s != 1 && fx <= 24 && fy <= 1) run_upfirdn2d_kernel_small<T, 2,1, 1,1, 24,1, 128,8,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }
    else if (p.up.x == 2 && p.up.y == 2 && p.down.x == 1 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 2  && fy <= 2 ) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 2,2,   16,16,8>(p);
        else if (s == 1 && fx <= 4  && fy <= 4 ) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 4,4,   16,16,8>(p);
        else if (s == 1 && fx <= 6  && fy <= 6 ) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 6,6,   16,16,8>(p);
        else if (s == 1 && fx <= 8  && fy <= 8 ) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 8,8,   16,16,8>(p);
        else if (s == 1 && fx <= 16 && fy <= 16) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 16,16, 32,32,1>(p);
        else if (s == 1 && fx <= 24 && fy <= 24) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 24,24, 32,32,1>(p);
        // contiguous
        else if (s != 1 && fx <= 2  && fy <= 2 ) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 2,2,   64,16,1>(p);
        else if (s != 1 && fx <= 4  && fy <= 4 ) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 4,4,   64,16,1>(p);
        else if (s != 1 && fx <= 6  && fy <= 6 ) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 6,6,   64,16,1>(p);
        else if (s != 1 && fx <= 8  && fy <= 8 ) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 8,8,   64,16,1>(p);
        else if (s != 1 && fx <= 16 && fy <= 16) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 16,16, 64,32,1>(p);
        else if (s != 1 && fx <= 24 && fy <= 24) run_upfirdn2d_kernel_small<T, 2,2, 1,1, 24,24, 64,32,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }

    // No up/downsampling.
    else if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1)
    {
        // channels_last
        if (s == 1 && fx <= 1  && fy <= 8 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 1,8,   1,128,16>(p);
        else if (s == 1 && fx <= 1  && fy <= 16) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 1,16,  1,128,16>(p);
        else if (s == 1 && fx <= 1  && fy <= 24) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 1,24,  1,128,16>(p);
        else if (s == 1 && fx <= 8  && fy <= 1 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 8,1,   128,1,16>(p);
        else if (s == 1 && fx <= 16 && fy <= 1 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 16,1,  128,1,16>(p);
        else if (s == 1 && fx <= 24 && fy <= 1 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 24,1,  128,1,16>(p);
        else if (s == 1 && fx <= 3  && fy <= 3 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 3,3,   16,16,8>(p);
        else if (s == 1 && fx <= 4  && fy <= 4 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 4,4,   16,16,8>(p);
        else if (s == 1 && fx <= 5  && fy <= 5 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 5,5,   16,16,8>(p);
        else if (s == 1 && fx <= 6  && fy <= 6 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 6,6,   16,16,8>(p);
        else if (s == 1 && fx <= 7  && fy <= 7 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 7,7,   16,16,8>(p);
        else if (s == 1 && fx <= 16 && fy <= 16) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 16,16, 32,32,1>(p);
        else if (s == 1 && fx <= 24 && fy <= 24) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 24,24, 32,32,1>(p);
        // contiguous
        else if (s != 1 && fx <= 1  && fy <= 8 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 1,8,   32,32,1>(p);
        else if (s != 1 && fx <= 1  && fy <= 16) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 1,16,  32,32,1>(p);
        else if (s != 1 && fx <= 1  && fy <= 24) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 1,24,  32,32,1>(p);
        else if (s != 1 && fx <= 8  && fy <= 1 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 8,1,   128,8,1>(p);
        else if (s != 1 && fx <= 16 && fy <= 1 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 16,1,  128,8,1>(p);
        else if (s != 1 && fx <= 24 && fy <= 1 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 24,1,  128,8,1>(p);
        else if (s != 1 && fx <= 3  && fy <= 3 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 3,3,   64,16,1>(p);
        else if (s != 1 && fx <= 4  && fy <= 4 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 4,4,   64,16,1>(p);
        else if (s != 1 && fx <= 5  && fy <= 5 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 5,5,   64,16,1>(p);
        else if (s != 1 && fx <= 6  && fy <= 6 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 6,6,   64,16,1>(p);
        else if (s != 1 && fx <= 7  && fy <= 7 ) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 7,7,   64,16,1>(p);
        else if (s != 1 && fx <= 16 && fy <= 16) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 16,16, 64,32,1>(p);
        else if (s != 1 && fx <= 24 && fy <= 24) run_upfirdn2d_kernel_small<T, 1,1, 1,1, 24,24, 64,32,1>(p);
        else throw std::runtime_error("unexpected kernel parameters");
    }

    // fallback to the large kernel

    else if (s == 1) run_upfirdn2d_kernel_large<T>(p, -1,-1,4, 1); // channels_last
    else run_upfirdn2d_kernel_large<T>(p, -1,-1,1, 4); // contiguous
}

static torch::Tensor upfirdn2d(torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(f.device() == x.device(), "f must reside on the same device as x");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
    TORCH_CHECK(f.numel() <= INT_MAX, "f is too large");
    TORCH_CHECK(x.numel() > 0, "x has zero size");
    TORCH_CHECK(f.numel() > 0, "f has zero size");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(f.dim() == 2, "f must be rank 2");
    TORCH_CHECK((x.size(0)-1)*x.stride(0) + (x.size(1)-1)*x.stride(1) + (x.size(2)-1)*x.stride(2) + (x.size(3)-1)*x.stride(3) <= INT_MAX, "x memory footprint is too large");
    TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1, "f must be at least 1x1");
    TORCH_CHECK(upx >= 1 && upy >= 1, "upsampling factor must be at least 1");
    TORCH_CHECK(downx >= 1 && downy >= 1, "downsampling factor must be at least 1");

    // Create output tensor.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    int outW = ((int)x.size(3) * upx + padx0 + padx1 - (int)f.size(1) + downx) / downx;
    int outH = ((int)x.size(2) * upy + pady0 + pady1 - (int)f.size(0) + downy) / downy;
    TORCH_CHECK(outW >= 1 && outH >= 1, "output must be at least 1x1");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), outH, outW}, x.options(), x.suggest_memory_format());
    TORCH_CHECK(y.numel() <= INT_MAX, "output is too large");
    TORCH_CHECK((y.size(0)-1)*y.stride(0) + (y.size(1)-1)*y.stride(1) + (y.size(2)-1)*y.stride(2) + (y.size(3)-1)*y.stride(3) <= INT_MAX, "output memory footprint is too large");

    // Initialize CUDA kernel parameters.
    upfirdn2d_kernel_params p;
    p.x             = x.data_ptr();
    p.f             = f.data_ptr<float>();
    p.y             = y.data_ptr();
    p.up            = make_int2(upx, upy);
    p.down          = make_int2(downx, downy);
    p.pad0          = make_int2(padx0, pady0);
    p.flip          = (flip) ? 1 : 0;
    p.gain          = gain;
    p.inSize        = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.inStride      = make_int4((int)x.stride(3), (int)x.stride(2), (int)x.stride(1), (int)x.stride(0));
    p.filterSize    = make_int2((int)f.size(1), (int)f.size(0));
    p.filterStride  = make_int2((int)f.stride(1), (int)f.stride(0));
    p.outSize       = make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
    p.outStride     = make_int4((int)y.stride(3), (int)y.stride(2), (int)y.stride(1), (int)y.stride(0));
    p.sizeMajor     = (p.inStride.z == 1) ? p.inSize.w : p.inSize.w * p.inSize.z;
    p.sizeMinor     = (p.inStride.z == 1) ? p.inSize.z : 1;

    // Choose CUDA kernel.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&]
    {
        choose_and_run_upfirdn2d_kernel<scalar_t>(p);
    });

    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("upfirdn2d", &upfirdn2d);
}

//------------------------------------------------------------------------
