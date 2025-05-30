// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "filtered_lrelu.h"


template <class T, class index_t, bool signWrite, bool signRead>
void choose_and_run_filtered_lrelu_kernel(filtered_lrelu_kernel_params& p, int sharedKB)
{
    // Run the first matching kernel.
#define CASE(SH, U, FU, D, FD, MODE, TW, TH, W, XR, WS) \
    if (sharedKB >= SH) \
    if ((p.fuShape.y == 0 && (MODE == MODE_SUSD || MODE == MODE_SUFD)) || (p.fuShape.y > 0 && (MODE == MODE_FUSD || MODE == MODE_FUFD))) \
    if ((p.fdShape.y == 0 && (MODE == MODE_SUSD || MODE == MODE_FUSD)) || (p.fdShape.y > 0 && (MODE == MODE_SUFD || MODE == MODE_FUFD))) \
    if (p.up == U && p.fuShape.x <= FU && p.fuShape.y <= FU && p.down == D && p.fdShape.x <= FD && p.fdShape.y <= FD) \
    { \
        static_assert((D*TW % 4) == 0, "down * tileWidth must be divisible by 4"); \
        static_assert(FU % U == 0, "upscaling filter size must be multiple of upscaling factor"); \
        static_assert(FD % D == 0, "downscaling filter size must be multiple of downscaling factor"); \
 \
        run_filtered_lrelu_kernel<T, index_t, signWrite, signRead, SH, MODE, U, FU, D, FD, TW, TH, W, XR, WS>(p); \
        return; \
    }

    // Launch parameters for various kernel specializations.
    // Small filters must be listed before large filters, otherwise the kernel for larger filter will always match first.
    // Kernels that use more shared memory must be listed before those that use less, for the same reason.

#include "filtered_lrelu_cases.h"

    #undef CASE

    // TODO throw - no kernel found
    TORCH_CHECK(false, "no kernel found")
    return;
}

//------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor, int> filtered_lrelu(
    torch::Tensor x, torch::Tensor fu, torch::Tensor fd, torch::Tensor b, torch::Tensor si,
    int up, int down, int px0, int px1, int py0, int py1, int sx, int sy, float gain, float slope, float clamp, bool flip_filters, bool writeSigns)
{
    // Set CUDA device.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    // Validate arguments.
    TORCH_CHECK(fu.device() == x.device() && fd.device() == x.device() && b.device() == x.device(), "all input tensors must reside on the same device");
    TORCH_CHECK(fu.dtype() == torch::kFloat && fd.dtype() == torch::kFloat, "fu and fd must be float32");
    TORCH_CHECK(b.dtype() == x.dtype(), "x and b must have the same dtype");
    TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kFloat, "x and b must be float16 or float32");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(x.size(0) * x.size(1) <= INT_MAX && x.size(2) <= INT_MAX && x.size(3) <= INT_MAX, "x is too large");
    TORCH_CHECK(x.numel() > 0, "x is empty");
    TORCH_CHECK((fu.dim() == 1 || fu.dim() == 2) && (fd.dim() == 1 || fd.dim() == 2), "fu and fd must be rank 1 or 2");
    TORCH_CHECK(fu.size(0) <= INT_MAX && fu.size(-1) <= INT_MAX, "fu is too large");
    TORCH_CHECK(fd.size(0) <= INT_MAX && fd.size(-1) <= INT_MAX, "fd is too large");
    TORCH_CHECK(fu.numel() > 0, "fu is empty");
    TORCH_CHECK(fd.numel() > 0, "fd is empty");
    TORCH_CHECK(b.dim() == 1 && b.size(0) == x.size(1), "b must be a vector with the same number of channels as x");
    TORCH_CHECK(up >= 1 && down >= 1, "up and down must be at least 1");

    // Figure out how much shared memory is available on the device.
    int maxSharedBytes = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, x.device().index()));
    int sharedKB = maxSharedBytes >> 10;

    // Populate enough launch parameters to check if a CUDA kernel exists.
    filtered_lrelu_kernel_params p;
    p.up      = up;
    p.down    = down;
    p.fuShape = make_int2((int)fu.size(-1), fu.dim() == 2 ? (int)fu.size(0) : 0); // shape [n, 0] indicates separable filter.
    p.fdShape = make_int2((int)fd.size(-1), fd.dim() == 2 ? (int)fd.size(0) : 0);

    // Input/output element size.
    int64_t sz = (x.dtype() == torch::kHalf) ? 2 : 4;

    // Input sizes.
    int64_t xw = (int)x.size(3);
    int64_t xh = (int)x.size(2);
    int64_t fut_w = (int)fu.size(-1) - 1;
    int64_t fut_h = (int)fu.size(0)  - 1;
    int64_t fdt_w = (int)fd.size(-1) - 1;
    int64_t fdt_h = (int)fd.size(0)  - 1;

    // Logical size of upsampled buffer.
    int64_t cw = xw * up + (px0 + px1) - fut_w;
    int64_t ch = xh * up + (py0 + py1) - fut_h;
    TORCH_CHECK(cw > fdt_w && ch > fdt_h, "upsampled buffer must be at least the size of downsampling filter");
    TORCH_CHECK(cw <= INT_MAX && ch <= INT_MAX, "upsampled buffer is too large");

    // Compute output size and allocate.
    int64_t yw = (cw - fdt_w + (down - 1)) / down;
    int64_t yh = (ch - fdt_h + (down - 1)) / down;
    TORCH_CHECK(yw > 0 && yh > 0, "output must be at least 1x1");
    TORCH_CHECK(yw <= INT_MAX && yh <= INT_MAX, "output is too large");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), yh, yw}, x.options(), x.suggest_memory_format());

    // Allocate sign tensor.
    torch::Tensor so;
    torch::Tensor s = si;
    bool readSigns = !!s.numel();
    int64_t sw_active = 0; // Active width of sign tensor.
    if (writeSigns)
    {
        sw_active = yw * down - (down - 1) + fdt_w;     // Active width in elements.
        int64_t sh = yh * down - (down - 1) + fdt_h;    // Height = active height.
        int64_t sw = (sw_active + 15) & ~15;            // Width  = active width in elements, rounded up to multiple of 16.
        TORCH_CHECK(sh <= INT_MAX && (sw >> 2) <= INT_MAX, "signs is too large");
        s = so = torch::empty({x.size(0), x.size(1), sh, sw >> 2}, x.options().dtype(torch::kUInt8), at::MemoryFormat::Contiguous);
    }
    else if (readSigns)
        sw_active = s.size(3) << 2;

    // Validate sign tensor if in use.
    if (readSigns || writeSigns)
    {
        TORCH_CHECK(s.is_contiguous(), "signs must be contiguous");
        TORCH_CHECK(s.dtype() == torch::kUInt8, "signs must be uint8");
        TORCH_CHECK(s.device() == x.device(), "signs must reside on the same device as x");
        TORCH_CHECK(s.dim() == 4, "signs must be rank 4");
        TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1), "signs must have same batch & channels as x");
        TORCH_CHECK(s.size(2) <= INT_MAX && s.size(3) <= INT_MAX, "signs is too large");
    }

    // Populate rest of CUDA kernel parameters.
    p.x         = x.data_ptr();
    p.y         = y.data_ptr();
    p.b         = b.data_ptr();
    p.s         = (readSigns || writeSigns) ? s.data_ptr<unsigned char>() : 0;
    p.fu        = fu.data_ptr<float>();
    p.fd        = fd.data_ptr<float>();
    p.pad0      = make_int2(px0, py0);
    p.gain      = gain;
    p.slope     = slope;
    p.clamp     = clamp;
    p.flip      = (flip_filters) ? 1 : 0;
    p.xShape    = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.yShape    = make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
    p.sShape    = (readSigns || writeSigns) ? make_int2((int)s.size(3), (int)s.size(2)) : make_int2(0, 0); // Width is in bytes. Contiguous.
    p.sOfs      = make_int2(sx, sy);
    p.swLimit   = (sw_active + 3) >> 2; // Rounded up to bytes.

    // x, y, b strides are in bytes.
    p.xStride   = make_longlong4(sz * x.stride(3), sz * x.stride(2), sz * x.stride(1), sz * x.stride(0));
    p.yStride   = make_longlong4(sz * y.stride(3), sz * y.stride(2), sz * y.stride(1), sz * y.stride(0));
    p.bStride   = sz * b.stride(0);

    // fu, fd strides are in elements.
    p.fuStride  = make_longlong3(fu.stride(-1), fu.dim() == 2 ? fu.stride(0) : 0, 0);
    p.fdStride  = make_longlong3(fd.stride(-1), fd.dim() == 2 ? fd.stride(0) : 0, 0);

    // Determine if indices don't fit in int32. Support negative strides although Torch currently never produces those.
    bool index64b = false;
    if (std::abs(p.bStride * x.size(1)) > INT_MAX) index64b = true;
    if (std::min(x.size(0) * p.xStride.w, 0ll) + std::min(x.size(1) * p.xStride.z, 0ll) + std::min(x.size(2) * p.xStride.y, 0ll) + std::min(x.size(3) * p.xStride.x, 0ll) < -INT_MAX) index64b = true;
    if (std::max(x.size(0) * p.xStride.w, 0ll) + std::max(x.size(1) * p.xStride.z, 0ll) + std::max(x.size(2) * p.xStride.y, 0ll) + std::max(x.size(3) * p.xStride.x, 0ll) >  INT_MAX) index64b = true;
    if (std::min(y.size(0) * p.yStride.w, 0ll) + std::min(y.size(1) * p.yStride.z, 0ll) + std::min(y.size(2) * p.yStride.y, 0ll) + std::min(y.size(3) * p.yStride.x, 0ll) < -INT_MAX) index64b = true;
    if (std::max(y.size(0) * p.yStride.w, 0ll) + std::max(y.size(1) * p.yStride.z, 0ll) + std::max(y.size(2) * p.yStride.y, 0ll) + std::max(y.size(3) * p.yStride.x, 0ll) >  INT_MAX) index64b = true;
    if (s.numel() > INT_MAX) index64b = true;

    // Choose CUDA kernel.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filtered_lrelu_cuda", [&]
    {
        if constexpr (sizeof(scalar_t) <= 4) // Exclude doubles. constexpr prevents template instantiation.
        {
            // Choose kernel based on index type, datatype and sign read/write modes.
            if      (!index64b &&  writeSigns && !readSigns) choose_and_run_filtered_lrelu_kernel<scalar_t, int32_t, true,  false>(p, sharedKB);
            else if (!index64b && !writeSigns &&  readSigns) choose_and_run_filtered_lrelu_kernel<scalar_t, int32_t, false, true >(p, sharedKB);
            else if (!index64b && !writeSigns && !readSigns) choose_and_run_filtered_lrelu_kernel<scalar_t, int32_t, false, false>(p, sharedKB);
            else if ( index64b &&  writeSigns && !readSigns) choose_and_run_filtered_lrelu_kernel<scalar_t, int64_t, true,  false>(p, sharedKB);
            else if ( index64b && !writeSigns &&  readSigns) choose_and_run_filtered_lrelu_kernel<scalar_t, int64_t, false, true >(p, sharedKB);
            else if ( index64b && !writeSigns && !readSigns) choose_and_run_filtered_lrelu_kernel<scalar_t, int64_t, false, false>(p, sharedKB);
            else TORCH_CHECK(false, "internal error - CUDA kernel not found") // This should not happen because we tested earlier that kernel exists.
        } else TORCH_CHECK(false, "internal error - CUDA kernel not found") // This should not happen because we tested earlier that kernel exists. - maybe not necessary to check?
    });

    // Done.
    return std::make_tuple(y, so, 0);
}

//------------------------------------------------------------------------

static torch::Tensor filtered_lrelu_act(torch::Tensor x, torch::Tensor si, int sx, int sy, float gain, float slope, float clamp, bool writeSigns)
{
    // Set CUDA device.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    // Validate arguments.
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(x.size(0) * x.size(1) <= INT_MAX && x.size(2) <= INT_MAX && x.size(3) <= INT_MAX, "x is too large");
    TORCH_CHECK(x.numel() > 0, "x is empty");
    TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kFloat || x.dtype() == torch::kDouble, "x must be float16, float32 or float64");

    // Output signs if we don't have sign input.
    torch::Tensor so;
    torch::Tensor s = si;
    bool readSigns = !!s.numel();
    if (writeSigns)
    {
        int64_t sw = x.size(3);
        sw = (sw + 15) & ~15; // Round to a multiple of 16 for coalescing.
        s = so = torch::empty({x.size(0), x.size(1), x.size(2), sw >> 2}, x.options().dtype(torch::kUInt8), at::MemoryFormat::Contiguous);
    }

    // Validate sign tensor if in use.
    if (readSigns || writeSigns)
    {
        TORCH_CHECK(s.is_contiguous(), "signs must be contiguous");
        TORCH_CHECK(s.dtype() == torch::kUInt8, "signs must be uint8");
        TORCH_CHECK(s.device() == x.device(), "signs must reside on the same device as x");
        TORCH_CHECK(s.dim() == 4, "signs must be rank 4");
        TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1), "signs must have same batch & channels as x");
        TORCH_CHECK(s.size(2) <= INT_MAX && (s.size(3) << 2) <= INT_MAX, "signs tensor is too large");
    }

    // Initialize CUDA kernel parameters.
    filtered_lrelu_act_kernel_params p;
    p.x         = x.data_ptr();
    p.s         = (readSigns || writeSigns) ? s.data_ptr<unsigned char>() : 0;
    p.gain      = gain;
    p.slope     = slope;
    p.clamp     = clamp;
    p.xShape    = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.xStride   = make_longlong4(x.stride(3), x.stride(2), x.stride(1), x.stride(0));
    p.sShape    = (readSigns || writeSigns) ? make_int2((int)s.size(3) << 2, (int)s.size(2)) : make_int2(0, 0); // Width is in elements. Contiguous.
    p.sOfs      = make_int2(sx, sy);

    // Choose CUDA kernel.
    void* func = 0;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filtered_lrelu_act_cuda", [&]
    {
        if (writeSigns) run_filtered_lrelu_act_kernel<scalar_t, true, false>(p);
        else if (readSigns) run_filtered_lrelu_act_kernel<scalar_t, false, true>(p);
        else run_filtered_lrelu_act_kernel<scalar_t, false, false>(p);
    });

    return so;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("filtered_lrelu",      &filtered_lrelu);      // The whole thing.
    m.def("filtered_lrelu_act_", &filtered_lrelu_act);  // Activation and sign tensor handling only. Modifies data tensor in-place.
}

//------------------------------------------------------------------------
