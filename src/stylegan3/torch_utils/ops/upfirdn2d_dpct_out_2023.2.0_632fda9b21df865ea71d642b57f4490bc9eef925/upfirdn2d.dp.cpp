// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "upfirdn2d.h"
#include <ipex.h>
#include <c10/util/Half.h>

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };

static __dpct_inline__ int floor_div(int a, int b)
{
    int t = 1 - a / b;
    return (a + t * b) / b - t;
}

static void update_params(upfirdn2d_kernel_params& p, int loopMinor, int loopX) {
    // Set looping options.
    p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1;
    p.loopMinor     = loopMinor;
    p.loopX         = loopX;
    p.launchMinor   = (p.sizeMinor - 1) / loopMinor + 1;
    p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1;
}

//------------------------------------------------------------------------
// Generic CUDA implementation for large filters.

template <class T> static void upfirdn2d_kernel_large(upfirdn2d_kernel_params p,
                                                      const sycl::nd_item<3> &item_ct1)
{
    typedef typename InternalType<T>::scalar_t scalar_t;

    // Calculate thread index.
    int minorBase = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                    item_ct1.get_local_id(0);
    int outY = minorBase / p.launchMinor;
    minorBase -= outY * p.launchMinor;
    int outXBase =
        item_ct1.get_group(1) * p.loopX * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
    int majorBase = item_ct1.get_group(2) * p.loopMajor;
    if (outXBase >= p.outSize.x() | outY >= p.outSize.y() |
        majorBase >= p.sizeMajor)
        return;

    // Setup Y receptive field.
    int midY = outY * p.down.y() + p.up.y() - 1 - p.pad0.y();
    int inY = sycl::min(sycl::max(floor_div(midY, p.up.y()), 0), p.inSize.y());
    int h =
        sycl::min(sycl::max(floor_div(midY + p.filterSize.y(), p.up.y()), 0),
                  p.inSize.y()) -
        inY;
    int filterY = midY + p.filterSize.y() - (inY + 1) * p.up.y();
    if (p.flip)
        filterY = p.filterSize.y() - 1 - filterY;

    // Loop over major, minor, and X.
    for (int majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    for (int minorIdx = 0, minor = minorBase; minorIdx < p.loopMinor & minor < p.sizeMinor; minorIdx++, minor += p.launchMinor)
    {
        int nc = major * p.sizeMinor + minor;
        int n = nc / p.inSize.z();
        int c = nc - n * p.inSize.z();
        for (int loopX = 0, outX = outXBase;
             loopX < p.loopX & outX < p.outSize.x();
             loopX++, outX += item_ct1.get_local_range(1))
        {
            // Setup X receptive field.
            int midX = outX * p.down.x() + p.up.x() - 1 - p.pad0.x();
            int inX = sycl::min(sycl::max(floor_div(midX, p.up.x()), 0),
                                p.inSize.x());
            int w =
                sycl::min(
                    sycl::max(floor_div(midX + p.filterSize.x(), p.up.x()), 0),
                    p.inSize.x()) -
                inX;
            int filterX = midX + p.filterSize.x() - (inX + 1) * p.up.x();
            if (p.flip)
                filterX = p.filterSize.x() - 1 - filterX;

            // Initialize pointers.
            const T *xp =
                &((const T *)p.x)[inX * p.inStride.x() + inY * p.inStride.y() +
                                  c * p.inStride.z() + n * p.inStride.w()];
            const float *fp = &p.f[filterX * p.filterStride.x() +
                                   filterY * p.filterStride.y()];
            int filterStepX =
                ((p.flip) ? p.up.x() : -p.up.x()) * p.filterStride.x();
            int filterStepY =
                ((p.flip) ? p.up.y() : -p.up.y()) * p.filterStride.y();

            // Inner loop.
            scalar_t v = 0;
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    v += (scalar_t)(*xp) * (scalar_t)(*fp);
                    xp += p.inStride.x();
                    fp += filterStepX;
                }
                xp += p.inStride.y() - w * p.inStride.x();
                fp += filterStepY - w * filterStepX;
            }

            // Store result.
            v *= p.gain;
            ((T *)p.y)[outX * p.outStride.x() + outY * p.outStride.y() +
                       c * p.outStride.z() + n * p.outStride.w()] = 
                       (T)v;
        }
    }
}

//------------------------------------------------------------------------
// Specialized CUDA implementation for small filters.

template <class T, int upx, int upy, int downx, int downy, int filterW,
          int filterH, int tileOutW, int tileOutH, int loopMinor>
/*
DPCT1110:2: The total declared local variable size in device function
upfirdn2d_kernel_small exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
static void
upfirdn2d_kernel_small(upfirdn2d_kernel_params p,
                       const sycl::nd_item<3> &item_ct1,
                       sycl::local_accessor<T, 2> sf,
                       sycl::local_accessor<T, 3> sx)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    const int tileInW = ((tileOutW - 1) * downx + filterW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + filterH - 1) / upy + 1;

    // Calculate tile index.
    int minorBase = item_ct1.get_group(0);
    int tileOutY = minorBase / p.launchMinor;
    minorBase -= tileOutY * p.launchMinor;
    minorBase *= loopMinor;
    tileOutY *= tileOutH;
    int tileOutXBase = item_ct1.get_group(1) * p.loopX * tileOutW;
    int majorBase = item_ct1.get_group(2) * p.loopMajor;
    if (tileOutXBase >= p.outSize.x() | tileOutY >= p.outSize.y() |
        majorBase >= p.sizeMajor)
        return;

    // Load filter (flipped).
    for (int tapIdx = item_ct1.get_local_id(0); tapIdx < filterH * filterW;
         tapIdx += item_ct1.get_local_range(0))
    {
        int fy = tapIdx / filterW;
        int fx = tapIdx - fy * filterW;
        scalar_t v = 0;
        if (fx < p.filterSize.x() & fy < p.filterSize.y())
        {
            int ffx = (p.flip) ? fx : p.filterSize.x() - 1 - fx;
            int ffy = (p.flip) ? fy : p.filterSize.y() - 1 - fy;
            v = (scalar_t)
                    p.f[ffx * p.filterStride.x() + ffy * p.filterStride.y()];
        }
        sf[fy][fx] = v;
    }

    // Loop over major and X.
    for (int majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    {
        int baseNC = major * p.sizeMinor + minorBase;
        int n = baseNC / p.inSize.z();
        int baseC = baseNC - n * p.inSize.z();
        for (int loopX = 0, tileOutX = tileOutXBase;
             loopX < p.loopX & tileOutX < p.outSize.x();
             loopX++, tileOutX += tileOutW)
        {
            // Load input pixels.
            int tileMidX = tileOutX * downx + upx - 1 - p.pad0.x();
            int tileMidY = tileOutY * downy + upy - 1 - p.pad0.y();
            int tileInX = floor_div(tileMidX, upx);
            int tileInY = floor_div(tileMidY, upy);

            item_ct1.barrier(sycl::access::fence_space::local_space);
            for (int inIdx = item_ct1.get_local_id(0);
                 inIdx < tileInH * tileInW * loopMinor;
                 inIdx += item_ct1.get_local_range(0))
            {
                int relC = inIdx;
                int relInX = relC / loopMinor;
                int relInY = relInX / tileInW;
                relC -= relInX * loopMinor;
                relInX -= relInY * tileInW;
                int c = baseC + relC;
                int inX = tileInX + relInX;
                int inY = tileInY + relInY;
                scalar_t v = 0;
                if (inX >= 0 & inY >= 0 & inX < p.inSize.x() &
                    inY < p.inSize.y() & c < p.inSize.z())
                    v = (scalar_t)((const T *)p.x)[inX * p.inStride.x() +
                                                   inY * p.inStride.y() +
                                                   c * p.inStride.z() +
                                                   n * p.inStride.w()];
                sx[relInY][relInX][relC] = v;
            }

            // Loop over output pixels.
            item_ct1.barrier(sycl::access::fence_space::local_space);
            for (int outIdx = item_ct1.get_local_id(0);
                 outIdx < tileOutH * tileOutW * loopMinor;
                 outIdx += item_ct1.get_local_range(0))
            {
                int relC = outIdx;
                int relOutX = relC / loopMinor;
                int relOutY = relOutX / tileOutW;
                relC -= relOutX * loopMinor;
                relOutX -= relOutY * tileOutW;
                int c = baseC + relC;
                int outX = tileOutX + relOutX;
                int outY = tileOutY + relOutY;

                // Setup receptive field.
                int midX = tileMidX + relOutX * downx;
                int midY = tileMidY + relOutY * downy;
                int inX = floor_div(midX, upx);
                int inY = floor_div(midY, upy);
                int relInX = inX - tileInX;
                int relInY = inY - tileInY;
                int filterX = (inX + 1) * upx - midX - 1; // flipped
                int filterY = (inY + 1) * upy - midY - 1; // flipped

                // Inner loop.
                if (outX < p.outSize.x() & outY < p.outSize.y() &
                    c < p.outSize.z())
                {
                    scalar_t v = 0;
                    #pragma unroll
                    for (int y = 0; y < filterH / upy; y++)
                        #pragma unroll
                        for (int x = 0; x < filterW / upx; x++)
                            v += sx[relInY + y][relInX + x][relC] * sf[filterY + y * upy][filterX + x * upx];
                    v *= p.gain;
                    ((T *)p.y)[outX * p.outStride.x() + outY * p.outStride.y() +
                               c * p.outStride.z() + n * p.outStride.w()] = (T)v;
                }
            }
        }
    }
}

//------------------------------------------------------------------------
// Helper functions for launching the kernels.

template <class T, int upx, int upy, int downx, int downy, int filterW, int filterH, int tileOutW, int tileOutH, int loopMinor>
void run_upfirdn2d_kernel_small(upfirdn2d_kernel_params p) {
    update_params(p, loopMinor, 1);
    
    // Compute grid size - for small kernels
    sycl::range<3> blockSize = sycl::range<3>(256, 1, 1);
    sycl::range<3> gridSize = sycl::range<3>(
        ((p.outSize.y() - 1) / tileOutH + 1) * p.launchMinor,
        (p.outSize.x() - 1) / (tileOutW * p.loopX) + 1, p.launchMajor);

    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
    // tileInW/H computation copied from inside the kernel (needed for accessor definition here)
    const int tileInW = ((tileOutW - 1) * downx + filterW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + filterH - 1) / upy + 1;

    auto device_type = c10::DeviceType::XPU;
    c10::impl::VirtualGuardImpl impl(device_type);
    c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
    auto& queue = xpu::get_queue_from_stream(c10_stream);
    
    queue.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<T, 2> sf_acc_ct1(
              sycl::range<2>(filterH, filterW), cgh);
          sycl::local_accessor<T, 3> sx_acc_ct1(
              sycl::range<3>(tileInH, tileInW, loopMinor), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(gridSize * blockSize, blockSize),
              [=](sycl::nd_item<3> item_ct1) {
                upfirdn2d_kernel_small<T, upx, upy, downx, downy, filterW,
                                       filterH, tileOutW, tileOutH, loopMinor>(
                    p, item_ct1, sf_acc_ct1, sx_acc_ct1);
              });
        }).wait();
  }
}

template <class T>
void run_upfirdn2d_kernel_large(upfirdn2d_kernel_params p, int tileOutW, int tileOutH, int loopMinor, int loopX) {
    update_params(p, loopMinor, loopX);
    
    // Compute grid size - for large kernels
    sycl::range<3> blockSize = sycl::range<3>(4, 32, 1);
    sycl::range<3> gridSize = sycl::range<3>(
        ((p.outSize.y() - 1) / blockSize[2] + 1) * p.launchMinor,
        (p.outSize.x() - 1) / (blockSize[1] * p.loopX) + 1,
        p.launchMajor);

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {

    auto device_type = c10::DeviceType::XPU;
    c10::impl::VirtualGuardImpl impl(device_type);
    c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
    auto& queue = xpu::get_queue_from_stream(c10_stream);

    queue.submit([&](sycl::handler &cgh) {

          cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
                           [=](sycl::nd_item<3> item_ct1) {
                             upfirdn2d_kernel_large<T>(p, item_ct1);
                           });
        }).wait();
  }
}

//------------------------------------------------------------------------
// Template specializations.

// "large" kernel specializations
template void run_upfirdn2d_kernel_large<double>(upfirdn2d_kernel_params p, int tileOutW, int tileOutH, int loopMinor, int loopX);
template void run_upfirdn2d_kernel_large<float>(upfirdn2d_kernel_params p, int tileOutW, int tileOutH, int loopMinor, int loopX);
template void run_upfirdn2d_kernel_large<c10::Half>(upfirdn2d_kernel_params p, int tileOutW, int tileOutH, int loopMinor, int loopX);


#define SPEC_with_type(...) \
    template void run_upfirdn2d_kernel_small<__VA_ARGS__>(upfirdn2d_kernel_params p);

#define SPEC(...) \
    SPEC_with_type(double, __VA_ARGS__) \
    SPEC_with_type(float, __VA_ARGS__) \
    SPEC_with_type(c10::Half, __VA_ARGS__)

// Instead of writing full specializations for all the variants of the "small" kernel and it data type (~300 difficult-to-read lines like this):
//   template void run_upfirdn2d_kernel_small<double, 1, 1, 1, 4, 1, 48, 32, 8, 1>(upfirdn2d_kernel_params p);
//   template void run_upfirdn2d_kernel_small<float, 1, 1, 1, 4, 1, 48, 32, 8, 1>(upfirdn2d_kernel_params p);
//   template void run_upfirdn2d_kernel_small<c10::Half, 1, 1, 1, 4, 1, 48, 32, 8, 1>(upfirdn2d_kernel_params p);
// we can just write "SPEC(params)", e.g. "SPEC(1, 1, 1, 4, 1, 48, 32, 8, 1)" to make specializations for all types of one kernel variation with a single line.
// These lines can be generated automatically from the `.cpp` file from where the `run_upfirdn2d_kernel_small<...>` functions are called, using the following command:
//   grep 'run_upfirdn2d_kernel_small<T, .*>(p)' torch_utils/ops/upfirdn2d.cpp | sed 's/.*run_upfirdn2d_kernel_small<T, *\(.*\)>.*/SPEC(\1)/'

SPEC(1,1, 1,4, 1,32, 1,32,8)
SPEC(1,1, 1,4, 1,48, 1,32,8)
SPEC(1,1, 1,4, 1,32, 32,8,1)
SPEC(1,1, 1,4, 1,48, 32,8,1)
SPEC(1,1, 4,1, 32,1, 32,1,8)
SPEC(1,1, 4,1, 48,1, 32,1,8)
SPEC(1,1, 4,1, 32,1, 32,8,1)
SPEC(1,1, 4,1, 48,1, 32,8,1)
SPEC(1,4, 1,1, 1,32, 1,128,16)
SPEC(1,4, 1,1, 1,48, 1,128,16)
SPEC(1,4, 1,1, 1,32, 32,32,1)
SPEC(1,4, 1,1, 1,48, 32,32,1)
SPEC(4,1, 1,1, 32,1, 128,1,16)
SPEC(4,1, 1,1, 48,1, 128,1,16)
SPEC(4,1, 1,1, 32,1, 128,8,1)
SPEC(4,1, 1,1, 48,1, 128,8,1)
SPEC(4,4, 1,1, 32,32, 32,32,1)
SPEC(4,4, 1,1, 48,48, 32,32,1)
SPEC(4,4, 1,1, 32,32, 64,32,1)
SPEC(4,4, 1,1, 48,48, 64,32,1)
SPEC(1,1, 1,2, 1,8,  1,64,8)
SPEC(1,1, 1,2, 1,16, 1,64,8)
SPEC(1,1, 1,2, 1,24, 1,64,8)
SPEC(1,1, 1,2, 1,8,  32,16,1)
SPEC(1,1, 1,2, 1,16, 32,16,1)
SPEC(1,1, 1,2, 1,24, 32,16,1)
SPEC(1,1, 2,1, 8,1,  64,1,8)
SPEC(1,1, 2,1, 16,1, 64,1,8)
SPEC(1,1, 2,1, 24,1, 64,1,8)
SPEC(1,1, 2,1, 8,1,  64,8,1)
SPEC(1,1, 2,1, 16,1, 64,8,1)
SPEC(1,1, 2,1, 24,1, 64,8,1)
SPEC(1,1, 2,2, 2,2,   8,8,8)
SPEC(1,1, 2,2, 4,4,   8,8,8)
SPEC(1,1, 2,2, 6,6,   8,8,8)
SPEC(1,1, 2,2, 8,8,   8,8,8)
SPEC(1,1, 2,2, 16,16, 16,16,1)
SPEC(1,1, 2,2, 24,24, 16,16,1)
SPEC(1,1, 2,2, 2,2,   32,8,1)
SPEC(1,1, 2,2, 4,4,   32,8,1)
SPEC(1,1, 2,2, 6,6,   32,8,1)
SPEC(1,1, 2,2, 8,8,   32,8,1)
SPEC(1,1, 2,2, 16,16, 32,16,1)
SPEC(1,1, 2,2, 24,24, 32,16,1)
SPEC(1,2, 1,1, 1,8,  1,128,16)
SPEC(1,2, 1,1, 1,16, 1,128,16)
SPEC(1,2, 1,1, 1,24, 1,128,16)
SPEC(1,2, 1,1, 1,8,  32,32,1)
SPEC(1,2, 1,1, 1,16, 32,32,1)
SPEC(1,2, 1,1, 1,24, 32,32,1)
SPEC(2,1, 1,1, 8,1,  128,1,16)
SPEC(2,1, 1,1, 16,1, 128,1,16)
SPEC(2,1, 1,1, 24,1, 128,1,16)
SPEC(2,1, 1,1, 8,1,  128,8,1)
SPEC(2,1, 1,1, 16,1, 128,8,1)
SPEC(2,1, 1,1, 24,1, 128,8,1)
SPEC(2,2, 1,1, 2,2,   16,16,8)
SPEC(2,2, 1,1, 4,4,   16,16,8)
SPEC(2,2, 1,1, 6,6,   16,16,8)
SPEC(2,2, 1,1, 8,8,   16,16,8)
SPEC(2,2, 1,1, 16,16, 32,32,1)
SPEC(2,2, 1,1, 24,24, 32,32,1)
SPEC(2,2, 1,1, 2,2,   64,16,1)
SPEC(2,2, 1,1, 4,4,   64,16,1)
SPEC(2,2, 1,1, 6,6,   64,16,1)
SPEC(2,2, 1,1, 8,8,   64,16,1)
SPEC(2,2, 1,1, 16,16, 64,32,1)
SPEC(2,2, 1,1, 24,24, 64,32,1)
SPEC(1,1, 1,1, 1,8,   1,128,16)
SPEC(1,1, 1,1, 1,16,  1,128,16)
SPEC(1,1, 1,1, 1,24,  1,128,16)
SPEC(1,1, 1,1, 8,1,   128,1,16)
SPEC(1,1, 1,1, 16,1,  128,1,16)
SPEC(1,1, 1,1, 24,1,  128,1,16)
SPEC(1,1, 1,1, 3,3,   16,16,8)
SPEC(1,1, 1,1, 4,4,   16,16,8)
SPEC(1,1, 1,1, 5,5,   16,16,8)
SPEC(1,1, 1,1, 6,6,   16,16,8)
SPEC(1,1, 1,1, 7,7,   16,16,8)
SPEC(1,1, 1,1, 16,16, 32,32,1)
SPEC(1,1, 1,1, 24,24, 32,32,1)
SPEC(1,1, 1,1, 1,8,   32,32,1)
SPEC(1,1, 1,1, 1,16,  32,32,1)
SPEC(1,1, 1,1, 1,24,  32,32,1)
SPEC(1,1, 1,1, 8,1,   128,8,1)
SPEC(1,1, 1,1, 16,1,  128,8,1)
SPEC(1,1, 1,1, 24,1,  128,8,1)
SPEC(1,1, 1,1, 3,3,   64,16,1)
SPEC(1,1, 1,1, 4,4,   64,16,1)
SPEC(1,1, 1,1, 5,5,   64,16,1)
SPEC(1,1, 1,1, 6,6,   64,16,1)
SPEC(1,1, 1,1, 7,7,   64,16,1)
SPEC(1,1, 1,1, 16,16, 64,32,1)
SPEC(1,1, 1,1, 24,24, 64,32,1)

//------------------------------------------------------------------------
