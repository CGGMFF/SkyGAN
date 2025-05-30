// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <c10/util/Half.h>
#include <ATen/cuda/CUDAContext.h>
#include "upfirdn2d.h"

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };

static __device__ __forceinline__ int floor_div(int a, int b)
{
    int t = 1 - a / b;
    return (a + t * b) / b - t;
}

static void update_params(upfirdn2d_kernel_params& p, int loopMinor, int loopX) {
    // Set looping options.
    p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1; // could be set earlier - does not depend on "spec"
    p.loopMinor     = loopMinor; // "spec" unnecessary - known from kernel's template argument
    p.loopX         = loopX; // "spec" unnecessary - always 1 (except for contiguous large kernel)
    p.launchMinor   = (p.sizeMinor - 1) / loopMinor + 1;
    p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1; // could be set earlier - does not depend on "spec"
}

//------------------------------------------------------------------------
// Generic CUDA implementation for large filters.

template <class T> static __global__ void upfirdn2d_kernel_large(upfirdn2d_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;

    // Calculate thread index.
    int minorBase = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = minorBase / p.launchMinor;
    minorBase -= outY * p.launchMinor;
    int outXBase = blockIdx.y * p.loopX * blockDim.y + threadIdx.y;
    int majorBase = blockIdx.z * p.loopMajor;
    if (outXBase >= p.outSize.x | outY >= p.outSize.y | majorBase >= p.sizeMajor)
        return;

    // Setup Y receptive field.
    int midY = outY * p.down.y + p.up.y - 1 - p.pad0.y;
    int inY = min(max(floor_div(midY, p.up.y), 0), p.inSize.y);
    int h = min(max(floor_div(midY + p.filterSize.y, p.up.y), 0), p.inSize.y) - inY;
    int filterY = midY + p.filterSize.y - (inY + 1) * p.up.y;
    if (p.flip)
        filterY = p.filterSize.y - 1 - filterY;

    // Loop over major, minor, and X.
    for (int majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    for (int minorIdx = 0, minor = minorBase; minorIdx < p.loopMinor & minor < p.sizeMinor; minorIdx++, minor += p.launchMinor)
    {
        int nc = major * p.sizeMinor + minor;
        int n = nc / p.inSize.z;
        int c = nc - n * p.inSize.z;
        for (int loopX = 0, outX = outXBase; loopX < p.loopX & outX < p.outSize.x; loopX++, outX += blockDim.y)
        {
            // Setup X receptive field.
            int midX = outX * p.down.x + p.up.x - 1 - p.pad0.x;
            int inX = min(max(floor_div(midX, p.up.x), 0), p.inSize.x);
            int w = min(max(floor_div(midX + p.filterSize.x, p.up.x), 0), p.inSize.x) - inX;
            int filterX = midX + p.filterSize.x - (inX + 1) * p.up.x;
            if (p.flip)
                filterX = p.filterSize.x - 1 - filterX;

            // Initialize pointers.
            const T* xp = &((const T*)p.x)[inX * p.inStride.x + inY * p.inStride.y + c * p.inStride.z + n * p.inStride.w];
            const float* fp = &p.f[filterX * p.filterStride.x + filterY * p.filterStride.y];
            int filterStepX = ((p.flip) ? p.up.x : -p.up.x) * p.filterStride.x;
            int filterStepY = ((p.flip) ? p.up.y : -p.up.y) * p.filterStride.y;

            // Inner loop.
            scalar_t v = 0;
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    v += (scalar_t)(*xp) * (scalar_t)(*fp);
                    xp += p.inStride.x;
                    fp += filterStepX;
                }
                xp += p.inStride.y - w * p.inStride.x;
                fp += filterStepY - w * filterStepX;
            }

            // Store result.
            v *= p.gain;
            ((T*)p.y)[outX * p.outStride.x + outY * p.outStride.y + c * p.outStride.z + n * p.outStride.w] = (T)v;
        }
    }
}

//------------------------------------------------------------------------
// Specialized CUDA implementation for small filters.

template <class T, int upx, int upy, int downx, int downy, int filterW, int filterH, int tileOutW, int tileOutH, int loopMinor>
static __global__ void upfirdn2d_kernel_small(upfirdn2d_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    const int tileInW = ((tileOutW - 1) * downx + filterW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + filterH - 1) / upy + 1;
    __shared__ volatile scalar_t sf[filterH][filterW];
    __shared__ volatile scalar_t sx[tileInH][tileInW][loopMinor];

    // Calculate tile index.
    int minorBase = blockIdx.x;
    int tileOutY = minorBase / p.launchMinor;
    minorBase -= tileOutY * p.launchMinor;
    minorBase *= loopMinor;
    tileOutY *= tileOutH;
    int tileOutXBase = blockIdx.y * p.loopX * tileOutW;
    int majorBase = blockIdx.z * p.loopMajor;
    if (tileOutXBase >= p.outSize.x | tileOutY >= p.outSize.y | majorBase >= p.sizeMajor)
        return;

    // Load filter (flipped).
    for (int tapIdx = threadIdx.x; tapIdx < filterH * filterW; tapIdx += blockDim.x)
    {
        int fy = tapIdx / filterW;
        int fx = tapIdx - fy * filterW;
        scalar_t v = 0;
        if (fx < p.filterSize.x & fy < p.filterSize.y)
        {
            int ffx = (p.flip) ? fx : p.filterSize.x - 1 - fx;
            int ffy = (p.flip) ? fy : p.filterSize.y - 1 - fy;
            v = (scalar_t)p.f[ffx * p.filterStride.x + ffy * p.filterStride.y];
        }
        sf[fy][fx] = v;
    }

    // Loop over major and X.
    for (int majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    {
        int baseNC = major * p.sizeMinor + minorBase;
        int n = baseNC / p.inSize.z;
        int baseC = baseNC - n * p.inSize.z;
        for (int loopX = 0, tileOutX = tileOutXBase; loopX < p.loopX & tileOutX < p.outSize.x; loopX++, tileOutX += tileOutW)
        {
            // Load input pixels.
            int tileMidX = tileOutX * downx + upx - 1 - p.pad0.x;
            int tileMidY = tileOutY * downy + upy - 1 - p.pad0.y;
            int tileInX = floor_div(tileMidX, upx);
            int tileInY = floor_div(tileMidY, upy);
            __syncthreads();
            for (int inIdx = threadIdx.x; inIdx < tileInH * tileInW * loopMinor; inIdx += blockDim.x)
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
                if (inX >= 0 & inY >= 0 & inX < p.inSize.x & inY < p.inSize.y & c < p.inSize.z)
                    v = (scalar_t)((const T*)p.x)[inX * p.inStride.x + inY * p.inStride.y + c * p.inStride.z + n * p.inStride.w];
                sx[relInY][relInX][relC] = v;
            }

            // Loop over output pixels.
            __syncthreads();
            for (int outIdx = threadIdx.x; outIdx < tileOutH * tileOutW * loopMinor; outIdx += blockDim.x)
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
                if (outX < p.outSize.x & outY < p.outSize.y & c < p.outSize.z)
                {
                    scalar_t v = 0;
                    #pragma unroll
                    for (int y = 0; y < filterH / upy; y++)
                        #pragma unroll
                        for (int x = 0; x < filterW / upx; x++)
                            v += sx[relInY + y][relInX + x][relC] * sf[filterY + y * upy][filterX + x * upx];
                    v *= p.gain;
                    ((T*)p.y)[outX * p.outStride.x + outY * p.outStride.y + c * p.outStride.z + n * p.outStride.w] = (T)v;
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
    dim3 blockSize = dim3(256, 1, 1);
    dim3 gridSize = dim3(
        ((p.outSize.y - 1) / tileOutH + 1) * p.launchMinor,
        (p.outSize.x - 1) / (tileOutW * p.loopX) + 1,
        p.launchMajor);
    
    void* args[] = {&p};
    cudaLaunchKernel((void*)&upfirdn2d_kernel_small<T, upx, upy, downx, downy, filterW, filterH, tileOutW, tileOutH, loopMinor>, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream());
}

template <class T>
void run_upfirdn2d_kernel_large(upfirdn2d_kernel_params p, int tileOutW, int tileOutH, int loopMinor, int loopX) {
    update_params(p, loopMinor, loopX);
    
    // Compute grid size - for large kernels
    dim3 blockSize = dim3(4, 32, 1);
    dim3 gridSize = dim3(
        ((p.outSize.y - 1) / blockSize.x + 1) * p.launchMinor,
        (p.outSize.x - 1) / (blockSize.y * p.loopX) + 1,
        p.launchMajor);
    
    void* args[] = {&p};
    cudaLaunchKernel((void*)&upfirdn2d_kernel_large<T>, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream());
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
