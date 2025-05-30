// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "dpct/dpct.hpp"

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct upfirdn2d_kernel_params
{
    const void*     x;
    const float*    f;
    void*           y;

    sycl::int2 up;
    sycl::int2 down;
    sycl::int2 pad0;
    int             flip;
    float           gain;

    sycl::int4 inSize; // [width, height, channel, batch]
    sycl::int4 inStride;
    sycl::int2 filterSize; // [width, height]
    sycl::int2 filterStride;
    sycl::int4 outSize; // [width, height, channel, batch]
    sycl::int4 outStride;
    int             sizeMinor;
    int             sizeMajor;

    int             loopMinor;
    int             loopMajor;
    int             loopX;
    int             launchMinor;
    int             launchMajor;
};

//------------------------------------------------------------------------
// kernel selection.

template <class T>
void run_upfirdn2d_kernel_large(upfirdn2d_kernel_params p, int tileOutW, int tileOutH, int loopMinor, int loopX);

template <class T, int upx, int upy, int downx, int downy, int filterW, int filterH, int tileOutW, int tileOutH, int loopMinor>
void run_upfirdn2d_kernel_small(upfirdn2d_kernel_params p);

//------------------------------------------------------------------------
