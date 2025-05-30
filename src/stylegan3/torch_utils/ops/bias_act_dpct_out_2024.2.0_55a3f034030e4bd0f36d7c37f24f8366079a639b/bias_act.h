#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>
#include <c10/util/Half.h>

// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct bias_act_kernel_params
{
    c10::ScalarType dtype = c10::ScalarType::Undefined;
    const void* x;      // [sizeX]
    const void* b;      // [sizeB] or NULL
    const void* xref;   // [sizeX] or NULL
    const void* yref;   // [sizeX] or NULL
    const void* dy;     // [sizeX] or NULL
    void*       y;      // [sizeX]

    int         grad;
    int         act;
    float       alpha;
    float       gain;
    float       clamp;

    int         sizeX;
    int         sizeB;
    int         stepB;
    int         loopX;
};

//------------------------------------------------------------------------
// CUDA kernel selection.

//template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p);
//template <class T, int A> __global__ void bias_act_kernel(bias_act_kernel_params p);
//template <class T> __global__ void bias_act_kernel(bias_act_kernel_params p);
//__global__ void bias_act_kernel(bias_act_kernel_params p);

void bias_act_kernel_launch(bias_act_kernel_params p);

//------------------------------------------------------------------------
