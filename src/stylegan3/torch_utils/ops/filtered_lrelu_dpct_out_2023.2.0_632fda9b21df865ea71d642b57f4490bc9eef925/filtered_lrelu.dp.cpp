// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "filtered_lrelu.h"
#include <sycl/accessor.hpp>
#include <ipex.h>
#include <torch/extension.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cmath>

#define TORCH_ATLEAST_2_3 (TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 3))
#if TORCH_ATLEAST_2_3
#include <c10/xpu/XPUStream.h>
#endif


//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>
{
    typedef double scalar_t; typedef sycl::double2 vec2_t;
        typedef sycl::double4 vec4_t;
    __dpct_inline__ static vec2_t zero_vec2() { return sycl::double2(0, 0); }
    __dpct_inline__ static vec4_t zero_vec4() {
        return sycl::double4(0, 0, 0, 0);
    }
    __dpct_inline__ static double clamp(double x, double c) {
        return sycl::fmin(sycl::fmax(x, -c), c);
    }
};

template <> struct InternalType<float>
{
    typedef float scalar_t; typedef sycl::float2 vec2_t;
        typedef sycl::float4 vec4_t;
    __dpct_inline__ static vec2_t zero_vec2() { return sycl::float2(0, 0); }
    __dpct_inline__ static vec4_t zero_vec4() {
        return sycl::float4(0, 0, 0, 0);
    }
    __dpct_inline__ static float clamp(float x, float c) {
        return sycl::fmin(sycl::fmax(x, -c), c);
    }
};

template <> struct InternalType<c10::Half>
{
    typedef float scalar_t; typedef sycl::float2 vec2_t;
        typedef sycl::float4 vec4_t;
    __dpct_inline__ static vec2_t zero_vec2() { return sycl::float2(0, 0); }
    __dpct_inline__ static vec4_t zero_vec4() {
        return sycl::float4(0, 0, 0, 0);
    }
    __dpct_inline__ static float clamp(float x, float c) {
        return sycl::fmin(sycl::fmax(x, -c), c);
    }
};

#define MIN(A, B)       ((A) < (B) ? (A) : (B))
#define MAX(A, B)       ((A) > (B) ? (A) : (B))
#define CEIL_DIV(A, B) (((B)==1) ? (A) : \
                        ((B)==2) ? ((int)((A)+1) >> 1) : \
                        ((B)==4) ? ((int)((A)+3) >> 2) : \
                        (((A) + ((A) > 0 ? (B) - 1 : 0)) / (B)))

// This works only up to blocks of size 256 x 256 and for all N that are powers of two.
template <int N>
__dpct_inline__ void fast_div_mod(int &x, int &y, unsigned int i)
{
    if ((N & (N-1)) && N <= 256)
        y = (i * ((1<<24)/N + 1)) >> 24; // Assumes N <= 256, i < N*256.
    else
        y = i/N;

    x = i - y*N;
}

// Type cast stride before reading it.
template <class T> __dpct_inline__ T get_stride(const int64_t &x)
{
    return *reinterpret_cast<const T*>(&x);
}

//------------------------------------------------------------------------
// Filters, setup kernel, copying function.

#define MAX_FILTER_SIZE 32

// Combined up/down filter buffers so that transfer can be done with one copy.
static dpct::global_memory<float, 1> g_fbuf(
    2 * MAX_FILTER_SIZE *
    MAX_FILTER_SIZE); // Filters in global memory, written by setup kernel.
static dpct::constant_memory<float, 1>
    c_fbuf(2 * MAX_FILTER_SIZE *
           MAX_FILTER_SIZE); // Filters in constant memory, read by main kernel.

// Accessors to combined buffers to index up/down filters individually.
#define c_fu (c_fbuf)
#define c_fd (c_fbuf + MAX_FILTER_SIZE * MAX_FILTER_SIZE)
#define g_fu (g_fbuf)
#define g_fd (g_fbuf + MAX_FILTER_SIZE * MAX_FILTER_SIZE)

// Set up filters into global memory buffer.
static void setup_filters_kernel(filtered_lrelu_kernel_params p,
                                 const sycl::nd_item<3> &item_ct1, float *g_fbuf)
{
    for (int idx = item_ct1.get_local_id(2);
         idx < MAX_FILTER_SIZE * MAX_FILTER_SIZE;
         idx += item_ct1.get_local_range(2))
    {
        int x, y;
        fast_div_mod<MAX_FILTER_SIZE>(x, y, idx);

        int fu_x = p.flip ? x : (p.fuShape.x() - 1 - x);
        int fu_y = p.flip ? y : (p.fuShape.y() - 1 - y);
        if (p.fuShape.y() > 0)
            g_fu[idx] =
                (x >= p.fuShape.x() || y >= p.fuShape.y())
                    ? 0.0f
                    : p.fu[fu_x * p.fuStride.x() + fu_y * p.fuStride.y()];
        else
            g_fu[idx] = (x >= p.fuShape.x() || y > 0)
                            ? 0.0f
                            : p.fu[fu_x * p.fuStride.x()];

        int fd_x = p.flip ? x : (p.fdShape.x() - 1 - x);
        int fd_y = p.flip ? y : (p.fdShape.y() - 1 - y);
        if (p.fdShape.y() > 0)
            g_fd[idx] =
                (x >= p.fdShape.x() || y >= p.fdShape.y())
                    ? 0.0f
                    : p.fd[fd_x * p.fdStride.x() + fd_y * p.fdStride.y()];
        else
            g_fd[idx] = (x >= p.fdShape.x() || y > 0)
                            ? 0.0f
                            : p.fd[fd_x * p.fdStride.x()];
    }
}

// Host function to copy filters written by setup kernel into constant buffer for main kernel.
template <bool, bool>
static dpct::err0 copy_filters(sycl::queue& stream) try {
    void* src = 0;
    dpct::err0 err = DPCT_CHECK_ERROR(*(&src) = g_fbuf.get_ptr());
    /*
    DPCT1001:36: The statement could not be removed.
    */
    /*
    DPCT1000:37: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err) return err;
    return DPCT_CHECK_ERROR(
        stream.memcpy(c_fbuf.get_ptr(stream), src,
                       2 * MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(float)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------
// Coordinate spaces:
// - Relative to input tensor:      inX, inY, tileInX, tileInY
// - Relative to input tile:        relInX, relInY, tileInW, tileInH
// - Relative to upsampled tile:    relUpX, relUpY, tileUpW, tileUpH
// - Relative to output tile:       relOutX, relOutY, tileOutW, tileOutH
// - Relative to output tensor:     outX, outY, tileOutX, tileOutY
//
// Relationships between coordinate spaces:
// - inX = tileInX + relInX
// - inY = tileInY + relInY
// - relUpX = relInX * up + phaseInX
// - relUpY = relInY * up + phaseInY
// - relUpX = relOutX * down
// - relUpY = relOutY * down
// - outX = tileOutX + relOutX
// - outY = tileOutY + relOutY

template <class T, class index_t, bool signWrite, bool signRead,
          int filterMode, int up, int fuSize, int down, int fdSize,
          int tileOutW, int tileOutH, int threadsPerBlock, bool enableXrep,
          bool enableWriteSkip, class AccessorT
          >
/*
DPCT1110:3: The total declared local variable size in device function
filtered_lrelu_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
static void filtered_lrelu_kernel(filtered_lrelu_kernel_params p,
                                  const sycl::nd_item<3> &item_ct1,
                                  float *c_fbuf,
                                  typename InternalType<T>::scalar_t *s_buf0_st,
                                  AccessorT yacc
                                  )
{

    //char* yacc = (char*)p.y; // for debugging (access the same indices through yacc[], but actually use a pointer instead of an accessor ... the accessor can be removed from the function parameters or just renamed)
    // Check that we don't try to support non-existing filter modes.
    static_assert(up   == 1 || up   == 2 || up   == 4, "only up=1, up=2, up=4 scales supported");
    static_assert(down == 1 || down == 2 || down == 4, "only down=1, down=2, down=4 scales supported");
    static_assert(fuSize >= up,   "upsampling filter size must be at least upsampling factor");
    static_assert(fdSize >= down, "downsampling filter size must be at least downsampling factor");
    static_assert(fuSize % up   == 0, "upsampling filter size must be divisible with upsampling factor");
    static_assert(fdSize % down == 0, "downsampling filter size must be divisible with downsampling factor");
    static_assert(fuSize <= MAX_FILTER_SIZE && fdSize <= MAX_FILTER_SIZE, "filter size greater than MAX_FILTER_SIZE");
    static_assert(up   != 1 || (fuSize == 1 && (filterMode == MODE_FUFD || filterMode == MODE_FUSD)), "up=1 supported only for 1x1 full filters");
    static_assert(down != 1 || (fdSize == 1 && (filterMode == MODE_FUFD || filterMode == MODE_SUFD)), "down=1 supported only for 1x1 full filters");
    static_assert(!(up   == 4 && (filterMode == MODE_FUFD || filterMode == MODE_FUSD)), "full filters not supported for up=4");
    static_assert(!(down == 4 && (filterMode == MODE_FUFD || filterMode == MODE_SUFD)), "full filters not supported for down=4");

    // Static definitions.
    typedef typename InternalType<T>::scalar_t scalar_t;
    typedef typename InternalType<T>::vec2_t vec2_t;
    typedef typename InternalType<T>::vec4_t vec4_t;
    const int tileUpW    = (tileOutW * down + (fdSize - 1) - (down - 1) + 3) & ~3;  // Upsampled tile width, rounded up to multiple of 4.
    const int tileUpH    = tileOutH * down + (fdSize - 1) - (down - 1);             // Upsampled tile height.
    const int tileInW    = CEIL_DIV(tileUpW  + (fuSize - 1), up);                   // Input tile width.
    const int tileInH    = CEIL_DIV(tileUpH  + (fuSize - 1), up);                   // Input tile height.
    const int tileUpH_up = CEIL_DIV(tileUpH, up) * up;                              // Upsampled tile height rounded up to a multiple of up.
    const int tileInH_up = CEIL_DIV(tileUpH_up + (fuSize - 1), up);                 // For allocations only, to avoid shared memory read overruns with up=2 and up=4.

    // Merge 1x1 downsampling into last upsampling step for upf1 and ups2.
    const bool downInline = (down == 1) && ((up == 1 && filterMode == MODE_FUFD) || (up == 2 && filterMode == MODE_SUFD));

    // Sizes of logical buffers.
    const int szIn    = tileInH_up * tileInW;
    const int szUpX   = tileInH_up * tileUpW;
    const int szUpXY  = downInline ? 0 : (tileUpH * tileUpW);
    const int szDownX = tileUpH * tileOutW;

    // Sizes for shared memory arrays.
    const int s_buf0_size_base =
        (filterMode == MODE_SUSD) ? MAX(szIn, szUpXY) :
        (filterMode == MODE_FUSD) ? MAX(szIn, szDownX) :
        (filterMode == MODE_SUFD) ? MAX(szIn, szUpXY) :
        (filterMode == MODE_FUFD) ? szIn :
        -1;
    const int s_buf1_size_base =
        (filterMode == MODE_SUSD) ? MAX(szUpX, szDownX) :
        (filterMode == MODE_FUSD) ? szUpXY :
        (filterMode == MODE_SUFD) ? szUpX  :
        (filterMode == MODE_FUFD) ? szUpXY :
        -1;

    // Ensure U128 alignment.
    const int s_buf0_size = (s_buf0_size_base + 3) & ~3;
    const int s_buf1_size = (s_buf1_size_base + 3) & ~3;

    // Declare shared memory arrays.
    scalar_t* s_buf0;
    scalar_t* s_buf1;

    s_buf0 = s_buf0_st;
    s_buf1 = s_buf0 + s_buf0_size;

    // Pointers to the buffers.
    scalar_t* s_tileIn;       // Input tile:                      [relInX * tileInH + relInY]
    scalar_t* s_tileUpX;      // After horizontal upsampling:     [relInY * tileUpW + relUpX]
    scalar_t* s_tileUpXY;     // After upsampling:                [relUpY * tileUpW + relUpX]
    scalar_t* s_tileDownX;    // After horizontal downsampling:   [relUpY * tileOutW + relOutX]
    if (filterMode == MODE_SUSD)
    {
        s_tileIn    = s_buf0;
        s_tileUpX   = s_buf1;
        s_tileUpXY  = s_buf0;
        s_tileDownX = s_buf1;
    }
    else if (filterMode == MODE_FUSD)
    {
        s_tileIn    = s_buf0;
        s_tileUpXY  = s_buf1;
        s_tileDownX = s_buf0;
    }
    else if (filterMode == MODE_SUFD)
    {
        s_tileIn    = s_buf0;
        s_tileUpX   = s_buf1;
        s_tileUpXY  = s_buf0;
    }
    else if (filterMode == MODE_FUFD)
    {
        s_tileIn    = s_buf0;
        s_tileUpXY  = s_buf1;
    }

    // Allow large grids in z direction via per-launch offset.
    int channelIdx = item_ct1.get_group(0) + p.blockZofs;
    int batchIdx = channelIdx / p.yShape.z();
    channelIdx -= batchIdx * p.yShape.z();

    // Offset to output feature map. In bytes.
    index_t mapOfsOut = channelIdx * get_stride<index_t>(p.yStride.z()) +
                        batchIdx * get_stride<index_t>(p.yStride.w());

    // Sign shift amount.
    uint32_t signXo = ((item_ct1.get_local_id(2) + p.sOfs.x()) << 1) & 6;

    // Inner tile loop.
    #pragma unroll 1
    for (int tileIdx = 0;
         !enableXrep ||
         (tileIdx <
          MIN(p.tilesXrep, p.tilesXdim - p.tilesXrep * item_ct1.get_group(1)));
         tileIdx++)
    {
        // Locate output tile.
        int tileX = enableXrep ? item_ct1.get_group(1) * p.tilesXrep + tileIdx
                               : item_ct1.get_group(2);
        int tileOutX = tileX * tileOutW;
        int tileOutY =
            (enableXrep ? item_ct1.get_group(2) : item_ct1.get_group(1)) *
            tileOutH;

        // Locate input tile.
        int tmpX = tileOutX * down - p.pad0.x();
        int tmpY = tileOutY * down - p.pad0.y();
        int tileInX = CEIL_DIV(tmpX, up);
        int tileInY = CEIL_DIV(tmpY, up);
        const int phaseInX = tileInX * up - tmpX;
        const int phaseInY = tileInY * up - tmpY;

        // Extra sync if input and output buffers are the same and we are not on first tile.
        if (enableXrep && tileIdx > 0 && (filterMode == MODE_FUSD || (filterMode == MODE_SUFD && !downInline) || (filterMode == MODE_FUFD && downInline)))
            item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load input tile & apply bias. Unrolled.
        scalar_t b = (scalar_t)*(const T*)((const char*)p.b + (channelIdx * get_stride<index_t>(p.bStride)));
        index_t mapOfsIn = channelIdx * get_stride<index_t>(p.xStride.z()) +
                           batchIdx * get_stride<index_t>(p.xStride.w());
        int idx = item_ct1.get_local_id(2);
        const int loopCountIN = CEIL_DIV(tileInW * tileInH, threadsPerBlock);
        #pragma unroll
        for (int loop = 0; loop < loopCountIN; loop++)
        {
            int relInX, relInY;
            fast_div_mod<tileInW>(relInX, relInY, idx);
            int inX = tileInX + relInX;
            int inY = tileInY + relInY;
            scalar_t v = 0;

            if ((uint32_t)inX < p.xShape.x() && (uint32_t)inY < p.xShape.y())
                v = (scalar_t) *
                        ((const T *)((const char *)p.x +
                                     (inX * get_stride<index_t>(p.xStride.x()) +
                                      inY * get_stride<index_t>(p.xStride.y()) +
                                      mapOfsIn))) +
                    b;

            bool skip = (loop == loopCountIN-1) && (idx >= tileInW * tileInH);
            if (!skip)
                s_tileIn[idx] = v;

            idx += threadsPerBlock;
        }

        if (filterMode == MODE_SUSD || filterMode == MODE_SUFD) // Separable upsampling filter.
        {
            // Horizontal upsampling.
            item_ct1.barrier(sycl::access::fence_space::local_space);
            if (up == 4)
            {
                for (int idx = item_ct1.get_local_id(2) * up;
                     idx < tileUpW * tileInH;
                     idx += item_ct1.get_local_range(2) * up)
                {
                    int relUpX0, relInY;
                    fast_div_mod<tileUpW>(relUpX0, relInY, idx);
                    int relInX0 = relUpX0 / up;
                    int src0 = relInX0 + tileInW * relInY;
                    int dst = relInY * tileUpW + relUpX0;
                    vec4_t v = InternalType<T>::zero_vec4();
                    scalar_t a = s_tileIn[src0];
                    if (phaseInX == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 3];
                            v.z() += a * (scalar_t)c_fu[step * up + 2];
                            v.w() += a * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else if (phaseInX == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                            v.z() += a * (scalar_t)c_fu[step * up + 3];
                            v.w() += a * (scalar_t)c_fu[step * up + 2];
                        }
                    }
                    else if (phaseInX == 2)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 2];
                            v.y() += a * (scalar_t)c_fu[step * up + 1];
                            v.z() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                            v.w() += a * (scalar_t)c_fu[step * up + 3];
                        }
                    }
                    else // (phaseInX == 3)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 3];
                            v.y() += a * (scalar_t)c_fu[step * up + 2];
                            v.z() += a * (scalar_t)c_fu[step * up + 1];
                            v.w() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                        }
                    }
                    s_tileUpX[dst + 0] = v.x();
                    s_tileUpX[dst + 1] = v.y();
                    s_tileUpX[dst + 2] = v.z();
                    s_tileUpX[dst + 3] = v.w();
                }
            }
            else if (up == 2)
            {
                bool p0 = (phaseInX == 0);
                for (int idx = item_ct1.get_local_id(2) * up;
                     idx < tileUpW * tileInH;
                     idx += item_ct1.get_local_range(2) * up)
                {
                    int relUpX0, relInY;
                    fast_div_mod<tileUpW>(relUpX0, relInY, idx);
                    int relInX0 = relUpX0 / up;
                    int src0 = relInX0 + tileInW * relInY;
                    int dst = relInY * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    scalar_t a = s_tileIn[src0];
                    if (p0) // (phaseInX == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else // (phaseInX == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                        }
                    }
                    s_tileUpX[dst + 0] = v.x();
                    s_tileUpX[dst + 1] = v.y();
                }
            }

            // Vertical upsampling & nonlinearity.

            item_ct1.barrier(sycl::access::fence_space::local_space);
            int groupMask = 15 << ((item_ct1.get_local_id(2) & 31) & ~3);
            int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH : 0; // Skip already written signs.
            int sShapeMaxY = MIN(p.sShape.y(),
                                 tileOutY * down +
                                     tileUpH); // Avoid out-of-tile sign writes.
            if (up == 4)
            {
                minY -= 3; // Adjust according to block height.
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileUpW * tileUpH_up / up;
                     idx += item_ct1.get_local_range(2))
                {
                    int relUpX, relInY0;
                    fast_div_mod<tileUpW>(relUpX, relInY0, idx);
                    int relUpY0 = relInY0 * up;
                    int src0 = relInY0 * tileUpW + relUpX;
                    int dst = relUpY0 * tileUpW + relUpX;
                    vec4_t v = InternalType<T>::zero_vec4();

                    scalar_t a = s_tileUpX[src0];
                    if (phaseInY == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                            v.y() += a * (scalar_t)c_fu[step * up + 3];
                            v.z() += a * (scalar_t)c_fu[step * up + 2];
                            v.w() += a * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else if (phaseInY == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                            v.z() += a * (scalar_t)c_fu[step * up + 3];
                            v.w() += a * (scalar_t)c_fu[step * up + 2];
                        }
                    }
                    else if (phaseInY == 2)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 2];
                            v.y() += a * (scalar_t)c_fu[step * up + 1];
                            v.z() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                            v.w() += a * (scalar_t)c_fu[step * up + 3];
                        }
                    }
                    else // (phaseInY == 3)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 3];
                            v.y() += a * (scalar_t)c_fu[step * up + 2];
                            v.z() += a * (scalar_t)c_fu[step * up + 1];
                            v.w() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                        }
                    }

                    int x = tileOutX * down + relUpX;
                    int y = tileOutY * down + relUpY0;
                    int signX = x + p.sOfs.x();
                    int signY = y + p.sOfs.y();
                    int signZ = item_ct1.get_group(0) + p.blockZofs;
                    int signXb = signX >> 2;
                    index_t si0 =
                        signXb +
                        p.sShape.x() * (signY + (index_t)p.sShape.y() * signZ);
                    index_t si1 = si0 + p.sShape.x();
                    index_t si2 = si0 + p.sShape.x() * 2;
                    index_t si3 = si0 + p.sShape.x() * 3;

                    v.x() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.z() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.w() *= (scalar_t)((float)up * (float)up * p.gain);

                    if (signWrite)
                    {
                        if (!enableWriteSkip)
                        {
                            // Determine and write signs.
                            int sx =
                                sycl::bit_cast<unsigned int>(v.x()) >> 31 << 0;
                            int sy =
                                sycl::bit_cast<unsigned int>(v.y()) >> 31 << 8;
                            int sz =
                                sycl::bit_cast<unsigned int>(v.z()) >> 31 << 16;
                            int sw =
                                sycl::bit_cast<unsigned int>(v.w()) >> 31 << 24;
                            if (sx) v.x() *= p.slope;
                            if (sy) v.y() *= p.slope;
                            if (sz) v.z() *= p.slope;
                            if (sw) v.w() *= p.slope;
                            if (sycl::fabs(v.x()) > p.clamp) {
                                sx = 2 << 0;
                                v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                            }
                            if (sycl::fabs(v.y()) > p.clamp) {
                                sy = 2 << 8;
                                v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                            }
                            if (sycl::fabs(v.z()) > p.clamp) {
                                sz = 2 << 16;
                                v.z() = InternalType<T>::clamp(v.z(), p.clamp);
                            }
                            if (sycl::fabs(v.w()) > p.clamp) {
                                sw = 2 << 24;
                                v.w() = InternalType<T>::clamp(v.w(), p.clamp);
                            }

                            if ((uint32_t)signXb < p.swLimit && signY >= minY)
                            {
                                // Combine signs.
                                uint32_t s = sx + sy + sw + sz;
                                s <<= (signX & 3) << 1;
                                /*
                                DPCT1023:7: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:40: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s |= dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s, 1);
                                /*
                                DPCT1023:8: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:41: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s |= dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s, 2);

                                // Write signs.
                                if ((uint32_t)(signY + 0) < sShapeMaxY) { p.s[si0] = (unsigned char)(s >>  0); }
                                if ((uint32_t)(signY + 1) < sShapeMaxY) { p.s[si1] = (unsigned char)(s >>  8); }
                                if ((uint32_t)(signY + 2) < sShapeMaxY) { p.s[si2] = (unsigned char)(s >> 16); }
                                if ((uint32_t)(signY + 3) < sShapeMaxY) { p.s[si3] = (unsigned char)(s >> 24); }
                            }
                        }
                        else
                        {
                            // Determine and write signs.
                            if ((uint32_t)signXb < p.swLimit && signY >= minY)
                            {
                                int sx = sycl::bit_cast<unsigned int>(v.x()) >>
                                         31 << 0;
                                int sy = sycl::bit_cast<unsigned int>(v.y()) >>
                                         31 << 8;
                                int sz = sycl::bit_cast<unsigned int>(v.z()) >>
                                         31 << 16;
                                int sw = sycl::bit_cast<unsigned int>(v.w()) >>
                                         31 << 24;
                                if (sx) v.x() *= p.slope;
                                if (sy) v.y() *= p.slope;
                                if (sz) v.z() *= p.slope;
                                if (sw) v.w() *= p.slope;
                                if (sycl::fabs(v.x()) > p.clamp) {
                                    sx = 2 << 0; v.x() = InternalType<T>::clamp(
                                                     v.x(), p.clamp);
                                }
                                if (sycl::fabs(v.y()) > p.clamp) {
                                    sy = 2 << 8; v.y() = InternalType<T>::clamp(
                                                     v.y(), p.clamp);
                                }
                                if (sycl::fabs(v.z()) > p.clamp) {
                                    sz = 2 << 16;
                                    v.z() =
                                        InternalType<T>::clamp(v.z(), p.clamp);
                                }
                                if (sycl::fabs(v.w()) > p.clamp) {
                                    sw = 2 << 24;
                                    v.w() =
                                        InternalType<T>::clamp(v.w(), p.clamp);
                                }

                                // Combine signs.
                                uint32_t s = sx + sy + sw + sz;
                                s <<= (signX & 3) << 1;
                                /*
                                DPCT1023:9: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:42: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s |= dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s, 1);
                                /*
                                DPCT1023:10: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:43: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s |= dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s, 2);

                                // Write signs.
                                if ((uint32_t)(signY + 0) < sShapeMaxY) { p.s[si0] = (unsigned char)(s >>  0); }
                                if ((uint32_t)(signY + 1) < sShapeMaxY) { p.s[si1] = (unsigned char)(s >>  8); }
                                if ((uint32_t)(signY + 2) < sShapeMaxY) { p.s[si2] = (unsigned char)(s >> 16); }
                                if ((uint32_t)(signY + 3) < sShapeMaxY) { p.s[si3] = (unsigned char)(s >> 24); }
                            }
                            else
                            {
                                // Just compute the values.
                                if (v.x() < 0.f) v.x() *= p.slope;
                                    v.x() =
                                        InternalType<T>::clamp(v.x(), p.clamp);
                                if (v.y() < 0.f) v.y() *= p.slope;
                                    v.y() =
                                        InternalType<T>::clamp(v.y(), p.clamp);
                                if (v.z() < 0.f) v.z() *= p.slope;
                                    v.z() =
                                        InternalType<T>::clamp(v.z(), p.clamp);
                                if (v.w() < 0.f) v.w() *= p.slope;
                                    v.w() =
                                        InternalType<T>::clamp(v.w(), p.clamp);
                            }
                        }
                    }
                    else if (signRead) // Read signs and apply.
                    {
                        if ((uint32_t)signXb < p.swLimit)
                        {
                            int ss = (signX & 3) << 1;
                            if ((uint32_t)(signY + 0) < p.sShape.y()) {
                                int s = p.s[si0] >> ss;
                                if (s & 1) v.x() *= p.slope;
                                if (s & 2) v.x() = 0.f;
                            }
                            if ((uint32_t)(signY + 1) < p.sShape.y()) {
                                int s = p.s[si1] >> ss;
                                if (s & 1) v.y() *= p.slope;
                                if (s & 2) v.y() = 0.f;
                            }
                            if ((uint32_t)(signY + 2) < p.sShape.y()) {
                                int s = p.s[si2] >> ss;
                                if (s & 1) v.z() *= p.slope;
                                if (s & 2) v.z() = 0.f;
                            }
                            if ((uint32_t)(signY + 3) < p.sShape.y()) {
                                int s = p.s[si3] >> ss;
                                if (s & 1) v.w() *= p.slope;
                                if (s & 2) v.w() = 0.f;
                            }
                        }
                    }
                    else // Forward pass with no sign write.
                    {
                        if (v.x() < 0.f) v.x() *= p.slope;
                            v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                        if (v.y() < 0.f) v.y() *= p.slope;
                            v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                        if (v.z() < 0.f) v.z() *= p.slope;
                            v.z() = InternalType<T>::clamp(v.z(), p.clamp);
                        if (v.w() < 0.f) v.w() *= p.slope;
                            v.w() = InternalType<T>::clamp(v.w(), p.clamp);
                    }

                    s_tileUpXY[dst + 0 * tileUpW] = v.x();
                    if (relUpY0 + 1 < tileUpH) s_tileUpXY[dst + 1 * tileUpW] =
                        v.y();
                    if (relUpY0 + 2 < tileUpH) s_tileUpXY[dst + 2 * tileUpW] =
                        v.z();
                    if (relUpY0 + 3 < tileUpH) s_tileUpXY[dst + 3 * tileUpW] =
                        v.w();
                }
            }
            else if (up == 2)
            {
                minY -= 1; // Adjust according to block height.
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileUpW * tileUpH_up / up;
                     idx += item_ct1.get_local_range(2))
                {
                    int relUpX, relInY0;
                    fast_div_mod<tileUpW>(relUpX, relInY0, idx);
                    int relUpY0 = relInY0 * up;
                    int src0 = relInY0 * tileUpW + relUpX;
                    int dst = relUpY0 * tileUpW + relUpX;
                    vec2_t v = InternalType<T>::zero_vec2();

                    scalar_t a = s_tileUpX[src0];
                    if (phaseInY == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                            v.y() += a * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else // (phaseInY == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                        }
                    }

                    int x = tileOutX * down + relUpX;
                    int y = tileOutY * down + relUpY0;
                    int signX = x + p.sOfs.x();
                    int signY = y + p.sOfs.y();
                    int signZ = item_ct1.get_group(0) + p.blockZofs;
                    int signXb = signX >> 2;
                    index_t si0 =
                        signXb +
                        p.sShape.x() * (signY + (index_t)p.sShape.y() * signZ);
                    index_t si1 = si0 + p.sShape.x();

                    v.x() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y() *= (scalar_t)((float)up * (float)up * p.gain);

                    if (signWrite)
                    {
                        if (!enableWriteSkip)
                        {
                            // Determine and write signs.
                            int sx =
                                sycl::bit_cast<unsigned int>(v.x()) >> 31 << 0;
                            int sy =
                                sycl::bit_cast<unsigned int>(v.y()) >> 31 << 8;
                            if (sx) v.x() *= p.slope;
                            if (sy) v.y() *= p.slope;
                            if (sycl::fabs(v.x()) > p.clamp) {
                                sx = 2 << 0;
                                v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                            }
                            if (sycl::fabs(v.y()) > p.clamp) {
                                sy = 2 << 8;
                                v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                            }

                            if ((uint32_t)signXb < p.swLimit && signY >= minY)
                            {
                                // Combine signs.
                                int s = sx + sy;
                                s <<= signXo;
                                /*
                                DPCT1023:11: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:44: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s |= dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s, 1);
                                /*
                                DPCT1023:12: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:45: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s |= dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s, 2);

                                // Write signs.
                                if ((uint32_t)(signY + 0) < sShapeMaxY) { p.s[si0] = (unsigned char)(s >>  0); }
                                if ((uint32_t)(signY + 1) < sShapeMaxY) { p.s[si1] = (unsigned char)(s >>  8); }
                            }
                        }
                        else
                        {
                            // Determine and write signs.
                            if ((uint32_t)signXb < p.swLimit && signY >= minY)
                            {
                                int sx = sycl::bit_cast<unsigned int>(v.x()) >>
                                         31 << 0;
                                int sy = sycl::bit_cast<unsigned int>(v.y()) >>
                                         31 << 8;
                                if (sx) v.x() *= p.slope;
                                if (sy) v.y() *= p.slope;
                                if (sycl::fabs(v.x()) > p.clamp) {
                                    sx = 2 << 0; v.x() = InternalType<T>::clamp(
                                                     v.x(), p.clamp);
                                }
                                if (sycl::fabs(v.y()) > p.clamp) {
                                    sy = 2 << 8; v.y() = InternalType<T>::clamp(
                                                     v.y(), p.clamp);
                                }

                                // Combine signs.
                                int s = sx + sy;
                                s <<= signXo;
                                /*
                                DPCT1023:13: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:46: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s |= dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s, 1);
                                /*
                                DPCT1023:14: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:47: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s |= dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s, 2);

                                // Write signs.
                                if ((uint32_t)(signY + 0) < sShapeMaxY) { p.s[si0] = (unsigned char)(s >>  0); }
                                if ((uint32_t)(signY + 1) < sShapeMaxY) { p.s[si1] = (unsigned char)(s >>  8); }
                            }
                            else
                            {
                                // Just compute the values.
                                if (v.x() < 0.f) v.x() *= p.slope;
                                    v.x() =
                                        InternalType<T>::clamp(v.x(), p.clamp);
                                if (v.y() < 0.f) v.y() *= p.slope;
                                    v.y() =
                                        InternalType<T>::clamp(v.y(), p.clamp);
                            }
                        }
                    }
                    else if (signRead) // Read signs and apply.
                    {
                        if ((uint32_t)signXb < p.swLimit)
                        {
                            if ((uint32_t)(signY + 0) < p.sShape.y()) {
                                int s = p.s[si0] >> signXo;
                                if (s & 1) v.x() *= p.slope;
                                if (s & 2) v.x() = 0.f;
                            }
                            if ((uint32_t)(signY + 1) < p.sShape.y()) {
                                int s = p.s[si1] >> signXo;
                                if (s & 1) v.y() *= p.slope;
                                if (s & 2) v.y() = 0.f;
                            }
                        }
                    }
                    else // Forward pass with no sign write.
                    {
                        if (v.x() < 0.f) v.x() *= p.slope;
                            v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                        if (v.y() < 0.f) v.y() *= p.slope;
                            v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                    }

                    if (!downInline)
                    {
                        // Write into temporary buffer.
                        s_tileUpXY[dst] = v.x();
                        if (relUpY0 < tileUpH - 1)
                            s_tileUpXY[dst + tileUpW] = v.y();
                    }
                    else
                    {
                        // Write directly into output buffer.
                        if ((uint32_t)x < p.yShape.x())
                        {
                            int ymax =
                                MIN(p.yShape.y(), tileUpH + tileOutY * down);
                            index_t ofs =
                                x * get_stride<index_t>(p.yStride.x()) +
                                y * get_stride<index_t>(p.yStride.y()) +
                                mapOfsOut;
                            if ((uint32_t)y + 0 < p.yShape.y())
                                (T&)(yacc[ofs]) = (T)(v.x() * (scalar_t)c_fd[0]);
                            if ((uint32_t)y + 1 < ymax)
                                (T&)(yacc[ofs + get_stride<index_t>(p.yStride.y())]) = (T)(v.y() * (scalar_t)c_fd[0]);
                        }
                    }
                }
            }
        }
        else if (filterMode == MODE_FUSD || filterMode == MODE_FUFD)
        {
            // Full upsampling filter.

            if (up == 2)
            {
                // 2 x 2-wide.
                item_ct1.barrier(sycl::access::fence_space::local_space);
                int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH +
                                          p.sOfs.y()
                                    : 0; // Skip already written signs.
                for (int idx = item_ct1.get_local_id(2) * 4;
                     idx < tileUpW * tileUpH;
                     idx += item_ct1.get_local_range(2) * 4)
                {
                    int relUpX0, relUpY0;
                    fast_div_mod<tileUpW>(relUpX0, relUpY0, idx);
                    int relInX0 = CEIL_DIV(relUpX0 - phaseInX, up);
                    int relInY0 = CEIL_DIV(relUpY0 - phaseInY, up);
                    int src0 = relInX0 + tileInW * relInY0;
                    int tap0y = (relInY0 * up + phaseInY - relUpY0);

#define X_LOOP(TAPY, PX)                                                       \
    for (int sx = 0; sx < fuSize / up; sx++)                                   \
    {                                                                          \
        v.x() += a * (scalar_t)c_fu[(sx * up + (((PX)-0) & (up - 1))) +        \
                                    (sy * up + (TAPY)) * MAX_FILTER_SIZE];     \
        v.z() += b * (scalar_t)c_fu[(sx * up + (((PX)-0) & (up - 1))) +        \
                                    (sy * up + (TAPY)) * MAX_FILTER_SIZE];     \
            if ((PX) == 0) {                                                   \
                a = b; b = s_tileIn[src0 + 2 + sx + sy * tileInW];             \
            }                                                                  \
        v.y() += a * (scalar_t)c_fu[(sx * up + (((PX)-1) & (up - 1))) +        \
                                    (sy * up + (TAPY)) * MAX_FILTER_SIZE];     \
        v.w() += b * (scalar_t)c_fu[(sx * up + (((PX)-1) & (up - 1))) +        \
                                    (sy * up + (TAPY)) * MAX_FILTER_SIZE];     \
            if ((PX) == 1) {                                                   \
                a = b; b = s_tileIn[src0 + 2 + sx + sy * tileInW];             \
            }                                                                  \
    }

                    vec4_t v = InternalType<T>::zero_vec4();
                    if (tap0y == 0 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t a = s_tileIn[src0 + sy * tileInW]; scalar_t b = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(0, 0) }
                    if (tap0y == 0 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t a = s_tileIn[src0 + sy * tileInW]; scalar_t b = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(0, 1) }
                    if (tap0y == 1 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t a = s_tileIn[src0 + sy * tileInW]; scalar_t b = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(1, 0) }
                    if (tap0y == 1 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t a = s_tileIn[src0 + sy * tileInW]; scalar_t b = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(1, 1) }

                    #undef X_LOOP

                    int x = tileOutX * down + relUpX0;
                    int y = tileOutY * down + relUpY0;
                    int signX = x + p.sOfs.x();
                    int signY = y + p.sOfs.y();
                    int signZ = item_ct1.get_group(0) + p.blockZofs;
                    int signXb = signX >> 2;
                    index_t si =
                        signXb +
                        p.sShape.x() * (signY + (index_t)p.sShape.y() * signZ);

                    v.x() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.z() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.w() *= (scalar_t)((float)up * (float)up * p.gain);

                    if (signWrite)
                    {
                        if (!enableWriteSkip)
                        {
                            // Determine and write signs.
                            int sx = sycl::bit_cast<unsigned int>(v.x()) >> 31;
                            int sy = sycl::bit_cast<unsigned int>(v.y()) >> 31;
                            int sz = sycl::bit_cast<unsigned int>(v.z()) >> 31;
                            int sw = sycl::bit_cast<unsigned int>(v.w()) >> 31;
                            if (sx) v.x() *= p.slope; if (sycl::fabs(v.x()) >
                                                          p.clamp) {
                                sx = 2;
                                v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                            }
                            if (sy) v.y() *= p.slope; if (sycl::fabs(v.y()) >
                                                          p.clamp) {
                                sy = 2;
                                v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                            }
                            if (sz) v.z() *= p.slope; if (sycl::fabs(v.z()) >
                                                          p.clamp) {
                                sz = 2;
                                v.z() = InternalType<T>::clamp(v.z(), p.clamp);
                            }
                            if (sw) v.w() *= p.slope; if (sycl::fabs(v.w()) >
                                                          p.clamp) {
                                sw = 2;
                                v.w() = InternalType<T>::clamp(v.w(), p.clamp);
                            }

                            if ((uint32_t)signXb < p.swLimit &&
                                (uint32_t)signY < p.sShape.y() && signY >= minY)
                            {
                                p.s[si] = sx + (sy << 2) + (sz << 4) + (sw << 6);
                            }
                        }
                        else
                        {
                            // Determine and write signs.
                            if ((uint32_t)signXb < p.swLimit &&
                                (uint32_t)signY < p.sShape.y() && signY >= minY)
                            {
                                int sx =
                                    sycl::bit_cast<unsigned int>(v.x()) >> 31;
                                int sy =
                                    sycl::bit_cast<unsigned int>(v.y()) >> 31;
                                int sz =
                                    sycl::bit_cast<unsigned int>(v.z()) >> 31;
                                int sw =
                                    sycl::bit_cast<unsigned int>(v.w()) >> 31;
                                if (sx) v.x() *= p.slope;
                                    if (sycl::fabs(v.x()) > p.clamp) {
                                        sx = 2; v.x() = InternalType<T>::clamp(
                                                    v.x(), p.clamp);
                                    }
                                if (sy) v.y() *= p.slope;
                                    if (sycl::fabs(v.y()) > p.clamp) {
                                        sy = 2; v.y() = InternalType<T>::clamp(
                                                    v.y(), p.clamp);
                                    }
                                if (sz) v.z() *= p.slope;
                                    if (sycl::fabs(v.z()) > p.clamp) {
                                        sz = 2; v.z() = InternalType<T>::clamp(
                                                    v.z(), p.clamp);
                                    }
                                if (sw) v.w() *= p.slope;
                                    if (sycl::fabs(v.w()) > p.clamp) {
                                        sw = 2; v.w() = InternalType<T>::clamp(
                                                    v.w(), p.clamp);
                                    }

                                p.s[si] = sx + (sy << 2) + (sz << 4) + (sw << 6);
                            }
                            else
                            {
                                // Just compute the values.
                                if (v.x() < 0.f) v.x() *= p.slope;
                                    v.x() =
                                        InternalType<T>::clamp(v.x(), p.clamp);
                                if (v.y() < 0.f) v.y() *= p.slope;
                                    v.y() =
                                        InternalType<T>::clamp(v.y(), p.clamp);
                                if (v.z() < 0.f) v.z() *= p.slope;
                                    v.z() =
                                        InternalType<T>::clamp(v.z(), p.clamp);
                                if (v.w() < 0.f) v.w() *= p.slope;
                                    v.w() =
                                        InternalType<T>::clamp(v.w(), p.clamp);
                            }
                        }
                    }
                    else if (signRead) // Read sign and apply.
                    {
                        if ((uint32_t)signY < p.sShape.y())
                        {
                            int s = 0;
                            if ((uint32_t)signXb     < p.swLimit) s  = p.s[si];
                            if ((uint32_t)signXb + 1 < p.swLimit) s |= p.s[si + 1] << 8;
                            s >>= (signX & 3) << 1;
                            if (s & 0x01) v.x() *= p.slope;
                                if (s & 0x02) v.x() = 0.f;
                            if (s & 0x04) v.y() *= p.slope;
                                if (s & 0x08) v.y() = 0.f;
                            if (s & 0x10) v.z() *= p.slope;
                                if (s & 0x20) v.z() = 0.f;
                            if (s & 0x40) v.w() *= p.slope;
                                if (s & 0x80) v.w() = 0.f;
                        }
                    }
                    else // Forward pass with no sign write.
                    {
                        if (v.x() < 0.f) v.x() *= p.slope;
                            v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                        if (v.y() < 0.f) v.y() *= p.slope;
                            v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                        if (v.z() < 0.f) v.z() *= p.slope;
                            v.z() = InternalType<T>::clamp(v.z(), p.clamp);
                        if (v.w() < 0.f) v.w() *= p.slope;
                            v.w() = InternalType<T>::clamp(v.w(), p.clamp);
                    }

                    s_tileUpXY[idx + 0] = v.x();
                    s_tileUpXY[idx + 1] = v.y();
                    s_tileUpXY[idx + 2] = v.z();
                    s_tileUpXY[idx + 3] = v.w();
                }
            }
            else if (up == 1)
            {
                item_ct1.barrier(sycl::access::fence_space::local_space);
                uint32_t groupMask = 15
                                     << ((item_ct1.get_local_id(2) & 31) & ~3);
                int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH : 0; // Skip already written signs.
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileUpW * tileUpH;
                     idx += item_ct1.get_local_range(2))
                {
                    int relUpX0, relUpY0;
                    fast_div_mod<tileUpW>(relUpX0, relUpY0, idx);
                    scalar_t v = s_tileIn[idx] * (scalar_t)c_fu[0]; // 1x1 filter.

                    int x = tileOutX * down + relUpX0;
                    int y = tileOutY * down + relUpY0;
                    int signX = x + p.sOfs.x();
                    int signY = y + p.sOfs.y();
                    int signZ = item_ct1.get_group(0) + p.blockZofs;
                    int signXb = signX >> 2;
                    index_t si =
                        signXb +
                        p.sShape.x() * (signY + (index_t)p.sShape.y() * signZ);
                    v *= (scalar_t)((float)up * (float)up * p.gain);

                    if (signWrite)
                    {
                        if (!enableWriteSkip)
                        {
                            // Determine and write sign.
                            uint32_t s = 0;
                            uint32_t signXbit = (1u << signXo);
                            if (v < 0.f)
                            {
                                s = signXbit;
                                v *= p.slope;
                            }
                            if (sycl::fabs(v) > p.clamp)
                            {
                                s = signXbit * 2;
                                v = InternalType<T>::clamp(v, p.clamp);
                            }
                            if ((uint32_t)signXb < p.swLimit &&
                                (uint32_t)signY < p.sShape.y() && signY >= minY)
                            {
                                /*
                                DPCT1023:17: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:48: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s += dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s,
                                    1); // Coalesce.
                                /*
                                DPCT1023:18: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:49: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s += dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s,
                                    2); // Coalesce.
                                p.s[si] = s;                            // Write.
                            }
                        }
                        else
                        {
                            // Determine and write sign.
                            if ((uint32_t)signXb < p.swLimit &&
                                (uint32_t)signY < p.sShape.y() && signY >= minY)
                            {
                                uint32_t s = 0;
                                uint32_t signXbit = (1u << signXo);
                                if (v < 0.f)
                                {
                                    s = signXbit;
                                    v *= p.slope;
                                }
                                if (sycl::fabs(v) > p.clamp)
                                {
                                    s = signXbit * 2;
                                    v = InternalType<T>::clamp(v, p.clamp);
                                }
                                /*
                                DPCT1023:19: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:50: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s += dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s,
                                    1); // Coalesce.
                                /*
                                DPCT1023:20: The SYCL sub-group does not support
                                mask options for dpct::permute_sub_group_by_xor.
                                You can specify
                                "--use-experimental-features=masked-sub-group-operation"
                                to use the experimental helper function to
                                migrate __shfl_xor_sync.
                                */
                                /*
                                DPCT1096:51: The right-most dimension of the
                                work-group used in the SYCL kernel that calls
                                this function may be less than "32". The
                                function "dpct::permute_sub_group_by_xor" may
                                return an unexpected result on the CPU device.
                                Modify the size of the work-group to ensure that
                                the value of the right-most dimension is a
                                multiple of "32".
                                */
                                s += dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), s,
                                    2); // Coalesce.
                                p.s[si] = s;                            // Write.
                            }
                            else
                            {
                                // Just compute the value.
                                if (v < 0.f) v *= p.slope;
                                v = InternalType<T>::clamp(v, p.clamp);
                            }
                        }
                    }
                    else if (signRead)
                    {
                        // Read sign and apply if within sign tensor bounds.
                        if ((uint32_t)signXb < p.swLimit &&
                            (uint32_t)signY < p.sShape.y())
                        {
                            int s = p.s[si];
                            s >>= signXo;
                            if (s & 1) v *= p.slope;
                            if (s & 2) v = 0.f;
                        }
                    }
                    else // Forward pass with no sign write.
                    {
                        if (v < 0.f) v *= p.slope;
                        v = InternalType<T>::clamp(v, p.clamp);
                    }

                    if (!downInline) // Write into temporary buffer.
                        s_tileUpXY[idx] = v;
                    else if ((uint32_t)x < p.yShape.x() &&
                             (uint32_t)y <
                                 p.yShape
                                     .y()) // Write directly into output buffer
                        (T&)(yacc[x * get_stride<index_t>(p.yStride.x()) +
                             y * get_stride<index_t>(p.yStride.y()) +
                             mapOfsOut]) = (T)(v * (scalar_t)c_fd[0]);
                }
            }
        }

        // Downsampling.
        if (filterMode == MODE_SUSD || filterMode == MODE_FUSD)
        {
            // Horizontal downsampling.
            item_ct1.barrier(sycl::access::fence_space::local_space);
            if (down == 4 && tileOutW % 4 == 0)
            {
                // Calculate 4 pixels at a time.
                for (int idx = item_ct1.get_local_id(2) * 4;
                     idx < tileOutW * tileUpH;
                     idx += item_ct1.get_local_range(2) * 4)
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src0 = relUpY * tileUpW + relUpX0;
                    vec4_t v = InternalType<T>::zero_vec4();
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                    {
                        v.x() +=
                            s_tileUpXY[src0 + 0 + step] * (scalar_t)c_fd[step];
                        v.y() +=
                            s_tileUpXY[src0 + 4 + step] * (scalar_t)c_fd[step];
                        v.z() +=
                            s_tileUpXY[src0 + 8 + step] * (scalar_t)c_fd[step];
                        v.w() +=
                            s_tileUpXY[src0 + 12 + step] * (scalar_t)c_fd[step];
                    }
                    s_tileDownX[idx + 0] = v.x();
                    s_tileDownX[idx + 1] = v.y();
                    s_tileDownX[idx + 2] = v.z();
                    s_tileDownX[idx + 3] = v.w();
                }
            }
            else if ((down == 2 || down == 4) && (tileOutW % 2 == 0))
            {
                // Calculate 2 pixels at a time.
                for (int idx = item_ct1.get_local_id(2) * 2;
                     idx < tileOutW * tileUpH;
                     idx += item_ct1.get_local_range(2) * 2)
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src0 = relUpY * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                    {
                        v.x() +=
                            s_tileUpXY[src0 + 0 + step] * (scalar_t)c_fd[step];
                        v.y() += s_tileUpXY[src0 + down + step] *
                                 (scalar_t)c_fd[step];
                    }
                    s_tileDownX[idx + 0] = v.x();
                    s_tileDownX[idx + 1] = v.y();
                }
            }
            else
            {
                // Calculate 1 pixel at a time.
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileOutW * tileUpH;
                     idx += item_ct1.get_local_range(2))
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src = relUpY * tileUpW + relUpX0;
                    scalar_t v = 0.f;
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                        v += s_tileUpXY[src + step] * (scalar_t)c_fd[step];
                    s_tileDownX[idx] = v;
                }
            }

            // Vertical downsampling & store output tile.
            item_ct1.barrier(sycl::access::fence_space::local_space);
            for (int idx = item_ct1.get_local_id(2); idx < tileOutW * tileOutH;
                 idx += item_ct1.get_local_range(2))
            {
                int relOutX, relOutY0;
                fast_div_mod<tileOutW>(relOutX, relOutY0, idx);
                int relUpY0 = relOutY0 * down;
                int src0 = relUpY0 * tileOutW + relOutX;
                scalar_t v = 0;
                #pragma unroll
                for (int step = 0; step < fdSize; step++)
                    v += s_tileDownX[src0 + step * tileOutW] * (scalar_t)c_fd[step];

                int outX = tileOutX + relOutX;
                int outY = tileOutY + relOutY0;

                if (outX < p.yShape.x() & outY < p.yShape.y())
                    (T&)(yacc[outX * get_stride<index_t>(p.yStride.x()) +
                         outY * get_stride<index_t>(p.yStride.y()) +
                         mapOfsOut]) = (T)v;
            }
        }
        else if (filterMode == MODE_SUFD || filterMode == MODE_FUFD)
        {
            // Full downsampling filter.
            if (down == 2)
            {
                // 2-wide.
                item_ct1.barrier(sycl::access::fence_space::local_space);
                for (int idx = item_ct1.get_local_id(2) * 2;
                     idx < tileOutW * tileOutH;
                     idx += item_ct1.get_local_range(2) * 2)
                {
                    int relOutX0, relOutY0;
                    fast_div_mod<tileOutW>(relOutX0, relOutY0, idx);
                    int relUpX0 = relOutX0 * down;
                    int relUpY0 = relOutY0 * down;
                    int src0 = relUpY0 * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    #pragma unroll
                    for (int sy = 0; sy < fdSize; sy++)
                    #pragma unroll
                    for (int sx = 0; sx < fdSize; sx++)
                    {
                        v.x() += s_tileUpXY[src0 + 0 + sx + sy * tileUpW] *
                                 (scalar_t)c_fd[sx + sy * MAX_FILTER_SIZE];
                        v.y() += s_tileUpXY[src0 + 2 + sx + sy * tileUpW] *
                                 (scalar_t)c_fd[sx + sy * MAX_FILTER_SIZE];
                    }

                    int outX = tileOutX + relOutX0;
                    int outY = tileOutY + relOutY0;
                    if ((uint32_t)outY < p.yShape.y())
                    {
                        index_t ofs =
                            outX * get_stride<index_t>(p.yStride.x()) +
                            outY * get_stride<index_t>(p.yStride.y()) +
                            mapOfsOut;
                        if (outX + 0 < p.yShape.x())
                            (T&)(yacc[ofs]) = (T)v.x();
                        if (outX + 1 < p.yShape.x())
                            (T&)(yacc[ofs + get_stride<index_t>(p.yStride.x())]) = (T)v.y();
                    }
                }
            }
            else if (down == 1 && !downInline)
            {
                // Thread per pixel.
                item_ct1.barrier(sycl::access::fence_space::local_space);
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileOutW * tileOutH;
                     idx += item_ct1.get_local_range(2))
                {
                    int relOutX0, relOutY0;
                    fast_div_mod<tileOutW>(relOutX0, relOutY0, idx);
                    scalar_t v = s_tileUpXY[idx] * (scalar_t)c_fd[0]; // 1x1 filter.

                    int outX = tileOutX + relOutX0;
                    int outY = tileOutY + relOutY0;
                    if ((uint32_t)outX < p.yShape.x() &&
                        (uint32_t)outY < p.yShape.y())
                        (T&)(yacc[outX * get_stride<index_t>(p.yStride.x()) +
                             outY * get_stride<index_t>(p.yStride.y()) +
                             mapOfsOut]) = (T)v;
                }
            }
        }

        if (!enableXrep)
            break;
    }
}

//------------------------------------------------------------------------
// Compute activation function and signs for upsampled data tensor, modifying data tensor in-place. Used for accelerating the generic variant.
// Sign tensor is known to be contiguous, and p.x and p.s have the same z, w dimensions. 64-bit indexing is always used.

template <class T, bool signWrite, bool signRead>
/*
DPCT1110:29: The total declared local variable size in device function
filtered_lrelu_act_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void filtered_lrelu_act_kernel(filtered_lrelu_act_kernel_params p,
                                      const sycl::nd_item<3> &item_ct1)
{
    typedef typename InternalType<T>::scalar_t scalar_t;

    // Indexing.
    int32_t x = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int32_t ymax = signWrite ? p.sShape.y() : p.xShape.y();
    int32_t qmax = p.xShape.z() *
                   p.xShape.w(); // Combined minibatch*channel maximum index.

    // Loop to accommodate oversized tensors.
    for (int32_t q = item_ct1.get_group(0); q < qmax;
         q += item_ct1.get_group_range(0))
    for (int32_t y = item_ct1.get_group(1); y < ymax;
         y += item_ct1.get_group_range(1))
    {
        // Extract z and w (channel, minibatch index).
        int32_t w = q / p.xShape.z();
        int32_t z = q - w * p.xShape.z();

        // Choose behavior based on sign read/write mode.
        if (signWrite)
        {
            // Process value if in p.x.
            uint32_t s = 0;
            if (x < p.xShape.x() && y < p.xShape.y())
            {
                int64_t ix = x * p.xStride.x() + y * p.xStride.y() +
                             z * p.xStride.z() + w * p.xStride.w();
                T* pv = ((T*)p.x) + ix;
                scalar_t v = (scalar_t)(*pv);

                // Gain, LReLU, clamp.
                v *= p.gain;
                if (v < 0.f)
                {
                    v *= p.slope;
                    s = 1; // Sign.
                }
                /*
                DPCT1064:38: Migrated fabsf call is used in a macro/template
                definition and may not be valid for all macro/template uses.
                Adjust the code.
                */
                if (sycl::fabs((float)v) > p.clamp)
                {
                    v = InternalType<T>::clamp(v, p.clamp);
                    s = 2; // Clamp.
                }

                *pv = (T)v; // Write value.
            }

            // Coalesce into threads 0 and 16 of warp.
            uint32_t m =
                (item_ct1.get_local_id(2) & 16) ? 0xffff0000u : 0x0000ffffu;
            s <<= ((item_ct1.get_local_id(2) & 15) << 1); // Shift into place.
            /*
            DPCT1023:25: The SYCL sub-group does not support mask options for
            dpct::permute_sub_group_by_xor. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the
            experimental helper function to migrate __shfl_xor_sync.
            */
            /*
            DPCT1096:52: The right-most dimension of the work-group used in the
            SYCL kernel that calls this function may be less than "32". The
            function "dpct::permute_sub_group_by_xor" may return an unexpected
            result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of
            "32".
            */
            s |= dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), s,
                                                1); // Distribute.
            /*
            DPCT1023:26: The SYCL sub-group does not support mask options for
            dpct::permute_sub_group_by_xor. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the
            experimental helper function to migrate __shfl_xor_sync.
            */
            /*
            DPCT1096:53: The right-most dimension of the work-group used in the
            SYCL kernel that calls this function may be less than "32". The
            function "dpct::permute_sub_group_by_xor" may return an unexpected
            result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of
            "32".
            */
            s |= dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), s, 2);
            /*
            DPCT1023:27: The SYCL sub-group does not support mask options for
            dpct::permute_sub_group_by_xor. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the
            experimental helper function to migrate __shfl_xor_sync.
            */
            /*
            DPCT1096:54: The right-most dimension of the work-group used in the
            SYCL kernel that calls this function may be less than "32". The
            function "dpct::permute_sub_group_by_xor" may return an unexpected
            result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of
            "32".
            */
            s |= dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), s, 4);
            /*
            DPCT1023:28: The SYCL sub-group does not support mask options for
            dpct::permute_sub_group_by_xor. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the
            experimental helper function to migrate __shfl_xor_sync.
            */
            /*
            DPCT1096:55: The right-most dimension of the work-group used in the
            SYCL kernel that calls this function may be less than "32". The
            function "dpct::permute_sub_group_by_xor" may return an unexpected
            result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of
            "32".
            */
            s |= dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), s, 8);

            // Write signs if leader and in p.s.
            if (!(item_ct1.get_local_id(2) & 15) &&
                x < p.sShape.x()) // y is always in.
            {
                uint64_t is = x + p.sShape.x() * (y + (int64_t)p.sShape.y() *
                                                          q); // Contiguous.
                ((uint32_t*)p.s)[is >> 4] = s;
            }
        }
        else if (signRead)
        {
            // Process value if in p.x.
            if (x < p.xShape.x()) // y is always in.
            {
                int64_t ix = x * p.xStride.x() + y * p.xStride.y() +
                             z * p.xStride.z() + w * p.xStride.w();
                T* pv = ((T*)p.x) + ix;
                scalar_t v = (scalar_t)(*pv);
                v *= p.gain;

                // Apply sign buffer offset.
                uint32_t sx = x + p.sOfs.x();
                uint32_t sy = y + p.sOfs.y();

                // Read and apply signs if we land inside valid region of sign buffer.
                if (sx < p.sShape.x() && sy < p.sShape.y())
                {
                    uint64_t is = (sx >> 2) + (p.sShape.x() >> 2) *
                                                  (sy + (uint64_t)p.sShape.y() *
                                                            q); // Contiguous.
                    unsigned char s = p.s[is];
                    s >>= (sx & 3) << 1; // Shift into place.
                    if (s & 1) // Sign?
                        v *= p.slope;
                    if (s & 2) // Clamp?
                        v = 0.f;
                }

                *pv = (T)v; // Write value.
            }
        }
        else
        {
            // Forward pass with no sign write. Process value if in p.x.
            if (x < p.xShape.x()) // y is always in.
            {
                int64_t ix = x * p.xStride.x() + y * p.xStride.y() +
                             z * p.xStride.z() + w * p.xStride.w();
                T* pv = ((T*)p.x) + ix;
                scalar_t v = (scalar_t)(*pv);
                v *= p.gain;
                if (v < 0.f)
                    v *= p.slope;
                /*
                DPCT1064:39: Migrated fabsf call is used in a macro/template
                definition and may not be valid for all macro/template uses.
                Adjust the code.
                */
                if (sycl::fabs((float)v) > p.clamp)
                    v = InternalType<T>::clamp(v, p.clamp);
                *pv = (T)v; // Write value.
            }
        }
    }
}

//------------------------------------------------------------------------
// XPU kernel launchers.

template <class T, class index_t, bool signWrite, bool signRead,
          int MODE, int U, int FU, int D, int FD, int TW, int TH, int W, int XR,
          int WS> // TODO: XR and WS could be just booleans and the actual value
                  // passed as a parameter - to reduce the number of template
                  // instantiations/specializations (but then duplicate
                  // specializations would need to be avoided)
void run_filtered_lrelu_kernel(filtered_lrelu_kernel_params &p) try {
    //std::cout << "run_filtered_lrelu_kernel" << std::endl;
    
#if TORCH_ATLEAST_2_3
    c10::xpu::XPUStream stream = c10::xpu::getCurrentXPUStream();
    auto& queue = stream.queue();
#else
    auto device_type = c10::DeviceType::XPU;
    c10::impl::VirtualGuardImpl impl(device_type);
    c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
    auto& queue = xpu::get_queue_from_stream(c10_stream);
#endif

    c_fbuf.init(queue);
    auto c_fbuf_ptr_ct1 = c_fbuf.get_ptr();

    // Launch XPU kernel.
    int bx = W * 32;
    int gx = (p.yShape.x() - 1) / TW + 1;
    int gy = (p.yShape.y() - 1) / TH + 1;
    int gz = p.yShape.z() * p.yShape.w();

    // Repeat multiple horizontal tiles in a CTA?
    if (XR)
    {
        p.tilesXrep = XR;
        p.tilesXdim = gx;
       
        gx = (gx + p.tilesXrep - 1) / p.tilesXrep;
        std::swap(gx, gy);
    }
    else
    {
        p.tilesXrep = 0;
        p.tilesXdim = 0;
    }

    // Launch filter setup kernel.
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
   auto task = queue.submit([&](sycl::handler &cgh) {
        g_fbuf.init(queue);

        auto g_fbuf_ptr_ct1 = g_fbuf.get_ptr();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1024),
                                           sycl::range<3>(1, 1, 1024)),
                         [=](sycl::nd_item<3> item_ct1) {
                           setup_filters_kernel(p, item_ct1,
                                                g_fbuf_ptr_ct1);
                         });
      });

    // Static definitions. - partially copied from inside the .cu (needed to compute e.g. the buffer sizes)
    typedef typename InternalType<T>::scalar_t scalar_t;
    const int tileUpW    = (TW * D + (FD - 1) - (D - 1) + 3) & ~3;  // Upsampled tile width, rounded up to multiple of 4.
    const int tileUpH    = TH * D + (FD - 1) - (D - 1);             // Upsampled tile height.
    const int tileInW    = CEIL_DIV(tileUpW  + (FU - 1), U);                   // Input tile width.
    const int tileInH    = CEIL_DIV(tileUpH  + (FU - 1), U);                   // Input tile height.
    const int tileUpH_up = CEIL_DIV(tileUpH, U) * U;                              // Upsampled tile height rounded up to a multiple of up.
    const int tileInH_up = CEIL_DIV(tileUpH_up + (FU - 1), U);                 // For allocations only, to avoid shared memory read overruns with up=2 and up=4.

    // Merge 1x1 downsampling into last upsampling step for upf1 and ups2.
    const bool downInline = (D == 1) && ((U == 1 && MODE == MODE_FUFD) || (U == 2 && MODE == MODE_SUFD));

    // Sizes of logical buffers.
    const int szIn    = tileInH_up * tileInW;
    const int szUpX   = tileInH_up * tileUpW;
    const int szUpXY  = downInline ? 0 : (tileUpH * tileUpW);
    const int szDownX = tileUpH * TW;

    // Sizes for shared memory arrays.
    const int s_buf0_size_base =
        (MODE == MODE_SUSD) ? MAX(szIn, szUpXY) :
        (MODE == MODE_FUSD) ? MAX(szIn, szDownX) :
        (MODE == MODE_SUFD) ? MAX(szIn, szUpXY) :
        (MODE == MODE_FUFD) ? szIn :
        -1;
    const int s_buf1_size_base =
        (MODE == MODE_SUSD) ? MAX(szUpX, szDownX) :
        (MODE == MODE_FUSD) ? szUpXY :
        (MODE == MODE_SUFD) ? szUpX  :
        (MODE == MODE_FUFD) ? szUpXY :
        -1;

    // Ensure U128 alignment.
    const int s_buf0_size = (s_buf0_size_base + 3) & ~3;
    const int s_buf1_size = (s_buf1_size_base + 3) & ~3;

    task.wait();
    
    // Copy kernels to constant memory.
    if      ( signWrite && !signRead) (copy_filters<true,  false>(queue));
    else if (!signWrite &&  signRead) (copy_filters<false, true >(queue));
    else if (!signWrite && !signRead) (copy_filters<false, false>(queue));

    // Launch main kernel.
    const int maxSubGz = 65535; // CUDA maximum for block z dimension.
    for (int zofs=0; zofs < gz; zofs += maxSubGz) // Do multiple launches if gz is too big.
    {
        p.blockZofs = zofs;
        int subGz = std::min(maxSubGz, gz - zofs);

        sycl::buffer<char> yBuf((char*)p.y, p.y_nbytes);
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
    queue.submit([&](sycl::handler &cgh) { // BASELINE: 1.42 it/s (stylegan3-r afhq2 512x512 inference in gen_video.py on A770)
          //const auto props = sycl::ext::oneapi::accessor_property_list{sycl::no_init, sycl::ext::oneapi::no_alias};
          //sycl::accessor yAccessor(yBuf, cgh, sycl::write_only, props);
          

          sycl::accessor yAccessor(yBuf, cgh, sycl::write_only, sycl::no_init);
          //sycl::accessor yAccessor(yBuf, cgh, sycl::write_only);
          //sycl::accessor yAccessor(yBuf, cgh);
//          T * ptr = yAccessor.get_pointer();

          sycl::local_accessor<scalar_t, 1> s_buf0_st_acc_ct1(
              s_buf0_size + s_buf1_size,
              cgh//, sycl::ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias}
              );
// return; // EARLY EXIT: ~ 24 it/s
          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(subGz, gy, gx) *
                                    sycl::range<3>(1, 1, bx),
                                sycl::range<3>(1, 1, bx)),
              [=](sycl::nd_item<3> item_ct1)
              // [[intel::reqd_sub_group_size(8)]] // GROUP SIZE TUNUNG: omitted: 7.45 it/s, 4: not supported, 8: 7.45 it/s, 16: 2.42it/s 32 (original): 1.42, 64 not supported
              { 
                filtered_lrelu_kernel<T, index_t, signWrite, signRead, MODE,
                                      U, FU, D, FD, TW, TH, W * 32, !!XR, !!WS>(
                    p, item_ct1, c_fbuf_ptr_ct1,
                    s_buf0_st_acc_ct1.get_pointer(), yAccessor
                    );
              });
        }).wait();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <class T, bool signWrite, bool signRead>
void run_filtered_lrelu_act_kernel(filtered_lrelu_act_kernel_params &p) try {
    //std::cout << "run_filtered_lrelu_act_kernel" << std::endl;
    // Launch kernel.
    void* args[] = {&p};
    int bx = 128; //1: 7.47 it/s, 64: 7.48 it/s, 128 (original): 7.49 it/s, 256: 7.48 it/s, 1024: 7.48 it/s, 4096: 7.47 it/s // 4 warps per block.

    // Logical size of launch = writeSigns ? p.s : p.x
    uint32_t gx = signWrite ? p.sShape.x() : p.xShape.x();
    uint32_t gy = signWrite ? p.sShape.y() : p.xShape.y();
    uint32_t gz =
        p.xShape.z() * p.xShape.w(); // Same as in p.sShape if signs are in use.
    gx = (gx - 1) / bx + 1;

    // Make sure grid y and z dimensions are within CUDA launch limits. Kernel loops internally to do the rest.
    const uint32_t gmax = 65535;
    gy = std::min(gy, gmax);
    gz = std::min(gz, gmax);

    // Launch.
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
#if TORCH_ATLEAST_2_3
    c10::xpu::XPUStream stream = c10::xpu::getCurrentXPUStream();
    auto& queue = stream.queue();
#else
    auto device_type = c10::DeviceType::XPU;
    c10::impl::VirtualGuardImpl impl(device_type);
    c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
    auto& queue = xpu::get_queue_from_stream(c10_stream);
#endif

    dpct::has_capability_or_fail(
        queue.get_device(),
        {sycl::aspect::fp64});
    queue.submit([&](sycl::handler &cgh) {
          auto p_ct0 = *(filtered_lrelu_act_kernel_params *)args[0];
//return; // EARLY EXIT: 7,48 it/s and correct result -> does not seem to be used in stylgan3-r inference; TODO what about training?
          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(gz, gy, gx) *
                                    sycl::range<3>(1, 1, bx),
                                sycl::range<3>(1, 1, bx)),
              [=](sycl::nd_item<3> item_ct1)
                  //[[intel::reqd_sub_group_size(32)]]
                  {
                    filtered_lrelu_act_kernel<T, signWrite, signRead>(p_ct0,
                                                                      item_ct1);
                  });
        }).wait();
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------
