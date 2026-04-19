#include "circular_pad_kernel.h"

#define DEFINE_CIRCULAR_PAD_2D_KERNEL(name, dtype, process_fn)                               \
    extern "C" __global__ __aicore__ void circular_pad_2d_##name##_custom(                   \
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)                             \
    {                                                                                        \
        AscendC::TPipe pipe;                                                                 \
        CircularPad2D<dtype> op(&pipe);                                                      \
        const __gm__ CircularPadTilingData* tiling_data = reinterpret_cast<__gm__ CircularPadTilingData*>(tiling); \
        op.Init2D(x, nullptr, y, workspace, tiling_data);                                    \
        op.process_fn();                                                                     \
    }                                                                                \
    extern "C" void circular_pad_2d_##name##_do(                                   \
        uint32_t blockDim, void* stream,                                             \
        uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling)                 \
    {                                                                                \
        circular_pad_2d_##name##_custom<<<blockDim, nullptr, stream>>>(              \
            x, y, workspace, tiling);                                                \
    }

#define DEFINE_CIRCULAR_PAD_3D_KERNEL(name, dtype, process_fn)                               \
    extern "C" __global__ __aicore__ void circular_pad_3d_##name##_custom(                   \
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)                             \
    {                                                                                        \
        AscendC::TPipe pipe;                                                                 \
        CircularPad3D<dtype> op(&pipe);                                                      \
        const __gm__ CircularPadTilingData* tiling_data = reinterpret_cast<__gm__ CircularPadTilingData*>(tiling); \
        op.Init3D(x, nullptr, y, workspace, tiling_data);                                    \
        op.process_fn();                                                                     \
    }                                                                                \
    extern "C" void circular_pad_3d_##name##_do(                                   \
        uint32_t blockDim, void* stream,                                             \
        uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling)                 \
    {                                                                                \
        circular_pad_3d_##name##_custom<<<blockDim, nullptr, stream>>>(              \
            x, y, workspace, tiling);                                                \
    }

// 2D kernels
DEFINE_CIRCULAR_PAD_2D_KERNEL(small_int8, int8_t, ProcessSmallShape)
DEFINE_CIRCULAR_PAD_2D_KERNEL(big_int8, int8_t, ProcessBigShape)
DEFINE_CIRCULAR_PAD_2D_KERNEL(small_half, half, ProcessSmallShape)
DEFINE_CIRCULAR_PAD_2D_KERNEL(big_half, half, ProcessBigShape)
DEFINE_CIRCULAR_PAD_2D_KERNEL(small_float, float, ProcessSmallShape)
DEFINE_CIRCULAR_PAD_2D_KERNEL(big_float, float, ProcessBigShape)
DEFINE_CIRCULAR_PAD_2D_KERNEL(small_int32, int32_t, ProcessSmallShape)
DEFINE_CIRCULAR_PAD_2D_KERNEL(big_int32, int32_t, ProcessBigShape)

// 3D kernels
DEFINE_CIRCULAR_PAD_3D_KERNEL(small_int8, int8_t, ProcessSmallShape)
DEFINE_CIRCULAR_PAD_3D_KERNEL(big_int8, int8_t, ProcessBigShape)
DEFINE_CIRCULAR_PAD_3D_KERNEL(small_half, half, ProcessSmallShape)
DEFINE_CIRCULAR_PAD_3D_KERNEL(big_half, half, ProcessBigShape)
DEFINE_CIRCULAR_PAD_3D_KERNEL(small_float, float, ProcessSmallShape)
DEFINE_CIRCULAR_PAD_3D_KERNEL(big_float, float, ProcessBigShape)
DEFINE_CIRCULAR_PAD_3D_KERNEL(small_int32, int32_t, ProcessSmallShape)
DEFINE_CIRCULAR_PAD_3D_KERNEL(big_int32, int32_t, ProcessBigShape)
