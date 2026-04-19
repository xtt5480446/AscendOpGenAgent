#include <algorithm>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "circular_pad_tiling.h"

// 2D kernels
extern "C" void circular_pad_2d_small_int8_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_2d_big_int8_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_2d_small_half_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_2d_big_half_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_2d_small_float_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_2d_big_float_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_2d_small_int32_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_2d_big_int32_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);

// 3D kernels
extern "C" void circular_pad_3d_small_int8_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_3d_big_int8_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_3d_small_half_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_3d_big_half_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_3d_small_float_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_3d_big_float_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_3d_small_int32_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);
extern "C" void circular_pad_3d_big_int32_do(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling);

namespace circular_pad_ext {

using LaunchFn = void (*)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);

static inline int64_t GetAlign(int64_t len, int64_t size)
{
    if (size == 0) {
        return 0;
    }
    return (len * size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / size;
}

at::Tensor run_circular_pad(const at::Tensor& x, const std::vector<int64_t>& padding)
{
    TORCH_CHECK(x.dim() >= 3, "x must be at least 3D for circular_pad");
    TORCH_CHECK(padding.size() == 4 || padding.size() == 6, "padding must be 4 or 6 elements");

    const bool is3D = (padding.size() == 6);
    TORCH_CHECK(is3D ? x.dim() >= 4 : x.dim() >= 3, "x dim mismatch with padding length");

    int64_t left = padding[0];
    int64_t right = padding[1];
    int64_t top = padding[2];
    int64_t bottom = padding[3];
    int64_t front = 0;
    int64_t back = 0;
    if (is3D) {
        front = padding[4];
        back = padding[5];
    }

    int64_t tSize = 0;
    int32_t dataTypeMode = 0;
    if (x.scalar_type() == at::kChar) {
        tSize = sizeof(int8_t);
        dataTypeMode = 1;
    } else if (x.scalar_type() == at::kHalf) {
        tSize = sizeof(uint16_t);
        dataTypeMode = 2;
    } else if (x.scalar_type() == at::kFloat) {
        tSize = sizeof(float);
        dataTypeMode = 4;
    } else if (x.scalar_type() == at::kInt) {
        tSize = sizeof(int32_t);
        dataTypeMode = 5;
    } else {
        TORCH_CHECK(false, "unsupported dtype for circular_pad");
    }

    int64_t inputH = x.size(x.dim() - 2);
    int64_t inputW = x.size(x.dim() - 1);
    int64_t inputL = is3D ? x.size(x.dim() - 3) : 0;

    int64_t outputH = inputH + top + bottom;
    int64_t outputW = inputW + left + right;
    int64_t outputL = inputL + front + back;

    int64_t totalTasks = 1;
    int64_t batchDims = x.dim() - 2;
    for (int64_t i = 0; i < batchDims; ++i) {
        totalTasks *= x.size(i);
    }

    int64_t pLeft = left > 0 ? left : 0;
    int64_t pRight = right > 0 ? right : 0;
    int64_t pTop = top > 0 ? top : 0;
    int64_t pBottom = bottom > 0 ? bottom : 0;
    int64_t pFront = front > 0 ? front : 0;
    int64_t pBack = back > 0 ? back : 0;

    int64_t nLeft = left > 0 ? 0 : left;
    int64_t nRight = right > 0 ? 0 : right;
    int64_t nTop = top > 0 ? 0 : top;
    int64_t nBottom = bottom > 0 ? 0 : bottom;
    int64_t nFront = front > 0 ? 0 : front;
    int64_t nBack = back > 0 ? 0 : back;

    int64_t inputWAlign = GetAlign(inputW, tSize);
    int64_t leftAlign = GetAlign(pLeft, tSize);
    int64_t rightAlign = GetAlign(pRight, tSize);
    int64_t inOutputH = inputH + nTop + nBottom;
    int64_t inOutputW = inputW + nLeft + nRight;
    int64_t inOutputL = inputL + nFront + nBack;
    int64_t outputWAlign = GetAlign(outputW, tSize);
    int64_t inOutputWAlign = GetAlign(inOutputW, tSize);

    int32_t shapeType = 0;
    int64_t workspaceLen = 0;
    if (inOutputH * outputWAlign < static_cast<int64_t>(UB_SIZE) / BUFFER_NUM / tSize) {
        shapeType = 1;
        workspaceLen = inOutputH * (leftAlign + inOutputWAlign + rightAlign);
    } else {
        shapeType = 2;
        leftAlign = left > 0 ? leftAlign : BLOCK_SIZE / tSize;
        rightAlign = right > 0 ? rightAlign : BLOCK_SIZE / tSize;
        leftAlign = leftAlign > inputW ? pLeft : leftAlign;
        rightAlign = rightAlign > inputW ? pRight : rightAlign;
        workspaceLen = inOutputH * (leftAlign + rightAlign);
    }

    int64_t coreNum = DEFAULT_NUM_PHYSICAL_CORES;
    int64_t perCoreTaskNum = 0;
    int64_t tailTaskNum = 0;
    int64_t useCoreNum = 0;
    bool dPad = is3D;

    if (dPad) {
        int64_t batchNum = totalTasks / inputL;
        perCoreTaskNum = (batchNum / coreNum) * inputL;
        tailTaskNum = (batchNum % coreNum) * inputL;
        useCoreNum = perCoreTaskNum > 0 ? coreNum : (tailTaskNum / inputL);
    } else {
        perCoreTaskNum = totalTasks / coreNum;
        tailTaskNum = totalTasks % coreNum;
        useCoreNum = perCoreTaskNum > 0 ? coreNum : tailTaskNum;
    }
    if (useCoreNum <= 0) {
        useCoreNum = 1;
    }

    CircularPadTilingData tiling;
    tiling.inputH = inputH;
    tiling.inputW = inputW;
    tiling.outputH = outputH;
    tiling.outputW = outputW;
    tiling.left = left;
    tiling.right = right;
    tiling.top = top;
    tiling.bottom = bottom;
    tiling.front = front;
    tiling.back = back;
    tiling.inputL = inputL;
    tiling.outputL = outputL;
    tiling.perCoreTaskNum = perCoreTaskNum;
    tiling.tailTaskNum = tailTaskNum;
    tiling.workspaceLen = workspaceLen;

    std::vector<int64_t> outShape(x.sizes().begin(), x.sizes().end());
    outShape[x.dim() - 2] = outputH;
    outShape[x.dim() - 1] = outputW;
    if (is3D) {
        outShape[x.dim() - 3] = outputL;
    }
    at::Tensor y = at::empty(outShape, x.options());

    // workspace size follows official: totalTasks * workspaceLen * sizeof(float)
    size_t workspaceBytes = static_cast<size_t>(totalTasks * workspaceLen * sizeof(float));
    at::Tensor workspace = at::empty(
        {static_cast<int64_t>(workspaceBytes)},
        at::device(at::kPrivateUse1).dtype(at::kByte));

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(CircularPadTilingData))},
        at::device(at::kCPU).dtype(at::kByte));
    std::memcpy(tilingCpu.data_ptr(), &tiling, sizeof(CircularPadTilingData));
    at::Tensor tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    LaunchFn launch = nullptr;
    if (!is3D) {
        if (x.scalar_type() == at::kChar) {
            launch = (shapeType == 1) ? circular_pad_2d_small_int8_do : circular_pad_2d_big_int8_do;
        } else if (x.scalar_type() == at::kHalf) {
            launch = (shapeType == 1) ? circular_pad_2d_small_half_do : circular_pad_2d_big_half_do;
        } else if (x.scalar_type() == at::kFloat) {
            launch = (shapeType == 1) ? circular_pad_2d_small_float_do : circular_pad_2d_big_float_do;
        } else if (x.scalar_type() == at::kInt) {
            launch = (shapeType == 1) ? circular_pad_2d_small_int32_do : circular_pad_2d_big_int32_do;
        }
    } else {
        if (x.scalar_type() == at::kChar) {
            launch = (shapeType == 1) ? circular_pad_3d_small_int8_do : circular_pad_3d_big_int8_do;
        } else if (x.scalar_type() == at::kHalf) {
            launch = (shapeType == 1) ? circular_pad_3d_small_half_do : circular_pad_3d_big_half_do;
        } else if (x.scalar_type() == at::kFloat) {
            launch = (shapeType == 1) ? circular_pad_3d_small_float_do : circular_pad_3d_big_float_do;
        } else if (x.scalar_type() == at::kInt) {
            launch = (shapeType == 1) ? circular_pad_3d_small_int32_do : circular_pad_3d_big_int32_do;
        }
    }
    TORCH_CHECK(launch != nullptr, "failed to select circular_pad kernel");

    launch(
        static_cast<uint32_t>(useCoreNum),
        aclStream,
        static_cast<uint8_t*>(const_cast<void*>(x.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(y.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(workspace.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(tilingNpu.storage().data())));
    return y;
}

}  // namespace circular_pad_ext

PYBIND11_MODULE(_circular_pad_ext, m)
{
    m.doc() = "circular_pad extension";
    m.def("run_circular_pad", &circular_pad_ext::run_circular_pad, "");
}
