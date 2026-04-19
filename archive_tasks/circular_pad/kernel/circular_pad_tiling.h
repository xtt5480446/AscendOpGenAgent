#ifndef CIRCULAR_PAD_TILING_H
#define CIRCULAR_PAD_TILING_H

#include <cstdint>

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct CircularPadTilingData {
    int64_t inputH;
    int64_t inputW;
    int64_t outputH;
    int64_t outputW;
    int64_t left;
    int64_t right;
    int64_t top;
    int64_t bottom;
    int64_t front;
    int64_t back;
    int64_t inputL;
    int64_t outputL;
    int64_t perCoreTaskNum;
    int64_t tailTaskNum;
    int64_t workspaceLen;
};

#endif
