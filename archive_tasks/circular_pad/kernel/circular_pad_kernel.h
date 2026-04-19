#ifndef CIRCULAR_PAD_KERNEL_H
#define CIRCULAR_PAD_KERNEL_H

#include "kernel_operator.h"
#include "circular_pad_tiling.h"

using namespace AscendC;

struct sDataCopyExtParams {
    DataCopyExtParams paramsIn = {0, 0, 0, 0, 0};
    DataCopyExtParams paramsOut = {0, 0, 0, 0, 0};
};

struct CopyParams {
    __aicore__ inline CopyParams(){};
    __aicore__ inline CopyParams(int64_t offset, int64_t strideLoop, DataCopyExtParams dcParams)
        : offset(offset), strideLoop(strideLoop), dcParams(dcParams){};
    int64_t offset{0};
    int64_t strideLoop{0};
    int64_t stridePage{0};
    DataCopyExtParams dcParams = {0, 0, 0, 0, 0};
};

struct LoopParams {
    int64_t loopW{0};
    int64_t loopH{0};
};

class CircularPadCommon {
public:
    __aicore__ inline CircularPadCommon(TPipe* pipe) : pipe_(pipe){};

    __aicore__ inline void InitCommon(const __gm__ CircularPadTilingData* tiling_data, int64_t T1Size, int64_t T2Size)
    {
        GetTiling(tiling_data);
        if (T2Size == 0) {
            return;
        }
        Align_ = BLOCK_SIZE / T2Size;
        inputWAlign_ = GetAlign(inputW_, T1Size);
        outputWAlign_ = GetAlign(outputW_, T1Size);
        inputLen_ = inputH_ * inputW_;
        outputLen_ = outputH_ * outputW_;

        pLeft_ = GetPositive(left_);
        pRight_ = GetPositive(right_);
        pTop_ = GetPositive(top_);
        pBottom_ = GetPositive(bottom_);
        nLeft_ = GetNegtive(left_);
        nRight_ = GetNegtive(right_);
        nTop_ = GetNegtive(top_);
        nBottom_ = GetNegtive(bottom_);

        leftAlign_ = GetAlign(pLeft_, T2Size);
        rightAlign_ = GetAlign(pRight_, T2Size);
    }

    __aicore__ inline void GetTiling(const __gm__ CircularPadTilingData* tiling_data)
    {
        inputH_ = tiling_data->inputH;
        inputW_ = tiling_data->inputW;
        outputH_ = tiling_data->outputH;
        outputW_ = tiling_data->outputW;
        left_ = tiling_data->left;
        right_ = tiling_data->right;
        top_ = tiling_data->top;
        bottom_ = tiling_data->bottom;
        perCoreTaskNum_ = tiling_data->perCoreTaskNum;
        workspaceLen_ = tiling_data->workspaceLen;
        tailTaskNum_ = tiling_data->tailTaskNum;
        workspaceLen_ = tiling_data->workspaceLen;
    }

    __aicore__ inline int64_t GetAlign(int64_t len, int64_t size)
    {
        if (size == 0) {
            return 0;
        }
        return (len * size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / size;
    }

    __aicore__ inline int64_t GetPositive(int64_t len)
    {
        return len > 0 ? len : 0;
    }

    __aicore__ inline int64_t GetNegtive(int64_t len)
    {
        return len < 0 ? len : 0;
    }

    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventId3To2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId3To2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId3To2);
    }

protected:
    TPipe* pipe_;
    int64_t inputH_{0};
    int64_t inputW_{0};
    int64_t outputH_{0};
    int64_t outputW_{0};
    int64_t left_{0};
    int64_t right_{0};
    int64_t top_{0};
    int64_t bottom_{0};
    int64_t perCoreTaskNum_{0};
    int64_t tailTaskNum_{0};
    int64_t workspaceLen_{0};

    uint8_t Align_{0};
    int64_t inputLen_{0};
    int64_t inputWAlign_{0};
    int64_t outputWAlign_{0};
    int64_t outputLen_{0};
    int64_t inOutputH_{0};
    int64_t inOutputW_{0};
    int64_t inOutputWAlign_{0};
    int64_t inOutputW32Align_{0};
    int64_t leftAlign_{0};
    int64_t rightAlign_{0};
    int64_t pLeft_{0};
    int64_t pRight_{0};
    int64_t pTop_{0};
    int64_t pBottom_{0};
    int64_t nLeft_{0};
    int64_t nRight_{0};
    int64_t nTop_{0};
    int64_t nBottom_{0};
};

template <typename T>
class CircularPad : public CircularPadCommon {
public:
    __aicore__ inline CircularPad(TPipe* pipe) : CircularPadCommon(pipe){};

    __aicore__ inline void Init(const __gm__ CircularPadTilingData* tiling_data)
    {
        TSize_ = sizeof(T);
        InitCommon(tiling_data, TSize_, TSize_);
        hasLR = (left_ > 0 || right_ > 0);
        inOutputH_ = inputH_ + nTop_ + nBottom_;
        inOutputW_ = inputW_ + nLeft_ + nRight_;
        inOutputWAlign_ = GetAlign(inOutputW_, TSize_);
        pipe_->InitBuffer(queBind_, BUFFER_NUM, UB_SIZE / BUFFER_NUM);
    }

    template <bool HASLR>
    __aicore__ inline void PadLeftAndRightSmallShape()
    {
        if constexpr (HASLR) {
            DataCopyExtParams paramsIn;
            DataCopyExtParams paramsOut;
            DataCopyExtParams paramsRight;

            int64_t offsetIn = -nTop_ * inputW_ - nLeft_;
            int64_t offsetOut = leftAlign_;
            int64_t offsetRight = leftAlign_ + inOutputW_;

            for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
                CalculateLeftAndRightParams(paramsIn, paramsOut, paramsRight);
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, xGM_[offsetIn], paramsIn, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsOut);

                if (right_ > 0) {
                    PipeBarrier<PIPE_MTE3>();
                    DataCopyPad(workspaceGM_[offsetRight], inLocal, paramsRight);
                    offsetRight += workspaceLen_;
                }
                queBind_.FreeTensor(inLocal);
                offsetIn += inputLen_;
                offsetOut += workspaceLen_;
            }
            MTE3ToMTE2Sync();
            if (left_ > 0) {
                paramsIn.blockLen = static_cast<uint32_t>(leftAlign_ * TSize_);
                paramsIn.srcStride = static_cast<uint32_t>((rightAlign_ + inOutputWAlign_) * TSize_);
                paramsOut.blockLen = static_cast<uint32_t>(leftAlign_ * TSize_);
                paramsOut.dstStride = static_cast<uint32_t>((leftAlign_ + inOutputWAlign_) * TSize_);
                offsetIn = inOutputW_;
                offsetOut = 0;

                for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
                    auto inLocal = queBind_.AllocTensor<T>();
                    DataCopyPad(inLocal, workspaceGM_[offsetIn], paramsIn, padParms);
                    queBind_.EnQue(inLocal);
                    inLocal = queBind_.DeQue<T>();
                    DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsOut);
                    queBind_.FreeTensor(inLocal);
                    offsetIn += workspaceLen_;
                    offsetOut += workspaceLen_;
                }
            }
        }
    }

    __aicore__ inline void CopyToOutSmallShapeOnePage(
        GlobalTensor<T>& srcGM, int64_t pageIdxOut, CopyParams& copyParamsIn, CopyParams& copyParamsOut)
    {
        auto inLocal = queBind_.AllocTensor<T>();
        DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
        queBind_.EnQue(inLocal);
        inLocal = queBind_.DeQue<T>();
        DataCopyPad(yGM_[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);

        if (top_ > 0) {
            copyParamsOut.dcParams.blockCount = static_cast<uint16_t>(top_);
            DataCopyPad(
                yGM_[pageIdxOut * outputLen_], inLocal[(inOutputH_ - top_) * outputWAlign_], copyParamsOut.dcParams);
        }
        if (bottom_ > 0) {
            copyParamsOut.dcParams.blockCount = static_cast<uint16_t>(bottom_);
            DataCopyPad(
                yGM_[pageIdxOut * outputLen_ + (outputH_ - bottom_) * outputW_], inLocal, copyParamsOut.dcParams);
        }
        queBind_.FreeTensor(inLocal);
    }

    template <bool HASLR>
    __aicore__ inline void PadLeftAndRightBigShape()
    {
        if constexpr (HASLR) {
            leftAlign_ = left_ > 0 ? leftAlign_ : Align_;
            rightAlign_ = right_ > 0 ? rightAlign_ : Align_;
            leftAlign_ = leftAlign_ > inputW_ ? pLeft_ : leftAlign_;
            rightAlign_ = rightAlign_ > inputW_ ? pRight_ : rightAlign_;

            LoopParams loopParams;
            loopParams.loopW = leftAlign_;
            loopParams.loopH = inOutputH_;
            CopyParams copyParamsIn;
            CopyParams copyParamsOut;
            uint32_t blockLen = static_cast<uint32_t>(leftAlign_ * TSize_);
            uint32_t srcStrideIn = static_cast<uint32_t>((inputW_ - leftAlign_) * TSize_);
            uint32_t dstStrideOut = static_cast<uint32_t>(rightAlign_ * TSize_);
            copyParamsIn.dcParams = {0, blockLen, srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = inputW_ - leftAlign_ + nRight_ - nTop_ * inputW_;
            copyParamsOut.offset = 0;
            copyParamsIn.strideLoop = inputW_;
            copyParamsOut.strideLoop = leftAlign_ + rightAlign_;
            copyParamsIn.stridePage = inputLen_;
            copyParamsOut.stridePage = workspaceLen_;
            CopyLines(xGM_, workspaceGM_, loopParams, copyParamsIn, copyParamsOut);

            loopParams.loopW = rightAlign_;
            blockLen = static_cast<uint32_t>(rightAlign_ * TSize_);
            srcStrideIn = static_cast<uint32_t>((inputW_ - rightAlign_) * TSize_);
            dstStrideOut = static_cast<uint32_t>(leftAlign_ * TSize_);
            copyParamsIn.dcParams = {0, blockLen, srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = -nLeft_ - nTop_ * inputW_;
            copyParamsOut.offset = leftAlign_;
            CopyLines(xGM_, workspaceGM_, loopParams, copyParamsIn, copyParamsOut);
        }
    }

    __aicore__ inline void CopyToOutBigShapeOnePage(int64_t pageIdxIn, int64_t pageIdxOut)
    {
        LoopParams loopParams;
        CopyParams copyParamsIn;
        CopyParams copyParamsOut;

        CopyWSToOutOnce(pageIdxIn, pageIdxOut, loopParams, copyParamsIn, copyParamsOut);
        MTE3ToMTE2Sync();

        CopyInToOutOnce(pageIdxIn, pageIdxOut, loopParams, copyParamsIn, copyParamsOut);
        MTE3ToMTE2Sync();

        PadTopAndBottomOnce(pageIdxIn, pageIdxOut, loopParams, copyParamsIn, copyParamsOut);
    }

    __aicore__ inline void CalculateLeftAndRightParams(
        DataCopyExtParams& paramsIn, DataCopyExtParams& paramsOut, DataCopyExtParams& paramsRight)
    {
        uint16_t blockCount = static_cast<uint16_t>(inOutputH_);
        uint32_t blockLenIn = static_cast<uint32_t>(inOutputW_ * TSize_);
        uint32_t srcStrideIn = static_cast<uint32_t>((-nLeft_ - nRight_) * TSize_);
        uint32_t blockLenOut = static_cast<uint32_t>(inOutputWAlign_ * TSize_);
        uint32_t dstStrideOut = static_cast<uint32_t>((leftAlign_ + rightAlign_) * TSize_);
        paramsIn = {blockCount, blockLenIn, srcStrideIn, 0, 0};
        paramsOut = {blockCount, blockLenOut, 0, dstStrideOut, 0};
        uint32_t blockLen = static_cast<uint32_t>(rightAlign_ * TSize_);
        uint32_t srcStride = static_cast<uint32_t>((inOutputWAlign_ - rightAlign_) * TSize_ / BLOCK_SIZE);
        uint32_t dstStride = static_cast<uint32_t>((leftAlign_ + inOutputWAlign_) * TSize_);
        paramsRight = {blockCount, blockLen, srcStride, dstStride, 0};
    }

    __aicore__ inline void CopyLines(
        GlobalTensor<T>& srcGM, GlobalTensor<T>& dstGM, LoopParams& loopParams, CopyParams& copyParamsIn,
        CopyParams& copyParamsOut)
    {
        uint16_t rowsNum = UB_SIZE / BUFFER_NUM / GetAlign(loopParams.loopW, TSize_) / TSize_;
        uint32_t loop = loopParams.loopH / rowsNum;
        uint16_t tail = loopParams.loopH % rowsNum;
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            for (uint32_t j = 0; j < loop; j++) {
                copyParamsIn.dcParams.blockCount = rowsNum;
                copyParamsOut.dcParams.blockCount = rowsNum;
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(dstGM[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);
                queBind_.FreeTensor(inLocal);
                copyParamsIn.offset += rowsNum * copyParamsIn.strideLoop;
                copyParamsOut.offset += rowsNum * copyParamsOut.strideLoop;
            }
            if (tail > 0) {
                copyParamsIn.dcParams.blockCount = tail;
                copyParamsOut.dcParams.blockCount = tail;
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(dstGM[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);
                queBind_.FreeTensor(inLocal);
            }
            copyParamsIn.offset += (copyParamsIn.stridePage - loop * rowsNum * copyParamsIn.strideLoop);
            copyParamsOut.offset += (copyParamsOut.stridePage - loop * rowsNum * copyParamsOut.strideLoop);
        }
    }

    __aicore__ inline void CopyWSToOutOnce(
        int64_t pageIdxIn, int64_t pageIdxOut, LoopParams& loopParams, CopyParams& copyParamsIn,
        CopyParams& copyParamsOut)
    {
        leftAlign_ = left_ > 0 ? leftAlign_ : Align_;
        rightAlign_ = right_ > 0 ? rightAlign_ : Align_;
        leftAlign_ = leftAlign_ > inputW_ ? pLeft_ : leftAlign_;
        rightAlign_ = rightAlign_ > inputW_ ? pRight_ : rightAlign_;
        loopParams.loopH = inOutputH_;
        copyParamsIn.strideLoop = leftAlign_ + rightAlign_;
        copyParamsOut.strideLoop = outputW_;
        copyParamsIn.stridePage = workspaceLen_;
        copyParamsOut.stridePage = outputLen_;
        if (left_ > 0) {
            loopParams.loopW = leftAlign_;
            uint32_t blockLen = static_cast<uint32_t>(leftAlign_ * TSize_);
            uint32_t srcStrideIn = static_cast<uint32_t>(rightAlign_ * TSize_);
            uint32_t dstStrideOut = static_cast<uint32_t>((outputW_ - leftAlign_) * TSize_);
            copyParamsIn.dcParams = {0, blockLen, srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = pageIdxIn * workspaceLen_ + leftAlign_ - left_;
            copyParamsOut.offset = pageIdxOut * outputLen_ + pTop_ * outputW_;
            CopyOnePage(workspaceGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
        if (right_ > 0) {
            loopParams.loopW = rightAlign_;
            uint32_t blockLen = static_cast<uint32_t>(rightAlign_ * TSize_);
            uint32_t srcStrideIn = static_cast<uint32_t>(leftAlign_ * TSize_);
            uint32_t dstStrideOut = static_cast<uint32_t>((outputW_ - rightAlign_) * TSize_);
            copyParamsIn.dcParams = {0, blockLen, srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = pageIdxIn * workspaceLen_ + leftAlign_ + right_ - rightAlign_;
            copyParamsOut.offset = pageIdxOut * outputLen_ + pTop_ * outputW_ + outputW_ - rightAlign_;
            CopyOnePage(workspaceGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
    }

    __aicore__ inline void CopyInToOutOnce(
        int64_t pageIdxIn, int64_t pageIdxOut, LoopParams& loopParams, CopyParams& copyParamsIn,
        CopyParams& copyParamsOut)
    {
        leftAlign_ = left_ > 0 ? leftAlign_ : 0;
        rightAlign_ = right_ > 0 ? rightAlign_ : 0;
        int64_t holeW = inOutputW_ - (leftAlign_ - pLeft_) - (rightAlign_ - pRight_);
        if (holeW > 0) {
            loopParams.loopW = inOutputWAlign_;
            uint32_t blockLen = static_cast<uint32_t>(holeW * TSize_);
            uint32_t srcStrideIn = static_cast<uint32_t>((inputW_ - holeW) * TSize_);
            uint32_t dstStrideOut = static_cast<uint32_t>((leftAlign_ + rightAlign_) * TSize_);
            copyParamsIn.dcParams = {0, blockLen, srcStrideIn, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, dstStrideOut, 0};
            copyParamsIn.offset = pageIdxIn * inputLen_ + leftAlign_ - pLeft_ - nLeft_ - nTop_ * inputW_;
            copyParamsOut.offset = pageIdxOut * outputLen_ + pTop_ * outputW_ + leftAlign_;
            copyParamsIn.strideLoop = inputW_;
            CopyOnePage(xGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
    }

    __aicore__ inline void PadTopAndBottomOnce(
        int64_t pageIdxIn, int64_t pageIdxOut, LoopParams& loopParams, CopyParams& copyParamsIn,
        CopyParams& copyParamsOut)
    {
        loopParams.loopW = outputWAlign_;
        copyParamsIn.strideLoop = outputW_;
        copyParamsOut.strideLoop = outputW_;
        copyParamsIn.stridePage = outputLen_;
        copyParamsOut.stridePage = outputLen_;
        if (top_ > 0) {
            loopParams.loopH = top_;
            uint32_t blockLen = static_cast<uint32_t>(outputW_ * TSize_);
            copyParamsIn.dcParams = {0, blockLen, 0, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, 0, 0};
            copyParamsIn.offset = pageIdxOut * outputLen_ + (outputH_ - pBottom_ - top_) * outputW_;
            copyParamsOut.offset = pageIdxOut * outputLen_;
            CopyOnePage(yGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
        if (bottom_ > 0) {
            loopParams.loopH = bottom_;
            uint32_t blockLen = static_cast<uint32_t>(outputW_ * TSize_);
            copyParamsIn.dcParams = {0, blockLen, 0, 0, 0};
            copyParamsOut.dcParams = {0, blockLen, 0, 0, 0};
            copyParamsIn.offset = pageIdxOut * outputLen_ + pTop_ * outputW_;
            copyParamsOut.offset = pageIdxOut * outputLen_ + (outputH_ - bottom_) * outputW_;
            CopyOnePage(yGM_, yGM_, loopParams, copyParamsIn, copyParamsOut);
        }
    }

    __aicore__ inline void CopyOnePage(
        GlobalTensor<T>& srcGM, GlobalTensor<T>& dstGM, LoopParams loopParams, CopyParams& copyParamsIn,
        CopyParams& copyParamsOut)
    {
        uint16_t rowsNum = UB_SIZE / BUFFER_NUM / GetAlign(loopParams.loopW, TSize_) / TSize_;
        uint32_t loop = loopParams.loopH / rowsNum;
        uint16_t tail = loopParams.loopH % rowsNum;
        for (uint32_t j = 0; j < loop; j++) {
            copyParamsIn.dcParams.blockCount = rowsNum;
            copyParamsOut.dcParams.blockCount = rowsNum;
            auto inLocal = queBind_.AllocTensor<T>();
            DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
            queBind_.EnQue(inLocal);
            inLocal = queBind_.DeQue<T>();
            DataCopyPad(dstGM[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);
            queBind_.FreeTensor(inLocal);
            copyParamsIn.offset += rowsNum * copyParamsIn.strideLoop;
            copyParamsOut.offset += rowsNum * copyParamsOut.strideLoop;
        }
        if (tail > 0) {
            copyParamsIn.dcParams.blockCount = tail;
            copyParamsOut.dcParams.blockCount = tail;
            auto inLocal = queBind_.AllocTensor<T>();
            DataCopyPad(inLocal, srcGM[copyParamsIn.offset], copyParamsIn.dcParams, padParms);
            queBind_.EnQue(inLocal);
            inLocal = queBind_.DeQue<T>();
            DataCopyPad(dstGM[copyParamsOut.offset], inLocal, copyParamsOut.dcParams);
            queBind_.FreeTensor(inLocal);
        }
    }

    __aicore__ inline void CopyGmToGm(
        int64_t pages, int64_t taskNum, int64_t offsetIn, int64_t offsetOut, int64_t stride)
    {
        int64_t loop = (outputLen_ * pages * TSize_) / (UB_SIZE / BUFFER_NUM);
        uint32_t tail = (outputLen_ * pages * TSize_) % (UB_SIZE / BUFFER_NUM);
        for (int64_t i = 0; i < taskNum; i++) {
            DataCopyExtParams paramsFront = {1, UB_SIZE / BUFFER_NUM, 0, 0, 0};
            for (int64_t j = 0; j < loop; j++) {
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, yGM_[offsetIn], paramsFront, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(yGM_[offsetOut], inLocal, paramsFront);
                queBind_.FreeTensor(inLocal);
                offsetIn += (UB_SIZE / BUFFER_NUM / TSize_);
                offsetOut += (UB_SIZE / BUFFER_NUM / TSize_);
            }
            if (tail > 0) {
                paramsFront.blockLen = tail;
                auto inLocal = queBind_.AllocTensor<T>();
                DataCopyPad(inLocal, yGM_[offsetIn], paramsFront, padParms);
                queBind_.EnQue(inLocal);
                inLocal = queBind_.DeQue<T>();
                DataCopyPad(yGM_[offsetOut], inLocal, paramsFront);
                queBind_.FreeTensor(inLocal);
            }
            offsetIn += (stride - loop * (UB_SIZE / BUFFER_NUM / TSize_));
            offsetOut += (stride - loop * (UB_SIZE / BUFFER_NUM / TSize_));
        }
    }

protected:
    bool hasLR{true};
    uint8_t TSize_{0};
    DataCopyPadExtParams<T> padParms = {false, 0, 0, 0};
    GlobalTensor<T> xGM_;
    GlobalTensor<T> yGM_;
    GlobalTensor<T> workspaceGM_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind_;
};

template <typename T>
class CircularPad2D : public CircularPad<T> {
public:
    __aicore__ inline CircularPad2D(TPipe* pipe) : CircularPad<T>(pipe){};

    __aicore__ inline void Init2D(
        GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace,
        const __gm__ CircularPadTilingData* tiling_data)
    {
        this->Init(tiling_data);
        uint32_t blockId = static_cast<uint32_t>(GetBlockIdx());
        uint32_t startIdx = this->perCoreTaskNum_ * blockId;
        if (blockId < this->tailTaskNum_) {
            this->perCoreTaskNum_ += 1;
            startIdx += blockId;
        } else {
            startIdx += this->tailTaskNum_;
        }
        this->xGM_.SetGlobalBuffer((__gm__ T*)x + this->inputLen_ * startIdx, this->inputLen_ * this->perCoreTaskNum_);
        this->yGM_.SetGlobalBuffer(
            (__gm__ T*)y + this->outputLen_ * startIdx, this->outputLen_ * this->perCoreTaskNum_);
        this->workspaceGM_.SetGlobalBuffer(
            (__gm__ T*)workspace + this->workspaceLen_ * startIdx, this->workspaceLen_ * this->perCoreTaskNum_);
    }

    __aicore__ inline void ProcessSmallShape()
    {
        if (this->hasLR) {
            this->template PadLeftAndRightSmallShape<true>();
            this->MTE3ToMTE2Sync();
            CopyToOutSmallShape<true>();
        } else {
            this->template PadLeftAndRightSmallShape<false>();
            this->MTE3ToMTE2Sync();
            CopyToOutSmallShape<false>();
        }
    }

    __aicore__ inline void ProcessBigShape()
    {
        if (this->hasLR) {
            this->template PadLeftAndRightBigShape<true>();
        } else {
            this->template PadLeftAndRightBigShape<false>();
        }
        this->MTE3ToMTE2Sync();
        CopyToOutBigShape();
    }

private:
    template <bool HASLR>
    __aicore__ inline void CopyToOutSmallShape()
    {
        CopyParams copyParamsIn;
        CopyParams copyParamsOut;
        if constexpr (HASLR) {
            uint16_t blockCount = static_cast<uint16_t>(this->inOutputH_);
            uint32_t blockLen = static_cast<uint32_t>(this->outputW_ * this->TSize_);
            uint32_t srcStride = static_cast<uint32_t>(
                (this->inOutputWAlign_ + this->leftAlign_ + this->rightAlign_ - this->outputW_) * this->TSize_);
            for (uint32_t i = 0; i < this->perCoreTaskNum_; i++) {
                copyParamsIn.dcParams = {blockCount, blockLen, srcStride, 0, 0};
                copyParamsOut.dcParams = {blockCount, blockLen, 0, 0, 0};
                copyParamsIn.offset = i * this->workspaceLen_ + this->leftAlign_ - this->pLeft_;
                copyParamsOut.offset = i * this->outputLen_ + this->pTop_ * this->outputW_;
                this->CopyToOutSmallShapeOnePage(this->workspaceGM_, i, copyParamsIn, copyParamsOut);
            }
        } else {
            uint16_t blockCount = static_cast<uint16_t>(this->inOutputH_);
            uint32_t blockLen = static_cast<uint32_t>(this->outputW_ * this->TSize_);
            uint32_t srcStride = static_cast<uint32_t>((this->inputW_ - this->outputW_) * this->TSize_);
            for (uint32_t i = 0; i < this->perCoreTaskNum_; i++) {
                copyParamsIn.dcParams = {blockCount, blockLen, srcStride, 0, 0};
                copyParamsOut.dcParams = {blockCount, blockLen, 0, 0, 0};
                copyParamsIn.offset = i * this->inputLen_ - this->nTop_ * this->inputW_ - this->nLeft_;
                copyParamsOut.offset = i * this->outputLen_ + this->pTop_ * this->outputW_;
                this->CopyToOutSmallShapeOnePage(this->xGM_, i, copyParamsIn, copyParamsOut);
            }
        }
    }

    __aicore__ inline void CopyToOutBigShape()
    {
        for (uint32_t i = 0; i < this->perCoreTaskNum_; i++) {
            this->CopyToOutBigShapeOnePage(i, i);
        }
    }
};

template <typename T>
class CircularPad3D : public CircularPad<T> {
public:
    __aicore__ inline CircularPad3D(TPipe* pipe) : CircularPad<T>(pipe){};

    __aicore__ inline void Init3D(
        GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace,
        const __gm__ CircularPadTilingData* tiling_data)
    {
        this->Init(tiling_data);

        front_ = tiling_data->front;
        back_ = tiling_data->back;
        inputL_ = tiling_data->inputL;
        outputL_ = tiling_data->outputL;

        pFront_ = this->GetPositive(front_);
        pBack_ = this->GetPositive(back_);
        nFront_ = this->GetNegtive(front_);
        nBack_ = this->GetNegtive(back_);
        inOutputL_ = inputL_ + nFront_ + nBack_;

        int32_t blockId = static_cast<uint32_t>(GetBlockIdx());
        int32_t startIdx = this->perCoreTaskNum_ * blockId;
        if (blockId < (this->tailTaskNum_ / inputL_)) {
            this->perCoreTaskNum_ += inputL_;
            startIdx += blockId * inputL_;
        } else {
            startIdx += this->tailTaskNum_;
        }

        this->xGM_.SetGlobalBuffer((__gm__ T*)x + this->inputLen_ * startIdx, this->inputLen_ * this->perCoreTaskNum_);
        this->yGM_.SetGlobalBuffer(
            (__gm__ T*)y + this->outputLen_ * (startIdx * outputL_ / inputL_),
            this->outputLen_ * (this->perCoreTaskNum_ * outputL_ / inputL_));
        this->workspaceGM_.SetGlobalBuffer(
            (__gm__ T*)workspace + this->workspaceLen_ * startIdx, this->workspaceLen_ * this->perCoreTaskNum_);
    }

    __aicore__ inline void ProcessSmallShape()
    {
        if (this->hasLR) {
            this->template PadLeftAndRightSmallShape<true>();
            this->MTE3ToMTE2Sync();
            CopyToOutSmallShape<true>();
        } else {
            this->template PadLeftAndRightSmallShape<false>();
            this->MTE3ToMTE2Sync();
            CopyToOutSmallShape<false>();
        }
        this->MTE3ToMTE2Sync();
        PadFrontAndBack();
    }

    __aicore__ inline void ProcessBigShape()
    {
        if (this->hasLR) {
            this->template PadLeftAndRightBigShape<true>();
        } else {
            this->template PadLeftAndRightBigShape<false>();
        }
        this->MTE3ToMTE2Sync();
        CopyToOutBigShape();
        this->MTE3ToMTE2Sync();
        PadFrontAndBack();
    }

private:
    template <bool HASLR>
    __aicore__ inline void CopyToOutSmallShape()
    {
        CopyParams copyParamsIn;
        CopyParams copyParamsOut;
        if constexpr (HASLR) {
            uint16_t blockCount = static_cast<uint16_t>(this->inOutputH_);
            uint32_t blockLen = static_cast<uint32_t>(this->outputW_ * this->TSize_);
            uint32_t srcStride = static_cast<uint32_t>(
                (this->inOutputWAlign_ + this->leftAlign_ + this->rightAlign_ - this->outputW_) * this->TSize_);

            for (uint32_t batchIdx = 0; batchIdx < this->perCoreTaskNum_ / inputL_; batchIdx++) {
                for (int32_t pageIdx = 0; pageIdx < inOutputL_; pageIdx++) {
                    copyParamsIn.dcParams = {blockCount, blockLen, srcStride, 0, 0};
                    copyParamsOut.dcParams = {blockCount, blockLen, 0, 0, 0};
                    int64_t pageIdxIn = batchIdx * inputL_ + pageIdx - nFront_;
                    int64_t pageIdxOut = batchIdx * outputL_ + pageIdx + pFront_;
                    copyParamsIn.offset = pageIdxIn * this->workspaceLen_ + this->leftAlign_ - this->pLeft_;
                    copyParamsOut.offset = pageIdxOut * this->outputLen_ + this->pTop_ * this->outputW_;
                    this->CopyToOutSmallShapeOnePage(this->workspaceGM_, pageIdxOut, copyParamsIn, copyParamsOut);
                }
            }
        } else {
            uint16_t blockCount = static_cast<uint16_t>(this->inOutputH_);
            uint32_t blockLen = static_cast<uint32_t>(this->outputW_ * this->TSize_);
            uint32_t srcStride = static_cast<uint32_t>((this->inputW_ - this->outputW_) * this->TSize_);

            for (uint32_t batchIdx = 0; batchIdx < this->perCoreTaskNum_ / inputL_; batchIdx++) {
                for (int32_t pageIdx = 0; pageIdx < inOutputL_; pageIdx++) {
                    copyParamsIn.dcParams = {blockCount, blockLen, srcStride, 0, 0};
                    copyParamsOut.dcParams = {blockCount, blockLen, 0, 0, 0};
                    int64_t pageIdxIn = batchIdx * inputL_ + pageIdx - nFront_;
                    int64_t pageIdxOut = batchIdx * outputL_ + pageIdx + pFront_;
                    copyParamsIn.offset = pageIdxIn * this->inputLen_ - this->nTop_ * this->inputW_ - this->nLeft_;
                    copyParamsOut.offset = pageIdxOut * this->outputLen_ + this->pTop_ * this->outputW_;
                    this->CopyToOutSmallShapeOnePage(this->xGM_, pageIdxOut, copyParamsIn, copyParamsOut);
                }
            }
        }
    }

    __aicore__ inline void CopyToOutBigShape()
    {
        for (uint32_t batchIdx = 0; batchIdx < this->perCoreTaskNum_ / inputL_; batchIdx++) {
            for (int32_t pageIdx = 0; pageIdx < inOutputL_; pageIdx++) {
                int64_t pageIdxIn = batchIdx * inputL_ + pageIdx - nFront_;
                int64_t pageIdxOut = batchIdx * outputL_ + pageIdx + pFront_;
                this->CopyToOutBigShapeOnePage(pageIdxIn, pageIdxOut);
            }
        }
    }

    __aicore__ inline void PadFrontAndBack()
    {
        int64_t stride = this->outputLen_ * outputL_;
        if (front_ > 0) {
            int64_t offsetIn = (outputL_ - pBack_ - front_) * this->outputLen_;
            int64_t offsetOut = 0;
            this->CopyGmToGm(front_, this->perCoreTaskNum_ / inputL_, offsetIn, offsetOut, stride);
        }
        if (back_ > 0) {
            int64_t offsetIn = pFront_ * this->outputLen_;
            int64_t offsetOut = (outputL_ - pBack_) * this->outputLen_;
            this->CopyGmToGm(back_, this->perCoreTaskNum_ / inputL_, offsetIn, offsetOut, stride);
        }
    }

private:
    int64_t inputL_{0};
    int64_t inOutputL_{0};
    int64_t outputL_{0};
    int64_t front_{0};
    int64_t back_{0};
    int64_t pFront_{0};
    int64_t pBack_{0};
    int64_t nFront_{0};
    int64_t nBack_{0};
};

#endif
