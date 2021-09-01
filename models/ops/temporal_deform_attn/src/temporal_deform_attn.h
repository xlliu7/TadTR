/*!
**************************************************************************
* TadTR: End-to-end Temporal Action Detection with Transformer
* Copyright (c) 2021. Xiaolong Liu.
**************************************************************************
* Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

#pragma once

#include "cpu/temporal_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/temporal_deform_attn_cuda.h"
#endif


at::Tensor
temporal_deform_attn_forward(
    const at::Tensor &value, 
    const at::Tensor &temporal_lens,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int seq2col_step)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return temporal_deform_attn_cuda_forward(
            value, temporal_lens, level_start_index, sampling_loc, attn_weight, seq2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
temporal_deform_attn_backward(
    const at::Tensor &value, 
    const at::Tensor &temporal_lens,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int seq2col_step)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return temporal_deform_attn_cuda_backward(
            value, temporal_lens, level_start_index, sampling_loc, attn_weight, grad_output, seq2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

