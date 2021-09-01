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
#include <torch/extension.h>

at::Tensor
temporal_deform_attn_cpu_forward(
    const at::Tensor &value, 
    const at::Tensor &temporal_lens,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step);

std::vector<at::Tensor>
temporal_deform_attn_cpu_backward(
    const at::Tensor &value, 
    const at::Tensor &temporal_lens,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step);


