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

#include "temporal_deform_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("temporal_deform_attn_forward", &temporal_deform_attn_forward, "temporal_deform_attn_forward");
  m.def("temporal_deform_attn_backward", &temporal_deform_attn_backward, "temporal_deform_attn_backward");
}
