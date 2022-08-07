# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import os
import glob
import pdb

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def get_sources(extensions_dir):
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    return main_file + source_cpu + source_cuda


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
   
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError('Cuda is not availabel')


    ext_modules = [
        # Temporal Deformable Attention, optional
        # CUDAExtension(
        #     "temporal_deform_attn.TemporalDeformableAttention",
        #     get_sources(os.path.join(this_dir, "temporal_deform_attn/src")),
        #     include_dirs=[os.path.join(this_dir, "temporal_deform_attn/src")],
        #     define_macros=define_macros,
        #     extra_compile_args=extra_compile_args
        # ),

        CUDAExtension('roi_align.Align1D', [
            'roi_align/src/roi_align_cuda.cpp',
            'roi_align/src/roi_align_kernel.cu'])
    ]
    return ext_modules

setup(
    name="TadTR_release",
    version="1.0",
    author="Xiaolong Liu",
    description="PyTorch Wrapper for CUDA Functions of TadTR",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
