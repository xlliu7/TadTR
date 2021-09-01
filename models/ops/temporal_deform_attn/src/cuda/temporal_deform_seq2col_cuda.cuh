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

#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}


template <typename scalar_t>
__device__ scalar_t ms_deform_attn_seq2col_linear(const scalar_t* &bottom_data, 
                                                  const int &width, const int &nheads, const int &channels,
                                                  const scalar_t &w, const int &m, const int &c)
{
  const int w_low = floor(w);
  const int w_high = w_low + 1;

  const scalar_t lw = w - w_low;
  const scalar_t hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (w_low >= 0)
  {
    const int ptr1 =  w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (w_high <= width - 1)
  {
    const int ptr2 = w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  
  const scalar_t w1 = hw, w2 = lw;

  const scalar_t val = (w1 * v1 + w2 * v2);
  return val;
}


template <typename scalar_t>
__global__ void ms_deformable_seq2col_gpu_kernel(const int n,
                                                const scalar_t *data_value, 
                                                const int64_t *data_temporal_lens,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int temporal_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * temporal_size * qid_stride;
    scalar_t col = 0;
    
    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int temporal_w = data_temporal_lens[l_col];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t w_im = loc_w * temporal_w - 0.5;

        if (w_im > -1 && w_im < temporal_w)
        {
          col += ms_deform_attn_seq2col_linear(data_value_ptr, temporal_w, num_heads, channels, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 1;
      }
    }
    *data_col_ptr = col;
  }
}






template <typename scalar_t>
void ms_deformable_seq2col_cuda(cudaStream_t stream,
                              const scalar_t* data_value,
                              const int64_t* data_temporal_lens, 
                              const int64_t* data_level_start_index, 
                              const scalar_t* data_sampling_loc,
                              const scalar_t* data_attn_weight,
                              const int batch_size,
                              const int temporal_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              scalar_t* data_col)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  ms_deformable_seq2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
          0, stream>>>(
      num_kernels, data_value, data_temporal_lens, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, temporal_size, num_heads, channels, num_levels, num_query, num_point, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_seq2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void ms_deformable_col2seq_cuda(cudaStream_t stream,
                              const scalar_t* grad_col,
                              const scalar_t* data_value,
                              const int64_t * data_temporal_lens,
                              const int64_t * data_level_start_index,
                              const scalar_t * data_sampling_loc,
                              const scalar_t * data_attn_weight,
                              const int batch_size, 
                              const int temporal_size, 
                              const int num_heads,
                              const int channels, 
                              const int num_levels,
                              const int num_query,
                              const int num_point, 
                              scalar_t* grad_value,
                              scalar_t* grad_sampling_loc,
                              scalar_t* grad_attn_weight)
{
  const int num_threads = (channels > CUDA_NUM_THREADS)?CUDA_NUM_THREADS:channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  // channels is 32 by default
  // aware reducev1/2 shm_reduce1/2
  
  AT_ERROR("Not implemented");

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_col2seq_cuda: %s\n", cudaGetErrorString(err));
  }

}