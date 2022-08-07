# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021 - 2022. Xiaolong Liu.
# ------------------------------------------------------------------------


import argparse
from easydict import EasyDict
import yaml



def str2bool(x):
    if x.lower() in ['true', 't', '1', 'y']:
        return True
    else:
        return False


def get_args_parser():
    parser = argparse.ArgumentParser('TadTR', add_help=False)

    parser.add_argument('--cfg', type=str, help='the config file to use')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    parser.add_argument('--eval', action='store_true', help='perform testing')
    parser.add_argument('--num_workers', default=2, type=int, help='number of dataloader workers')

    # Multi-GPU training
    # We support both DataParallel and Distributed DataParallel (DDP)
    parser.add_argument('--multi_gpu', action='store_true', help='use nn.DataParallel')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Other options
    parser.add_argument('opt', nargs=argparse.REMAINDER,
                        help='Command arguments that override configs')
    return parser



cfg = EasyDict()
# ---- Basic option ----
# whether to enable tensorboard
cfg.tensorboard = False
# Disable CUDA extensions so that we can run the model on CPU
cfg.disable_cuda = False
# The backend of deformable attention, pytorch or CUDA
cfg.dfm_att_backend = 'pytorch'

# path where to save, empty for no saving
cfg.output_dir = ''


# # ------ Data options ------
cfg.dataset_name = 'thumos14'
# Use feature input or raw image input (jointly train the video encoder and the detection head). Choices: {feature, image}
cfg.input_type = 'feature'   
# Which kind of feature to use. e.g. i3d, tsn.
cfg.feature = 'i3d2s'
# dimension (channels) of the video feature
cfg.feature_dim = 2048
# Perform binary detection (proposal generation) only 
cfg.binary = False
# Testing on Which subset 'val' or 'test' (For Anet and HACS). Note that we rename the training/validation/testing subsets for all datasets. For example, the validation subset used for training on THUMOS14 is renamed as 'train' subset.
cfg.test_set = 'val'
# whether to crop video into windows (A window is also called a slice in this codebase). Required for THUMOS14
cfg.online_slice = False
# length of video slices. For feature input, the length is for feature sequence. For video input, the length is for frame sequence.
cfg.slice_len = None
# overlap ratio (=overlap_length/slice_length) between adjacent slices during training
cfg.slice_overlap = 0
# overlap ratio between adjacent slices during inference 
cfg.test_slice_overlap = 0


# ---- Model option --------
# Name of the convolutional backbone to use. If we use video features as input, backbone should be 'none' 
cfg.backbone = 'none'

# whether to use position embedding
cfg.use_pos_embed = True
# Type of positional embedding to use on top of the video features. Only support sine embedding.
cfg.position_embedding = "sine"

# Number of encoding layers in the transformer
cfg.enc_layers = 2
# Number of decoding layers in the transformer
cfg.dec_layers = 4
# Intermediate size of the feedforward layers in the transformer blocks
cfg.dim_feedforward = 2048
# Size of the embeddings (dimension of the transformer)
cfg.hidden_dim = 256
# Dropout applied in the transformer
cfg.dropout = 0.1
# Number of attention heads inside the transformer's attentions
cfg.nheads = 8
# Number of sampled points per head for deformable attention in the encoder
cfg.enc_n_points = 4
# Number of sampled points per head for deformable attention in the decoder
cfg.dec_n_points = 4
# Number of action queries
cfg.num_queries = 30
# Transformer activation type, relu|leaky_relu|gelu
cfg.activation = 'relu'
# Whether to enable segment refinement mechanism
cfg.seg_refine = True
# Whether to enable actionness regression head
cfg.act_reg = True
# whether to disable self-attention between action queries
cfg.disable_query_self_att = False


# ----- Loss and matcher setting -------
# Enable auxiliary decoding losses (loss at each layer)
cfg.aux_loss = True

# Loss weight 
cfg.act_loss_coef = 4
cfg.cls_loss_coef = 2
cfg.seg_loss_coef = 5
cfg.iou_loss_coef = 2
# Relative classification weight of the no-action class
cfg.eos_coef = 0.1
# For focal loss
cfg.focal_alpha = 0.25

# Set cost weight
cfg.set_cost_class = 6    # Class coefficient 
cfg.set_cost_seg = 5      # Segment L1 coefficient 
cfg.set_cost_iou = 2      # Segment IoU coefficient


# ----- Training option -------
# base learning rate. If you set lr in yaml file, don't use this format, use 0.0002 instead
cfg.lr = 2e-4

# Valid only when the input is video frames
# specify the name pattern of the backbone layers.
cfg.lr_backbone_names = ['backbone']
# learning rate of backbone layers
cfg.lr_backbone = 1e-5

# special linear projection layers that need to use smaller lr
cfg.lr_linear_proj_names = ['reference_points', 'sampling_offsets']
cfg.lr_linear_proj_mult = 0.1

# which optimizer to use, choose from ['AdamW', 'Adam', 'SGD']
cfg.optimizer = 'AdamW'
cfg.batch_size = 16
cfg.weight_decay = 1e-4
# gradient clipping max norm
cfg.clip_max_norm = 0.1

# maximum number of training epochs
cfg.epochs = 16

# when to decay lr
cfg.lr_step = [14]
# save checkpoint every N epochs. Set it to a small value if you want to save intermediate models
cfg.ckpt_interval = 10
# update parameters every N forward-backward passes. N=1 (default)
cfg.iter_size = 1
# test model every N epochs. N=1 (default)
cfg.test_interval = 1


# ----- Postproc option -------
# How to rank the predicted instances. 
# 1: for each query, generate a instance for each class; then pick top-scored instance from the whole set
# 2: pick top classes for each query
cfg.postproc_rank = 1
# for each query, pick top k classes; keep all queries
# this setting is useful for debug
cfg.postproc_cls_topk = 1
# for each video, pick topk detections
cfg.postproc_ins_topk = 100
# IoU threshold for NMS. Note that NMS is not necessary.
cfg.nms_thr = 0.4



def update_cfg_with_args(cfg, arg_list):
    from ast import literal_eval
    for i in range(0, len(arg_list), 2):
        cur_entry = cfg
        key_parts = arg_list[i].split('.')
        for k in key_parts[:-1]:
            cur_entry = cur_entry[k]
        node = key_parts[-1]
        try:
            cur_entry[node] = literal_eval(arg_list[i+1])
        except:
            # print(f'literal_eval({arg_list[i+1]}) failed, directly take the value')
            cur_entry[node] = arg_list[i+1]


def update_cfg_from_file(cfg, cfg_path):
    import os
    assert os.path.exists(cfg_path), 'cfg_path is invalid'
    cfg_from_file = yaml.load(open(cfg_path), yaml.FullLoader)
    cfg.update(cfg_from_file)