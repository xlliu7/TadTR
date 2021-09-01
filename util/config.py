# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------

from easydict import EasyDict

cfg = EasyDict()

# basic option
cfg.TENSORBOARD = True

# model option
cfg.ACTIONNESS_REG = True
cfg.SEGMENT_REFINE = True
cfg.USE_POS_EMBED = True
cfg.DISABLE_QUERY_SELF_ATT = False

# postproc option
# 1: pick top detections from all queries
# 2: pick top classes for each query
cfg.POSTPROC_RANK = 1
cfg.POSTPROC_CLS_TOPK = 1    # for each query, pick topk classes; keep all queries
cfg.POSTPROC_INS_TOPK = 100  # for each video, pick topk detections


# for thumos14, take video slices for training and testing
cfg.SLICE_OVERLAP = 0
cfg.TEST_SLICE_OVERLAP = 0


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
