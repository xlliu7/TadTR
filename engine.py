# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os.path as osp
import sys
from typing import Iterable
import tqdm
import logging

import torch

import util.misc as utils
from datasets.tad_eval import TADEvaluator
import pickle

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cfg, max_norm: float = 0):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['segments', 'labels']
                    else v for k, v in t.items()} for t in targets]

        outputs = model((samples.tensors, samples.mask))
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss of each type
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # weighted_loss of each type
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logging.info("Loss is {}, stopping training".format(loss_value))
            logging.info(str(loss_dict_reduced))
            sys.exit(1)

        losses.backward()
        if (cnt + 1) % cfg.iter_size == 0:
            # scale gradients when iter size is functioning
            if cfg.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= cfg.iter_size

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        cnt += 1

    optimizer.zero_grad()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def to_device(t, device):
    if isinstance(t, (list, tuple)):
        return t
    else:
        return t.to(device)


@torch.no_grad()
def test(model, criterion, postprocessor, data_loader, base_ds, device, output_dir, cfg, subset='val', epoch=None, test_mode=False):
    '''
    Run inference and evaluation. Do not compute loss
    test_mode: indicates that we are evaluating specific epoch during testing
    '''
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))

    iou_range = [0.3, 0.4, 0.5, 0.6, 0.7] if cfg.dataset_name == 'thumos14' else [
        num/100 for num in range(50, 100, 5)]
    # logging.info('iou range {}'.format(iou_range))

    # action_evaluator = None
    action_evaluator = TADEvaluator(cfg.dataset_name, subset, base_ds, nms_mode=[
                                          'raw'], iou_range=iou_range, epoch=epoch)

    # raw_res = []
    cnt = 0
    for (samples, targets) in tqdm.tqdm(data_loader, total=len(data_loader)):
        samples = samples.to(device)
        outputs = model((samples.tensors, samples.mask))

        # raw_res.append((outputs, targets))
        video_duration = torch.FloatTensor(
            [t["video_duration"] for t in targets]).to(device)
        results = postprocessor(outputs, video_duration, fuse_score=cfg.act_reg)

        res = {target['video_id']: output for target,
               output in zip(targets, results)}
        if action_evaluator is not None:
            action_evaluator.update(res, assign_cls_labels=cfg.binary)
        # if cnt >= 9:
        #     break
        cnt += 1

    # accumulate predictions from all videos
    if action_evaluator is not None:
        action_evaluator.synchronize_between_processes()
        action_evaluator.accumulate(cfg.test_slice_overlap)
        # dump detections
        if test_mode:
            save_path = osp.join('outputs', 'detection_{}.json')
            action_evaluator.dump_detection(save_path)
        action_evaluator.summarize()

    stats = {}

    if action_evaluator is not None:
        for k, v in action_evaluator.stats.items():
            for vk, vv in v.items():
                stats[vk + '_' + k] = vv

        mAP_values = ' '.join([f'{k}: {100*v:.2f}'.format(k, v)
                              for k, v in stats.items() if k.startswith('mAP')])
        logging.info(mAP_values)

        stats['stats_summary'] = action_evaluator.stats_summary

    # with open('raw_outputs.pkl', 'wb') as f:
    #     pickle.dump(raw_res, f)

    return stats
