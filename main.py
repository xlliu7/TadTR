# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021 - 2012. Xiaolong Liu
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
'''Entry for training and testing'''

import datetime
import json
import random
import time
from pathlib import Path
import re
import os
import logging
import sys
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from opts import get_args_parser, cfg, update_cfg_with_args, update_cfg_from_file
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, test
from models import build_model
if cfg.tensorboard:
    from torch.utils.tensorboard import SummaryWriter

        

def main(args):
    from util.logger import setup_logger

    if args.cfg is not None:
        update_cfg_from_file(cfg, args.cfg)

    update_cfg_with_args(cfg, args.opt)

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # The actionness regression module requires CUDA support
    # If your machine does not have CUDA enabled, this module will be disabled.
    if cfg.disable_cuda:
        cfg.act_reg = False

    utils.init_distributed_mode(args)

    if not args.eval:
        mode = 'train'
    else:
        mode = 'test'

    # Logs will be saved in log_path
    log_path = os.path.join(cfg.output_dir, mode + '.log')
    setup_logger(log_path)

    logging.info("git:\n  {}\n".format(utils.get_sha()))

    logging.info(' '.join(sys.argv))

    with open(osp.join(cfg.output_dir, mode + '_cmd.txt'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    logging.info(str(args))
    logging.info(str(cfg))

    device = torch.device(args.device)

    # fix the seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cfg.input_type == 'image':
        # We plan to support image input in the future
        raise NotImplementedError

    model, criterion, postprocessors = build_model(cfg)

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    elif args.multi_gpu:
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters())
    logging.info('number of params: {}'.format(n_parameters))

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        # non-backbone, non-offset
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, cfg.lr_backbone_names) and not match_name_keywords(n, cfg.lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.lr,
            "initial_lr": cfg.lr
        },
        # backbone
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.lr_backbone_names) and p.requires_grad],
            "lr": cfg.lr_backbone,
            "initial_lr": cfg.lr_backbone
        },
        # offset
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.lr * cfg.lr_linear_proj_mult,
            "initial_lr": cfg.lr * cfg.lr_linear_proj_mult
        }
    ]

    optimizer = torch.optim.__dict__[cfg.optimizer](param_dicts, lr=cfg.lr,
                                                     weight_decay=cfg.weight_decay)

    output_dir = Path(cfg.output_dir)

    if args.resume == 'latest':
        args.resume = osp.join(cfg.output_dir, 'checkpoint.pth')
    elif args.resume == 'best':
        args.resume = osp.join(cfg.output_dir, 'model_best.pth')

    if 'model_best.pth' in os.listdir(cfg.output_dir) and not args.resume and not args.eval:
        # for many times, my trained models were accidentally overwrittern by new models😂. So I add this to avoid that
        logging.error(
            'Danger! You are overwriting an existing output dir {}, probably because you forget to change the output_dir option'.format(cfg.output_dir))
        confirm = input('confirm: y/n')
        if confirm != 'y':
            return

    last_epoch = -1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        last_epoch = checkpoint['epoch']

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.lr_step, last_epoch=last_epoch)

    dataset_val = build_dataset(subset=cfg.test_set, args=cfg, mode='val')
    if not args.eval:
        dataset_train = build_dataset(subset='train', args=cfg, mode='train')

    if args.distributed:
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

    else:
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    data_loader_val = DataLoader(dataset_val, cfg.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    base_ds = dataset_val.video_dict

    if not args.eval and cfg.tensorboard and utils.is_main_process():
        smry_writer = SummaryWriter(output_dir)
    else:
        smry_writer = None

    best_metric = -1
    best_metric_txt = ''

    if args.eval and not args.resume:
        args.resume = osp.join(output_dir, 'model_best.pth')

    # start training from this epoch. You do not to set this option.
    start_epoch = 0
    if args.resume:
        print('loading checkpint {}'.format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

        if 'best_metric' in checkpoint:
            best_metric = checkpoint['best_metric']

    if args.eval:
        test_stats = test(model, criterion, postprocessors,
                          data_loader_val, base_ds, device, cfg.output_dir, cfg, subset=cfg.test_set, epoch=checkpoint['epoch'], test_mode=True)

        return

    logging.info("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        for group in optimizer.param_groups:
            logging.info('lr={}'.format(group['lr']))
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, cfg,
            cfg.clip_max_norm)

        lr_scheduler.step()

        if cfg.output_dir:
            # save checkpoint every `cfg.ckpt_interval` epochs, also when reducing the learning rate
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) in cfg.lr_step or (epoch + 1) % cfg.ckpt_interval == 0:
                checkpoint_paths.append(
                    output_dir / f'checkpoint{epoch:04}.pth')
            ckpt = {
                'model': model_without_ddp.state_dict(),
                'epoch': epoch,
                'args': args,
                'cfg': cfg,
                'best_metric': best_metric,
            }
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(ckpt, checkpoint_path)

        if (epoch + 1) % cfg.test_interval == 0:
            test_stats = test(
                model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.output_dir, cfg, epoch=epoch
            )
            prime_metric = 'mAP_raw'
            if test_stats[prime_metric] > best_metric:
                best_metric = test_stats[prime_metric]
                best_metric_txt = test_stats['stats_summary']
                logging.info(
                    'new best metric {:.4f}@epoch{}'.format(best_metric, epoch))
                if cfg.output_dir:
                    ckpt['best_metric'] = best_metric
                    best_ckpt_path = output_dir / 'model_best.pth'
                    utils.save_on_master(ckpt, best_ckpt_path)

        else:
            test_stats = {}

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if cfg.output_dir and utils.is_main_process():
            for k, v in log_stats.items():
                if isinstance(v, np.ndarray):
                    log_stats[k] = v.tolist()
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if smry_writer:
                for k, v in log_stats.items():
                    if re.findall('loss_\S+unscaled', k) or k.endswith('loss') or 'lr' in k or 'AP50' in k or 'AP75' in k or 'AP95' in k or 'mAP' in k or 'AR' in k:
                        smry_writer.add_scalar(k, v, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.is_main_process():
        logging.info('Training time {}'.format(total_time_str))
        logging.info(str(
            ['{}:{}'.format(k, v) for k, v in test_stats.items() if 'AP' in k or 'AR' in k]))
        if smry_writer is not None:
            smry_writer.close()
    logging.info('best det result\n{}'.format(best_metric_txt))
    logging.info(log_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        'TadTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    s_ = time.time()
    main(args)
    logging.info('main takes {:.3f} seconds'.format(time.time() - s_))
