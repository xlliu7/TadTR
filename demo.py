# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------


from models import build_model
from opts import update_cfg_from_file
from util.misc import NestedTensor
import torch
import time
import pdb


# @torch.no_grad()
def demo(args, cfg):
    device = torch.device(args.device)
    model, _, _ = build_model(cfg)

    bs, t = 1, 100
    x = torch.rand([bs, cfg.feature_dim, t]).to(device)
    mask = torch.ones([bs, t], dtype=torch.bool).to(device)
    samples = NestedTensor(x, mask)
    targets = [
        {
            'labels': torch.LongTensor([0, 0]).to(device),
            'segments': torch.FloatTensor([[0.5, 0.2], [0.7, 0.3]]).to(device),
            'orig_size': 100.0
        } for i in range(bs)]

    model.to(device)

    outputs = model(samples)
    
    # orig_target_sizes = torch.FloatTensor(
    #         [t["orig_size"] for t in targets]).cuda()
    # results = postprocessor(outputs, orig_target_sizes)
    print('Passed')


if __name__ == '__main__':
    from opts import get_args_parser, cfg, update_cfg_with_args
    args = get_args_parser().parse_args()

    if args.cfg:
        update_cfg_from_file(cfg, args.cfg)
    update_cfg_with_args(cfg, args.opt)

    if cfg.disable_cuda:
        cfg.act_reg = False
    demo(args, cfg)
