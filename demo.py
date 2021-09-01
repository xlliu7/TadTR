# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------


from models import build_model
from util.misc import NestedTensor
import torch
import time
import pdb


# @torch.no_grad()
def demo(args):

    model = build_model(args)

    bs, t = 1, 100
    x = torch.rand([bs, args.feature_dim, t]).cuda()
    mask = torch.ones([bs, t], dtype=torch.bool).cuda()
    samples = NestedTensor(x, mask)
    targets = [
        {
            'labels': torch.LongTensor([0, 0]).cuda(),
            'segments': torch.FloatTensor([[0.5, 0.2], [0.7, 0.3]]).cuda(),
            'orig_size': 100.0
        } for i in range(bs)]

    model.cuda()

    outputs = model(samples)
    
    # orig_target_sizes = torch.FloatTensor(
    #         [t["orig_size"] for t in targets]).cuda()
    # results = postprocessor(outputs, orig_target_sizes)
    print('Passed')


if __name__ == '__main__':
    from opts import get_args_parser
    
    from util.config import cfg, update_cfg_with_args
    args = get_args_parser().parse_args()
    update_cfg_with_args(cfg, args.opt)
    demo(args)
