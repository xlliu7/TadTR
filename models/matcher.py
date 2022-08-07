# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.segment_ops import segment_cw_to_t1t2, segment_iou
import pdb


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_seg: float = 1, cost_iou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_seg: This is the relative weight of the L1 error of the segment coordinates in the matching cost
            cost_iou: This is the relative weight of the iou loss of the segment in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_seg = cost_seg
        self.cost_iou = cost_iou
        assert cost_class != 0 or cost_seg!= 0 or cost_iou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_segments": Tensor of dim [batch_size, num_queries, 2] with the predicted segment coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_segments] (where num_target_segments is the number of ground-truth
                           objects in the target) containing the class labels
                 "segments": Tensor of dim [num_target_segments, 2] containing the target segment coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_segments)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  #  [batch_size * num_queries, num_classes]
        out_seg = outputs["pred_segments"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and segments
        tgt_ids = torch.cat([v["labels"] for v in targets])  # shape = n1+n2+...
        tgt_seg = torch.cat([v["segments"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        # Compute the L1 cost between segments
        cost_seg = torch.cdist(out_seg, tgt_seg, p=1)

        # Compute the iou cost betwen segments
        cost_iou = -segment_iou(segment_cw_to_t1t2(out_seg), segment_cw_to_t1t2(tgt_seg))

        # Final cost matrix, [bs x nq, batch_ngt]
        C = self.cost_seg * cost_seg + self.cost_class * cost_class + self.cost_iou * cost_iou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["segments"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_seg=args.set_cost_seg, cost_iou=args.set_cost_iou)