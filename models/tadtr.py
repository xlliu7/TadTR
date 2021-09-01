# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
TadTR model and criterion classes.
"""
from models.position_encoding import build_position_encoding
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import segment_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)

# from .matcher import build_matcher
from .custom_loss import sigmoid_focal_loss
from .transformer import build_deformable_transformer
import copy
from util.config import cfg
from models.ops.roi_align import ROIAlign
import pdb


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == 'gn':
        assert num_groups is not None, 'num_groups must be specified'
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class TadTR(nn.Module):
    """ This is the TadTR module that performs temporal action detection """

    def __init__(self, position_embedding, transformer, num_classes, num_queries, aux_loss=True, with_segment_refine=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See deformable_transformer.py
            num_classes: number of action classes
            num_queries: number of action queries, ie detection slot. This is the maximal number of actions
                         TadTR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_segment_refine: iterative segment refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2048, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])
        # self.backbone = backbone
        self.position_embedding = position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_segment_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.segment_embed = _get_clones(self.segment_embed, num_pred)
            nn.init.constant_(
                self.segment_embed[0].layers[-1].bias.data[1:], -2.0)
            # hack implementation for segment refinement
            self.transformer.decoder.segment_embed = self.segment_embed
        else:
            nn.init.constant_(
                self.segment_embed.layers[-1].bias.data[1:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)])
            self.segment_embed = nn.ModuleList(
                [self.segment_embed for _ in range(num_pred)])
            self.transformer.decoder.segment_embed = None

        if cfg.ACTIONNESS_REG:
            # RoIAlign params
            self.roi_size = 16
            self.roi_scale = 0
            self.roi_extractor = ROIAlign(self.roi_size, self.roi_scale)
            self.actionness_pred = nn.Sequential(
                nn.Linear(self.roi_size * hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def _to_roi_align_format(self, rois, T, scale_factor=1):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1]
        rois_size = rois[:, :, 1:2] * scale_factor
        rois_abs = torch.cat(
            (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * T
        # expand the RoIs
        rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
        batch_ind = batch_ind.repeat(1, N, 1)
        rois_abs = torch.cat((batch_ind, rois_abs), dim=2)
        # NOTE: stop gradient here to stablize training
        return rois_abs.view((B*N, 3)).detach()

    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            or a tuple of tensors and mask

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (n, c, t)

        pos = [self.position_embedding(samples)]
        src, mask = samples.tensors, samples.mask
        srcs = [self.input_proj[0](src)]
        masks = [mask]

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, memory = self.transformer(
            srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        # gather outputs from each decoder layer
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.segment_embed[lvl](hs[lvl])
            # the l-th layer (l >= 2)
            if reference.shape[-1] == 2:
                tmp += reference
            # the first layer
            else:
                assert reference.shape[-1] == 1
                tmp[..., 0] += reference[..., 0]
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if not cfg.ACTIONNESS_REG:
            out = {'pred_logits': outputs_class[-1],
                   'pred_segments': outputs_coord[-1]}
        else:
            # perform RoIAlign
            B, N = outputs_coord[-1].shape[:2]
            origin_feat = memory

            rois = self._to_roi_align_format(
                outputs_coord[-1], origin_feat.shape[2], scale_factor=1.5)
            roi_features = self.roi_extractor(origin_feat, rois)
            roi_features = roi_features.view((B, N, -1))
            pred_actionness = self.actionness_pred(roi_features)

            last_layer_cls = outputs_class[-1]
            last_layer_reg = outputs_coord[-1]

            out = {'pred_logits': last_layer_cls,
                   'pred_segments': last_layer_reg, 'pred_actionness': pred_actionness}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for TadTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_name == 'thumos14':
            num_classes = 20
        elif args.dataset_name == 'muses':
            num_classes = 25
        else:
            num_classes = 200

    device = torch.device(args.device)

    pos_embed = build_position_encoding(args)
    transformer = build_deformable_transformer(args)

    model = TadTR(
        pos_embed,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        with_segment_refine=cfg.SEGMENT_REFINE,
    )

    # postprocessor = PostProcess()

    return model
