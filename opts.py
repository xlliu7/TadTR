# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------


import argparse


def str2bool(x):
    if x.lower() in ['true', 't', '1', 'y']:
        return True
    else:
        return False


def get_args_parser():
    parser = argparse.ArgumentParser('TadTR', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    # Valid only when the input is video frames
    parser.add_argument('--lr_backbone_names',
                        default=["backbone"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr_step', nargs='+', default=[25], type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='none', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'none'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=30, type=int,
                        help="Number of action queries")
    parser.add_argument('--activation', type=str, default='relu',
                        help="transformer activation type, relu|leaky_relu|gelu")
    
    parser.add_argument('--num_feature_levels', default=1,
                        type=int, help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # dataset parameters
    parser.add_argument('--dataset_name', default='hacs')
    parser.add_argument(
        '--input_type', choices=['image', 'feature'], default='feature')
    parser.add_argument('--feature', default='i3d',
                        help='which kind of feature to use. e.g. i3d, tsn.')
    parser.add_argument('--feature_dim', default=2048, type=int,
                        help='dimension (channels) of the video feature')
    parser.add_argument('--binary', type=str2bool,
                        default=False, help='do binary classification only')
    parser.add_argument('--slice_len', type=int, default=128,
                        help='length of video slices for batch training')
    parser.add_argument('--online_slice', type=str2bool, default=False,
                        help='whether to crop videos during training')
    parser.add_argument('--test_set', type=str, default='val')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # training parameters
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--optimizer', type=str, choices=[
                        'AdamW', 'Adam', 'SGD'], default='AdamW', help='which optimizer to use')
    parser.add_argument('--ckpt_interval', type=int, default=100,
                        help='save checkpoint every N epochs. N=100 (default)')
    parser.add_argument('--iter_size', type=int, default=1,
                        help='update parameters every N forward-backward passes. N=1 (default)')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='test model every N epochs. N=1 (default)')

    parser.add_argument('opt', nargs=argparse.REMAINDER,
                        help='Command arguments that override configs')
    return parser
