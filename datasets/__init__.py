from .tad_dataset import build as build_video_dataset


def build_dataset(subset, args, mode):
    if args.dataset_name in ['activitynet', 'thumos14', 'hacs', 'muses']:
        return build_video_dataset(args.dataset_name, subset, args, mode)
    
    raise ValueError(f'dataset {args.dataset_name} not supported')