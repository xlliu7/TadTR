# TadTR: End-to-end Temporal Action Detection with Transformer

import json
import os.path as osp
import os
import pandas as pd
import time
import numpy as np
import logging
import concurrent.futures
import sys
import logging
# import ipdb as pdb
import pickle

from opts import cfg

from Evaluation.eval_detection import compute_average_precision_detection
# from Evaluation.eval_proposal import average_recall_vs_avg_nr_proposals
import matplotlib.pyplot as plt
# from util.proposal_utils import soft_nms
from .data_utils import get_dataset_dict
from util.misc import all_gather
from util.segment_ops import soft_nms, temporal_nms


def eval_ap(iou, cls, gt, predition):
    ap = compute_average_precision_detection(gt, predition, iou)
    sys.stdout.flush()
    return cls, ap


def apply_nms(dets_arr, nms_thr=0.4, use_soft_nms=False):
    # the last column are class ids
    unique_classes = np.unique(dets_arr[:, 3])
    output_dets = []
    for cls in unique_classes:
        this_cls_dets = dets_arr[dets_arr[:,3] == cls]
        if not use_soft_nms:
            this_cls_dets_kept = temporal_nms(this_cls_dets, nms_thr)
        else:
            classes = this_cls_dets[:, [3]]
            this_cls_dets_kept = soft_nms(this_cls_dets, 0.8, 0, 0, 100)
            this_cls_dets_kept = np.concatenate((this_cls_dets_kept, classes), -1)
        output_dets.append(this_cls_dets_kept)
    output_dets = np.concatenate(output_dets, axis=0)
    sort_idx = output_dets[:, 2].argsort()[::-1]
    output_dets = output_dets[sort_idx, :]
    return output_dets


class TADEvaluator(object):
    def __init__(self, dataset_name, subset, video_dict=None, nms_mode=['raw'], iou_range=[0.5], epoch=None, num_workers=None):
        '''dataset_name:  thumos14, activitynet or hacs
        subset: val or test
        video_dict: the dataset dict created in video_dataset.py
        iou_range: [0.3:0.7:0.1] for thumos14; [0.5:0.95:0.05] for anet and hacs.
        '''

        self.epoch = epoch
        self.iou_range = iou_range
        self.nms_mode = nms_mode
        self.dataset_name = dataset_name
        self.ignored_videos = list()

        if dataset_name == 'thumos14':
            subset_mapping = {'train': 'val', 'val': 'test'}
            anno_file = 'data/thumos14/th14_annotations_with_fps_duration.json'
            # follow SSN/PGCN/AFSD/MUSES to remove three falsely annotated videos
            self.ignored_videos = ['video_test_0000270', 'video_test_0001292', 'video_test_0001496']
        else:
            raise NotImplementedError
        anno_dict = json.load(open(anno_file))
        classes = self._get_classes(anno_dict)
        num_classes = len(classes)
        
        database = anno_dict['database']
        all_gt = []

        unique_video_list = [x for x in database if database[x]['subset'] in subset_mapping[subset]]

        for vid in unique_video_list:
            if vid in self.ignored_videos:
                continue
            this_gts = [x for x in database[vid]['annotations'] if x['label'] != 'Ambiguous']
            all_gt += [[vid, classes.index(x['label']), x['segment'][0], x['segment'][1]] for x in this_gts]

        all_gt = pd.DataFrame(all_gt, columns=["video-id", "cls","t-start", "t-end"])
        self.video_ids = all_gt['video-id'].unique().tolist()
        logging.info('{} ground truth instances from {} videos'.format(len(all_gt), len(self.video_ids)))

        # per class ground truth
        gt_by_cls = []
        for cls in range(num_classes):
            gt_by_cls.append(all_gt[all_gt.cls == cls].reset_index(drop=True).drop('cls', 1))

        self.gt_by_cls = gt_by_cls
        self.all_pred = {k: [] for k in self.nms_mode}
        self.num_classes = num_classes
        self.classes = classes
        self.anno_dict = anno_dict
        self.all_gt = all_gt
        self.num_workers = num_classes if num_workers is None else num_workers
        self.video_dict = video_dict
        self.stats = {k: dict() for k in self.nms_mode}
        self.subset = subset

    def _get_classes(self, anno_dict):
        if 'classes' in anno_dict:
            classes = anno_dict['classes']
        else:
            
            database = anno_dict['database']
            all_gts = []
            for vid in database:
                all_gts += database[vid]['annotations']
            classes = list(sorted({x['label'] for x in all_gts}))
        return classes

    def update(self, pred, assign_cls_labels=False):
        '''pred: a dict of predictions for each video. For each video, the predictions are in a dict with these fields: scores, labels, segments
        assign_cls_labels: manually assign class labels to the detections. This is necessary when the predictions are class-agnostic.
        '''
        pred_numpy = {k: {kk: vv.detach().cpu().numpy() for kk, vv in v.items()} for k,v in pred.items()}
        for k, v in pred_numpy.items():
            # pdb.set_trace()
            if 'window' not in k:
                this_dets = [
                    [v['segments'][i, 0], 
                     v['segments'][i, 1],
                     v['scores'][i], v['labels'][i]]
                     for i in range(len(v['scores']))]
                video_id = k
            else:
                window_start = self.video_dict[k]['time_offset']
                video_id = self.video_dict[k]['src_vid_name']
                this_dets = [
                    [v['segments'][i, 0] + window_start, 
                     v['segments'][i, 1] + window_start, 
                     v['scores'][i],
                     v['labels'][i]]
                    for i in range(len(v['scores']))]
            
            # ignore videos that are not in ground truth set
            if video_id not in self.video_ids:
                continue
            this_dets = np.array(this_dets)   # start, end, score, label
            
            for nms_mode in self.nms_mode:
                input_dets = np.copy(this_dets)
                # if nms_mode == 'nms' and not (cfg.TEST_SLICE_OVERLAP > 0 and self.dataset_name == 'thumos14'):  # when cfg.TEST_SLICE_OVERLAP > 0, only do nms at summarization
                #     dets = apply_nms(input_dets, nms_thr=cfg.nms_thr, use_soft_nms=self.dataset_name=='activitynet' and assign_cls_labels)
                # else:
                if True:
                    sort_idx = input_dets[:, 2].argsort()[::-1]
                    dets = input_dets[sort_idx, :]

                # only keep top 200 detections per video
                dets = dets[:200, :]

                # On ActivityNet, follow the tradition to use external video label
                if assign_cls_labels:
                        raise NotImplementedError
                self.all_pred[nms_mode] += [[video_id, k] + det for det in dets.tolist()]


    def nms_whole_dataset(self):
        video_ids = list(set([v['src_vid_name'] for k, v in self.video_dict.items()]))
        all_pred = []
        for vid in video_ids:
            this_dets = self.all_pred['nms'][self.all_pred['nms']['video-id'] == vid][['t-start', 't-end', 'score', 'cls']].values
            
            this_dets = apply_nms(this_dets)[:200, ...]
            this_dets = [[vid] + x.tolist() for x in this_dets]
            all_pred += this_dets
        self.all_pred['nms'] = pd.DataFrame(all_pred, columns=["video-id", "t-start", "t-end", "score", "cls"])

    def cross_window_fusion(self):
        '''
        merge detections in the overlapped regions of adjacent windows. Only used for THUMOS14
        '''
        # video_ids = list(set([v['src_vid_name'] for k, v in self.video_dict.items()]))
        all_pred = []

        video_ids = self.all_pred['raw']['video-id'].unique()
        vid = video_ids[0]

        for vid in video_ids:
            this_dets = self.all_pred['raw'][self.all_pred['raw']['video-id'] == vid]
            slice_ids = this_dets['slice-id'].unique().tolist()
            if len(slice_ids) > 1:
                slice_sorted = sorted(slice_ids, key=lambda k: int(k.split('_')[4]))
               
                overlap_region_time_list = []
                for i in range(0, len(slice_sorted) - 1):
                    slice_name = slice_sorted[i]
                    feature_fps = self.video_dict[slice_name]['feature_fps']
                    time_base = 0  # self.video_dict[slice_name]['time_base']
                    # parse the temporal coordinate from name
                    cur_slice = [int(x) for x in slice_sorted[i].split('_')[4:6]]
                    next_slice = [int(x) for x in slice_sorted[i+1].split('_')[4:6]]
                    overlap_region_time = [next_slice[0], cur_slice[1]]
                    # add time offset of each window/slice
                    overlap_region_time = [time_base + overlap_region_time[iii] / feature_fps for iii in range(2)]
                    overlap_region_time_list.append(overlap_region_time)
                
                mask_union = None
                processed_dets = []
                for overlap_region_time in overlap_region_time_list:
                    inters = np.minimum(this_dets['t-end'], overlap_region_time[1]) - np.maximum(this_dets['t-start'], overlap_region_time[0])
                    # we only perform NMS to the overlapped regions
                    mask = inters > 0
                    overlap_dets = this_dets[mask]
                    overlap_dets_arr = overlap_dets[['t-start', 't-end', 'score', 'cls']].values
                    if len(overlap_dets) > 0:
                        kept_dets_arr = apply_nms(np.concatenate((overlap_dets_arr, np.arange(len(overlap_dets_arr))[:, None]), axis=1))
                        processed_dets.append(overlap_dets.iloc[kept_dets_arr[:, -1].astype('int64')])
                    
                    if mask_union is not None:
                        mask_union = mask_union | mask
                    else:
                        mask_union = mask
                # instances not in overlapped region
                processed_dets.append(this_dets[~mask_union])
                all_pred += processed_dets
            else:
                all_pred.append(this_dets)

        all_pred = pd.concat(all_pred)
        self.all_pred['raw'] = all_pred

    def accumulate(self, test_slice_overlap=0):
        '''accumulate detections in all videos'''
        for nms_mode in self.nms_mode:
            self.all_pred[nms_mode] = pd.DataFrame(self.all_pred[nms_mode], columns=["video-id", "slice-id", "t-start", "t-end", "score", "cls"])
        
        self.pred_by_cls = {}
        for nms_mode in self.nms_mode:
            if self.dataset_name == 'thumos14' and nms_mode == 'raw' and test_slice_overlap > 0:
                self.cross_window_fusion()
            # if you really want to use NMS
            if self.dataset_name == 'thumos14' and nms_mode == 'nms' and test_slice_overlap > 0:
                self.nms_whole_dataset()

            self.pred_by_cls[nms_mode] = [self.all_pred[nms_mode][self.all_pred[nms_mode].cls == cls].reset_index(drop=True).drop('cls', 1) for cls in range(self.num_classes)]

    def import_prediction(self):
        pass

    def format_arr(self, arr, format='{:.2f}'):
        line = ' '.join([format.format(x) for x in arr])
        return line

    def synchronize_between_processes(self):
        mode = self.nms_mode[0]
        print(
            len(self.all_pred[mode]),
            len({x[0] for x in self.all_pred[mode]})
        )
        self.all_pred = merge_distributed(self.all_pred)

    def summarize(self):
        '''Compute mAP and collect stats'''
        if self.dataset_name in ['thumos14', 'muses']:
            # 0.3~0.7 avg
            display_iou_thr_inds = [0, 1, 2, 3, 4]
        else:
            # 0.5 0.75 0.95 avg
            display_iou_thr_inds = [0, 5, 9]
        
        for nms_mode in self.nms_mode:
            logging.info(
                'mode={} {} predictions from {} videos'.format(
                    nms_mode,
                    len(self.all_pred[nms_mode]),
                    len(self.all_pred[nms_mode]['video-id'].unique()))
            )

        header = ' '.join('%.2f' % self.iou_range[i] for i in display_iou_thr_inds) + ' avg'  # 0 5 9
        lines = []
        for nms_mode in self.nms_mode:
            per_iou_ap = self.compute_map(nms_mode)
            line = ' '.join(['%.2f' % (100*per_iou_ap[i]) for i in display_iou_thr_inds]) + ' %.2f' % (100*per_iou_ap.mean()) + ' {} epoch{}'.format(nms_mode, self.epoch)
            lines.append(line)
        msg = header
        for l in lines:
            msg += '\n' + l
        logging.info('\n' + msg)

        for nms_mode in self.nms_mode:
            if self.dataset_name == 'thumos14':
                self.stats[nms_mode]['AP50'] = self.stats[nms_mode]['per_iou_ap'][2]
            else:
                self.stats[nms_mode]['AP50'] = self.stats[nms_mode]['per_iou_ap'][0]
        self.stats_summary = msg

    def compute_map(self, nms_mode):
        '''Compute mean average precision'''
        start_time = time.time()

        gt_by_cls, pred_by_cls = self.gt_by_cls, self.pred_by_cls[nms_mode]

        iou_range = self.iou_range
        num_classes = self.num_classes
        ap_values = np.zeros((num_classes, len(iou_range)))

        with concurrent.futures.ProcessPoolExecutor(min(self.num_workers, 8)) as p:
            futures = []
            for cls in range(len(pred_by_cls)):
                if len(gt_by_cls[cls]) == 0:
                    logging.info('no gt for class {}'.format(self.classes[cls]))
                if len(pred_by_cls[cls]) == 0:
                    logging.info('no prediction for class {}'.format(self.classes[cls]))
                futures.append(p.submit(eval_ap, iou_range, cls, gt_by_cls[cls], pred_by_cls[cls]))
            for f in concurrent.futures.as_completed(futures):
                x = f.result()
                ap_values[x[0], :] = x[1]

        per_iou_ap = ap_values.mean(axis=0)
        per_cls_ap = ap_values.mean(axis=1)
        mAP = per_cls_ap.mean()
       
        self.stats[nms_mode]['mAP'] = mAP
        self.stats[nms_mode]['ap_values'] = ap_values
        self.stats[nms_mode]['per_iou_ap'] = per_iou_ap
        self.stats[nms_mode]['per_cls_ap'] = per_cls_ap
        return per_iou_ap

    def dump_to_json(self, dets, save_path):
        result_dict = {}
        videos = dets['video-id'].unique()
        for video in videos:
            this_detections = dets[dets['video-id'] == video]
            det_list = []
            for idx, row in this_detections.iterrows():
                det_list.append(
                    {'segment': [float(row['t-start']), float(row['t-end'])], 'label': self.classes[int(row['cls'])], 'score': float(row['score'])}
                )
            
            video_id = video[2:] if video.startswith('v_') else video
            result_dict[video_id] = det_list

        # the standard detection format for ActivityNet
        output_dict={
            "version": "VERSION 1.3",
            "results": result_dict,
            "external_data":{}}
        if save_path:
            dirname = osp.dirname(save_path)
            if not osp.exists(dirname):
                os.makedirs(dirname)
            with open(save_path, 'w') as f:
                json.dump(output_dict, f)
        # return output_dict

    def dump_detection(self, save_path=None):
        for nms_mode in self.nms_mode:
            logging.info(
                'dump detection result in JSON format to {}'.format(save_path.format(nms_mode)))
            self.dump_to_json(self.all_pred[nms_mode], save_path.format(nms_mode))


def merge_distributed(all_pred):
    '''gather outputs from different nodes at distributed mode'''
    all_pred_gathered = all_gather(all_pred)
    
    merged_all_pred = {k: [] for k in all_pred}
    for p in all_pred_gathered:
        for k in p:
            merged_all_pred[k] += p[k]

    return merged_all_pred

    
if __name__ == '__main__':
    pass


