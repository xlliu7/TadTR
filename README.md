# TadTR
This repo holds the code for TadTR, described in the technical report:
```
End-to-end temporal action detection with Transformer
Xiaolong Liu, Qimeng Wang, Yao Hu, Xu Tang, Song Bai, Xiang Bai 
```
TadTR achieves an ideal speed-accuracy trade-off for temporal action detection. It outperforms recent temporal action detectors while running more than 4x faster. It achieves 30.87 average mAP on HACS Segments and 60.0 mAP@IoU=0.5 when combined with MUSES-Net.

For more details, please refer to the report on [ArXiv](https://arxiv.org/abs/2106.10271). 

# Updates
2021.06.24 A combination of TadTR and MUSES-Net achieves 60.0% mAP, a new record on THUMOS14!.

# Related Projects
- [MUSES](https://github.com/xlliu7/MUSES): a dataset along with a baseline approach for multi-shot temporal event localization, presented at CVPR 2021.
