# TadTR: End-to-end Temporal Action Detection with Transformer

By [Xiaolong Liu](https://github.com/xlliu7), [Qimeng Wang](https://scholar.google.com/citations?user=hi7AeE8AAAAJ), [Yao Hu](https://scholar.google.com/citations?user=LIu7k7wAAAAJ), [Xu Tang](https://scholar.google.com/citations?user=grP24aAAAAAJ), Shiwei Zhang, [Song Bai](http://songbai.site), [Xiang Bai](https://scholar.google.com/citations?user=UeltiQ4AAAAJ).

This repo holds the code for TadTR, described in the technical report:
[End-to-end temporal action detection with Transformer](https://arxiv.org/abs/2106.10271)

_We have significantly improved the performance of TadTR since our initial submission to arxiv in June 2021. It achives much better performance now. Please refer to the latest version (v3) on arxiv._ 

We have also explored fully end-to-end training from RGB images with TadTR. See our CVPR 2022 work [E2E-TAD][e2e-tad].


## Introduction

TadTR is an end-to-end Temporal Action Detection TRansformer. It has the following advantages over previous methods:
- Simple. It adopts a set-prediction pipeline and achieves TAD with a *single network*. It does not require a separate proposal generation stage.
- Flexible. It removes hand-crafted design such as anchor setting and NMS.
- Sparse. It produces very sparse detections (e.g. 10 on ActivityNet), thus requiring lower computation cost.
- Strong. As a *self-contained* temporal action detector, TadTR achieves state-of-the-art performance on HACS and THUMOS14. It is also much stronger than concurrent Transformer-based methods such as **RTD-Net** and **AGT**.

![](arch.png "Architecture")

## Updates
[2022.7] Glad to share that this paper will appear in IEEE Transactions on Image Processing (TIP). Although I am still busy with my thesis, I will try to make the code accessible soon. Thanks for your patience.

[2022.6] Update the technical report of this work on arxiv (now v3).

[2022.3] Our new work [E2E-TAD][e2e-tad] based on TadTR is accepted to CVPR 2022. It supports fully end-to-end training from RGB images.

[2021.9.15] Update the performance on THUMOS14.

[2021.9.1] Add demo code.

## TODOs
- [x] add model code
- [ ] add inference code
- [ ] add training code
- [ ] support training/inference with video input

## Main Results
- HACS Segments

|Method|Feature|mAP@0.5|mAP@0.75|mAP@0.95|Avg. mAP|Model|
| :----: |:----: | :--: | :----: | :---: | :----: |:----: |  
|TadTR|I3D RGB|47.14 |32.11 |10.94| 32.09|[OneDrive]|


- THUMOS14

|Method|Feature|mAP@0.3|mAP@0.4|mAP@0.5|mAP@0.6|mAP@0.7|Avg. mAP|Model|
| :----: |:----: | :--: | :----: | :---: | :----: |:----: | :----: |:----: |
|TadTR|I3D 2stream|74.8 |69.1| 60.1| 46.6| 32.8| 56.7|[OneDrive]

- ActivityNet-1.3

|Method|Feature|mAP@0.5|mAP@0.75|mAP@0.95|Avg. mAP|Model|
| :----: |:----: | :--: | :----: | :---: | :----: |:----: | 
|TadTR|TSN 2stream|51.29 |34.99| 9.49| 34.64|[OneDrive]|
|TadTR|TSP|53.62| 37.52| 10.56| 36.75|[OneDrive]|


## Install
### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```
### Compiling CUDA extensions

```bash
cd model/ops;

# If you have multiple installations of CUDA Toolkits, you'd better add a prefix
# CUDA_HOME=<your_cuda_toolkit_path> to specify the correct version. 
python setup.py build_ext --inplace
```

### Run a quick test
```
python demo.py
```

## Data Preparation
To be updated.

## Training
Run the following command
```
bash scripts/train.sh DATASET
```

## Testing
```
bash scripts/test.sh DATASET WEIGHTS
```

## Acknowledgement
The code is based on the [DETR](https://github.com/facebookresearch/detr) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). We also borrow the implementation of the RoIAlign1D from [G-TAD](https://github.com/Frostinassiky/gtad). Thanks for their great works.

## Citing
```
@article{liu2021end,
  title={End-to-end Temporal Action Detection with Transformer},
  author={Liu, Xiaolong and Wang, Qimeng and Hu, Yao and Tang, Xu and Zhang, Shiwei and Bai, Song and Bai, Xiang},
  journal={arXiv preprint arXiv:2106.10271},
  year={2021}
}
```

## Contact

For questions and suggestions, please contact Xiaolong Liu at "liuxl at hust dot edu dot cn".

[e2e-tad]: https://github.com/xlliu7/E2E-TAD
