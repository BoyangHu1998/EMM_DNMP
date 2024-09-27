  # Enhanced Mesh Method (EMM) for Deformable Neural Mesh Primitives (DNMP): Improving Rendering Accuracy in Sparse Data Environments


<!-- [![arXiv](https://img.shields.io/badge/arXiv-2307.10173-b31b1b.svg)](https://arxiv.org/abs/2307.10776) <a href="https://dnmp.github.io/">
<img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> 
<a href="https://www.youtube.com/watch?v=JABhlaVq4VA"><img alt="Demo" src="https://img.shields.io/badge/-Demo-ea3323?logo=youtube"></a>  -->

<!-- ## Introduction
The repository contains the official implementation of source code and pre-trained models of our paper:*"[Urban Radiance Field Representation with Deformable Neural Mesh Primitives]()"*. It is a new representation to model urban scenes for efficient and high-quality rendering! -->

## Datasets
<!-- We conduct experiments on two outdoor datasets: KITTI-360 dataset, Waymo-Open-Dataset.
Please refer to preprocess/README.md for more details. -->
DTU

## Environments

1. Compile fairnr.
```
python setup.py build_ext --inplace
```

2. Main requirements:
- CUDA (tested on cuda-11.1)
- PyTorch (tested on torch-1.9.1)
- [pytorch3d](https://pytorch3d.org/)
- [torchsparse](https://github.com/mit-han-lab/torchsparse)

3. Other requirements are provided in `requirements.txt`

## Preprocessing

1. Run util/dtu_dataset.py to get the `dtu_pcs.npz`

2. Ensure  `pretrained/mesh_ae/mesh_ae.pth` existing

## Training
NOTE: USE launch.json sent by wechat to replace the original one.

1. Optimize geometry using our pre-trained auto-encoder by running `train_geo.py`. (Please specify `scence_name`,`dataroot`,`pts_file`, `log_dir` and `checkpoint_dir` in the script.) (run on two different voxel size)

2. Train radiance field by running `train_render.py`. (Please specify `scence_name`,`dataroot`,`pts_file`, `log_dir`, `checkpoint_dir`, `voxel_size_list`, `pretrained_geo_list` in the script.)

## Evaluation

You can run `test_render.py` for evaluation. (Please specify `train_render.py`. (Please specify `scence_name`,`dataroot`,`pts_file`, `log_dir`, `checkpoint_dir`, `voxel_size_list`, `pretrained_geo_list` and the `pretrained_render` in the script.)


## Citation

```
@article{lu2023dnmp,
  author    = {Lu, Fan and Xu, Yan and Chen, Guang and Li, Hongsheng and Lin, Kwan-Yee and Jiang, Changjun},
  title     = {Urban Radiance Field Representation with Deformable Neural Mesh Primitives},
  journal   = {ICCV},
  year      = {2023},
}
```
