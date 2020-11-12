# MEVA: 3D Human Motion Estimation via Motion Compression and Refinement 


[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2008.03789)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/3d-human-motion-estimation-via-motion/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=3d-human-motion-estimation-via-motion)

[This repo is still under construction]
---



<div float="center">
  <img src="media/meva_teaser.gif" />
</div>



**3D Human Motion Estimation via Motion Compression and Refinement**

Zhengyi Luo, S. Alireza Golestaneh, Kris M. Kitani

**ACCV 2020, Oral**  
[[Project website](https://zhengyiluo.github.io/projects/meva/)][[Quantitative Demo](https://youtu.be/YBb9NDz3ngM)][[10min Talk](https://youtu.be/-TN3NRpCEc0)]


## Notable

MEVA (Motion Estimation vis Variational Autoencoding) is a video-based 3D human pose estimation method that focus on producing **stable** and **natural-looking** human motion from videos. MEVA achieves state-of-the-art human pose estimation accuracy while reducing acceleration error siginificantly. Pleaser refer to our [paper](https://arxiv.org/abs/2008.03789) for more details.  



## Updates
- November 11, 2020 – 14:16 Inference code uploaded

## Getting Started

### Install:
#### Environment
- Tested OS: Linux
- Python >= 3.6

### How to install
Install the dependencies:
```
pip install -r requirements.txt
```

### Running inference/Demo

#### Prepare necessary data
To run pre-trained models, please run the script:
```
bash scripts/prepare_data.sh
```

Command:

```
python scripts/run_meva_on_video.py --cfg train_meva_2  --vid_file zen_talking_phone.mp4  --output_folder results/output --exp train_meva_2
```

## Training 

Training code coming soon!

### Prepare Datasets


## Evaluation

Here we compare MEVA with recent state-of-the-art methods on 3D pose estimation datasets. Evaluation metric is
Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) in mm.

| Models         | 3DPW &#8595; | MPI-INF-3DHP &#8595; | H36M &#8595; |
|----------------|:----:|:------------:|:----:|
| SPIN           | 59.2 |     67.5     | **41.1** |
| Temporal HMR   | 76.7 |     89.8     | 56.8 |
| VIBE           | 56.5 |     63.4     | 41.5 |
| MEVA           | **51.9** |     **62.6**     | 48.1 |

Eval code coming soon!

## Known issues
1. Visulization scale seems off somehow (the humanoid is not scaled properly), still debugging!


## Citation
If you find our work useful in your research, please cite our paper [MEVA](https://arxiv.org/abs/2008.03789):
```
@article{Luo20203DHM,
  title={3D Human Motion Estimation via Motion Compression and Refinement},
  author={Zhengyi Luo and S. Golestaneh and Kris M. Kitani},
  journal={ArXiv},
  year={2020},
  volume={abs/2008.03789}
}
```


## References:
Notice that this repo builds upon a number of previous great works (especially, [VIBE](https://github.com/mkocabas/VIBE)), and borrow scripts from them for convenience. For each file that is borrowed, we indicate that it is referenced and please adhere to their liscnece for usage. 

- Dataloaders, part of the loss function, data pre-processing are from: [VIBE](https://github.com/mkocabas/VIBE) 
- SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)
- Feature extractors are from: [SPIN](https://github.com/nkolot/SPIN)
- Some NN modules are from: [DLOW](https://github.com/Khrylx/DLow)