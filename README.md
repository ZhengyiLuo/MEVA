# MEVA: 3D Human Motion Estimation via Motion Compression and Refinement 


[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2008.03789)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/3d-human-motion-estimation-via-motion/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=3d-human-motion-estimation-via-motion)

[This repo is still under construction]
---



<div float="center">
  <img src="media/meva_teaser.gif" />
</div>



**3D Human Motion Estimation via Motion Compression and Refinement**

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

```
python scripts/run_meva_on_video.py --cfg train_meva_2  --vid_file zen_talking_phone.mp4  --output_folder results/output --exp 10-11-2020_20-51-44_meva
```

## Training 

### Prepare Datasets


## Evaluation


