# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

import os
import cv2
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from collections import defaultdict

from meva.utils.smooth_bbox import get_all_bbox_params, smooth_bbox_params
from meva.utils.image_utils import get_single_image_crop_demo
from meva.utils.tools import get_chunk_with_overlap


class Inference(Dataset):
    def __init__(self, image_folder, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224, smooth = True):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)
        self.image_file_names = np.array(self.image_file_names)[frames]

        if smooth:
            smoothed_bbox = smooth_bbox_params(bboxes[:, :3])
            bboxes = np.hstack((smoothed_bbox, smoothed_bbox[:, 2:3]))

        self.bboxes = bboxes 
        self.joints2d = joints2d
        self.scale = scale
        self.crop_size = crop_size
        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False

        self.norm_joints2d = np.zeros_like(self.joints2d)

        if self.has_keypoints:
            bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
            bboxes[:, 2:] = 150. / bboxes[:, 2:]
            self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

            self.image_file_names = self.image_file_names[time_pt1:time_pt2]
            self.joints2d = joints2d[time_pt1:time_pt2]
            self.frames = frames[time_pt1:time_pt2]

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)
        bbox = self.bboxes[idx]
        j2d = self.joints2d[idx] if self.has_keypoints else None
        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=self.scale,
            crop_size=self.crop_size)
        if self.has_keypoints:
            return norm_img, kp_2d
        else:
            return norm_img

    def iter_data(self, fr_num = 90, overlap = 20):

        images = np.array([cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB) for img_name in self.image_file_names])
        bboxes = self.bboxes
        all_norm_images = [get_single_image_crop_demo(images[idx], bboxes[idx], scale=self.scale, crop_size=self.crop_size)[0] for idx in range(images.shape[0])]
        norm_imgs = torch.stack(all_norm_images, dim = 0)
        

        seq_len = norm_imgs.shape[0]
        chunk_idxes, chunk_selects = get_chunk_with_overlap(seq_len, window_size = fr_num, overlap=overlap)
        data_chunk = []
        for curr_idx in range(len(chunk_idxes)):
            chunk_idx = chunk_idxes[curr_idx]
            chunk_select = chunk_selects[curr_idx]
            data_return = {}
            data_return["batch"] = norm_imgs[chunk_idx]
            data_return['cl'] = np.array(chunk_select)
            data_chunk.append(data_return)
            
        return data_chunk


class ImageFolder(Dataset):
    def __init__(self, image_folder):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)
        return to_tensor(img)
