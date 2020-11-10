# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

import h5py
import torch
import logging
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from meva.utils.video_config import VIBE_DB_DIR
from meva.utils.kp_utils import convert_kps
from meva.utils.image_utils import normalize_2d_kp, split_into_chunks

logger = logging.getLogger(__name__)

class Insta(Dataset):
    def __init__(self, seqlen, overlap=0., debug=False):
        self.seqlen = seqlen
        self.stride = int(seqlen * (1-overlap))

        self.h5_file = osp.join(VIBE_DB_DIR, 'insta_train_db.h5')

        with h5py.File(self.h5_file, 'r') as db:
            self.db = db
            self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)

        print(f'InstaVariety number of dataset objects {self.__len__()}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        with h5py.File(self.h5_file, 'r') as db:
            self.db = db

            kp_2d = self.db['joints2D'][start_index:end_index + 1]
            kp_2d = convert_kps(kp_2d, src='insta', dst='spin')
            kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)


            input = torch.from_numpy(self.db['features'][start_index:end_index+1]).float()

            vid_name = self.db['vid_name'][start_index:end_index + 1]
            frame_id = self.db['frame_id'][start_index:end_index + 1].astype(str)
            instance_id = np.array([v.decode('ascii') + f for v, f in zip(vid_name, frame_id)])

        for idx in range(self.seqlen):
            kp_2d[idx,:,:2] = normalize_2d_kp(kp_2d[idx,:,:2], 224)
            kp_2d_tensor[idx] = kp_2d[idx]

        target = {
            'features': input,
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(), # 2D keypoints transformed according to bbox cropping
            # 'instance_id': instance_id
        }

        return target