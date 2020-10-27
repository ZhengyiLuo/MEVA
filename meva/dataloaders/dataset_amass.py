from PIL import Image
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import time
import random
import copy
import scipy.misc
import scipy.io as scio
import glob
import pickle as pk
import joblib

## AMASS Datatset with Class
class DatasetAMASSCLS(data.Dataset):
    def __init__(self, data_specs, mode = "all"):
        print("******* Reading AMASS Class Data, Pytorch! ***********")
        np.random.seed(0) # train test split need to stablize

        self.data_root = data_specs['file_path']
        self.pickle_data = joblib.load(open(self.data_root, "rb"))
        # self.amass_data = amass_path
        self.has_smpl_root = data_specs['has_smpl_root']
        self.load_class = data_specs['load_class']
        self.flip_cnd  = data_specs['flip_cnd']
        self.t_total = data_specs['t_total']
        self.nc = data_specs['nc']
        self.to_one_hot = data_specs.get("to_one_hot", True)
        
        self.prepare_data()
        
        
        print("Dataset Root: ", self.data_root)
        print("Dataset Flip setting: ", self.flip_cnd)
        print("Dataset has SMPL root?: ", self.has_smpl_root)
        print("Dataset Num Sequences: ", self.seq_len)
        print("Traj Dimsnion: ", self.traj_dim)
        print("Load Class: ", self.load_class)
        print("******* Finished AMASS Class Data ***********")
        
    def prepare_data(self):
        trajs, target_trajs, class_labels, entry_names = self.process_data_pickle(self.pickle_data)
        self.data = {'traj': trajs, 'target_trajs': target_trajs, 'label': class_labels, "entry_name": entry_names}
        self.data_keys = list(trajs.keys())
        self.traj_dim = list(trajs.values())[0].shape[1]
        self.seq_len = len(trajs)

    def process_data_pickle(self, pk_data):
        trajs = {}
        target_trajs = {}
        class_labels = {}
        entry_names = {}
        for k, v in pk_data.items():
            smpl_squence = v["pose"]
            if "flip_pose" in v:
                smpl_flip_squence = v["flip_pose"]
            else:
                smpl_flip_squence = []
            class_label = v["label"]
            
            if not self.has_smpl_root:
                smpl_squence = smpl_squence[:,6:132]
                smpl_flip_squence = [i[:,6:132] for i in smpl_flip_squence]

            if self.load_class != -1:
                if class_label != self.load_class:
                    continue
            

            ## orig_traj -> target traj
            if smpl_squence.shape[0] >= self.t_total:
                # print(self.idx_to_one_hot(num_class, i), num_class, i)
                trajs[k] = smpl_squence
                target_trajs[k] = smpl_squence
                entry_names[k] = k

                
                class_labels[k] = self.idx_to_one_hot(self.nc, class_label) if self.to_one_hot else class_label

                if self.flip_cnd == 1: # Cnd = 1, has cross hold
                    if class_label == 0:
                        # Input is VIBE sequences
                        k_v2m = k + "_v2m"
                        trajs[k_v2m] = smpl_squence
                        target_trajs[k_v2m] = smpl_flip_squence
                        entry_names[k_v2m] = k_v2m

                        class_labels[k_v2m] = self.idx_to_one_hot(self.nc, 1) if self.to_one_hot else class_label
                    elif class_label == 1 and len(smpl_flip_squence) > 0:
                        k_m2v = k + "_m2v"
                        trajs[k_m2v] = smpl_squence
                        target_trajs[k_m2v] = smpl_flip_squence[np.random.choice(len(smpl_flip_squence))]
                        entry_names[k_m2v] = k_m2v

                        class_labels[k_m2v] = self.idx_to_one_hot(self.nc, 0) if self.to_one_hot else class_label

        return trajs, target_trajs, class_labels, entry_names
        

    def __getitem__(self, index):

        curr_key = self.data_keys[index]
        curr_traj = self.data['traj'][curr_key]
        seq_len = curr_traj.shape[0]
        fr_start = torch.randint(seq_len - self.t_total, (1, )) if seq_len - self.t_total != 0 else 0
        fr_end = fr_start + self.t_total
        curr_traj = curr_traj[fr_start:fr_end]
        curr_tgt_traj = self.data['target_trajs'][curr_key][fr_start:fr_end]
        
        sample = {
            'traj': curr_traj,
            'target_trajs': curr_tgt_traj,
            'label': self.data['label'][curr_key],
            'entry_name': self.data['entry_name'][curr_key], 
        }
        return sample

    def __len__(self):
        return self.seq_len
            
    def string_to_one_hot(self, class_name):
        return np.array(self.chosen_classes == class_name).reshape(1, -1).astype(np.uint8)

    def string_to_cls_index(self, class_name):
        max_label = np.argmax(self.string_to_one_hot(class_name), axis = 1)
        return max_label

    def idx_to_one_hot(self, num_class, idx):
        hot = np.zeros(num_class)
        hot[idx] = 1
        return hot[np.newaxis,:]
    
    def sampling_generator(self, batch_size=8, num_samples=5000, num_workers = 8):
        # self.seq_len = num_samples ## Using only subset of the data
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=True, num_workers=num_workers)
        return loader
    
    def iter_generator(self, batch_size=8, num_workers= 8):
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=False, num_workers=num_workers)
        return loader

if __name__ == '__main__':
    np.random.seed(0)
    data_specs = {
        "dataset_name": "amass_rf",
        "file_path": "/hdd/zen/data/ActBound/AMASS/amass_take7_test.pkl",
        "flip_cnd": 0,
        "has_smpl_root": True,
        "traj_dim": 144,
        "t_total": 90,
        "nc": 2,
        "load_class": -1,
        "root_dim": 6,
    }
    amass_path = "/hdd/zen/data/ActBound/AMASS/real_fake_all_take3.pkl"
    dataset = DatasetAMASSCLS(data_specs)
    for i in range(10):
        generator = dataset.sampling_generator(num_samples=5000, batch_size=1, num_workers=1)
        for data in generator:
            break
        print("-------")