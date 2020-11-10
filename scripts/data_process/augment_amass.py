import os
import sys
import pdb
sys.path.append(os.getcwd())

import numpy as np
import glob
import pickle as pk 
import joblib
import torch 

from tqdm import tqdm
from meva.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root,
    rotation_matrix_to_angle_axis, rot6d_to_rotmat
)

from scipy.spatial.transform import Rotation as sRot
np.random.seed(1)
left_right_idx = [ 0,  2,  1,  3,  5,  4,  6,  8,  7,  9, 11, 10, 12, 14, 13, 15, 17,16, 19, 18, 21, 20, 23, 22]

def left_to_rigth_euler(pose_euler):
    pose_euler[:,:, 0] = pose_euler[:,:,0] *  -1
    pose_euler[:,:,2] = pose_euler[:,:,2] * -1
    pose_euler = pose_euler[:,left_right_idx,:]
    return pose_euler

def flip_smpl(pose):
    '''
        Pose input batch * 72
    '''
    curr_spose = sRot.from_rotvec(pose.reshape(-1, 3))
    curr_spose_euler = curr_spose.as_euler('ZXY', degrees=False).reshape(pose.shape[0], 24, 3)
    curr_spose_euler = left_to_rigth_euler(curr_spose_euler)
    curr_spose_aa = sRot.from_euler("ZXY", curr_spose_euler.reshape(-1, 3), degrees = False).as_rotvec().reshape(pose.shape[0], 24, 3)
    return curr_spose_aa.reshape(-1, 72)


def sample_random_hemisphere_root():
    rot = np.random.random() * np.pi * 2
    pitch =  np.random.random() * np.pi/3 + np.pi
    r = sRot.from_rotvec([pitch, 0, 0])
    r2 = sRot.from_rotvec([0, rot, 0])
    root_vec = (r * r2).as_rotvec()
    return root_vec

def sample_seq_length(seq, tran, seq_length = 150):
    if seq_length != -1:
        num_possible_seqs = seq.shape[0] // seq_length
        max_seq = seq.shape[0]

        start_idx = np.random.randint(0, 10)
        start_points = [max(0, max_seq - (seq_length + start_idx))]

        for i in range(1, num_possible_seqs - 1):
            start_points.append(i * seq_length + np.random.randint(-10, 10))

        if num_possible_seqs >= 2:
            start_points.append(max_seq - seq_length - np.random.randint(0, 10))

        seqs = [seq[i:(i + seq_length)] for i in start_points]
        trans = [tran[i:(i + seq_length)] for i in start_points]
    else:
        seqs = [seq]
        trans = [tran]
        start_points = []
    return seqs, trans, start_points

def get_random_shape(batch_size):
    shape_params = torch.rand(1, 10).repeat(batch_size, 1)
    s_id = torch.tensor(np.random.normal(scale = 1.5, size = (3)))
    shape_params[:,:3] = s_id
    return shape_params




if __name__ == "__main__":
    amass_base = "/hdd/zen/data/ActBound/AMASS/"
    take_num = "take8"
    # amass_cls_data = pk.load(open(os.path.join(amass_base, "amass_class.pkl"), "rb"))
    amass_seq_data = {}
    seq_length = -1

    target_frs = [20,30,40] # target framerate
    video_annot = {}
    counter = 0 
    seq_counter = 0
    amass_db = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_db.pt")
    pbar = tqdm(amass_db.items())
    for (k, v) in pbar:
        pbar.set_description(k)
        amass_pose = v['poses']
        amass_trans = v['trans']

        amass_pose_back = amass_pose[::-1].copy() 
        amass_trans_back = amass_trans[::-1].copy()
        amass_fr = v['mocap_framerate']
        
        
        amass_pose_flip = flip_smpl(amass_pose)
        skips = np.unique([int(amass_fr/target_fr) for target_fr in target_frs]).astype(int)
        
        seqs, start_points, trans = [],[],[]
        for skip in skips:
            seqs_org, trans_org, start_points_org = sample_seq_length(amass_pose[::skip], amass_trans[::skip], seq_length)
            seqs_flip, trans_flip, start_points_flip = sample_seq_length(amass_pose_flip[::skip], amass_trans[::skip], seq_length)
            
            seqs = seqs + seqs_org + seqs_flip 
            trans = trans + trans_org + trans_flip
            start_points = start_points + start_points_org + start_points_flip 
            
        seq_counter += len(seqs)
        for idx in range(len(seqs)):
            
            curr_seq = torch.tensor(seqs[idx])
    #         if curr_seq.shape[0] != seq_length: break
            cur_trans = trans[idx]
            for rnd_root in range(3):
                video_id = "{:06d}".format(counter)
                amass_key = video_id + "_amass"
    #             video_annot[video_id] = (k, idx, rnd_root, start_points[idx])
                video_annot[video_id] = (k, idx, rnd_root)
                
                sampled_root = sample_random_hemisphere_root()
                curr_seq = vertizalize_smpl_root(curr_seq, root_vec = sampled_root)
                pose_seq_6d = convert_aa_to_orth6d(torch.tensor(curr_seq)).reshape(-1, 144)
                beta = get_random_shape(1).squeeze().numpy()
                amass_seq_data[amass_key] = {
                    "label": 1,
                    "pose": pose_seq_6d,
                    'trans': cur_trans,
                    'beta': beta
                }
                counter += 1

    amass_output_file_name = "/hdd/zen/data/ActBound/AMASS/amass_{}.pkl".format(take_num)
    print(amass_output_file_name, len(amass_seq_data))
    joblib.dump(amass_seq_data, open(amass_output_file_name, "wb"))
