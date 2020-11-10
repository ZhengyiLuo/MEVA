import os
import sys
sys.path.append(os.getcwd())

import time
import numpy as np
from torch import nn
from torch.nn import functional as F

from meva.khrylib.models.mlp import MLP
from meva.khrylib.models.rnn import RNN
from meva.khrylib.utils.torch import *

from meva.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root,
    rotation_matrix_to_angle_axis, convert_orth_6d_to_mat, perspective_projection_cam
)

def MSE_func(X_r, X):
    diff = X_r - X
    MSE = diff.pow(2).sum() / X.shape[1]
    return MSE

def MSE_RT_func(X_r, X, root_dim):
    diff = X_r[:,:,:root_dim] - X[:,:,:root_dim]
    MSE_RT = diff.pow(2).sum() / X.shape[1]
    return MSE_RT

def MSE_TP_func(X_r, X):
    diff_tp = torch.abs(X_r[:-1,:,:] - X_r[1:,:,:]) - torch.abs(X[:-1,:,:] - X[1:,:,:])
    MSE_TP = diff_tp.pow(2).sum()/X.shape[1]
    return MSE_TP

def MSE_SMPL_Pt_func(X_r, X, chunk_size = 8):
    X_r_aa = convert_orth_6d_to_aa(X_r)
    X_aa = convert_orth_6d_to_aa(X)
    
    X_r_aa_chunks = torch.split(X_r_aa, chunk_size, dim=1)
    X_aa_chunks = torch.split(X_aa, chunk_size, dim=1)
    

    MSE_SMPL_PT = 0
    for i in range(len(X_r_aa_chunks)):
        X_r_verts, _ = smpl_p.get_vert_from_pose(X_r_aa_chunks[i])
        X_verts, _ = smpl_p.get_vert_from_pose(X_aa_chunks[i])
        diff_verts = X_r_verts - X_verts
        MSE_SMPL_PT += diff_verts.pow(2).sum()/X.shape[1]
        del X_r_verts, X_verts, diff_verts
        torch.cuda.empty_cache()   
    return MSE_SMPL_PT

def MSE_2D_PT_func(body_pose, betas, pred_cam, target_k2d, smpl):
    body_pose_aa = convert_orth_6d_to_aa(body_pose).permute(1, 0, 2)
    body_pose_aa_flat = body_pose_aa.reshape(-1, 24, 3)

    target_k2d_flat = target_k2d.reshape(-1, 49, 3)
    betas_flat = betas.reshape(-1, 10)
    pred_output = smpl(
            betas= betas_flat,
            body_pose=body_pose_aa_flat[:,1:],
            global_orient=body_pose_aa_flat[:, 0:1],
        )
    pred_joints = pred_output.joints

    # import pdb
    # pdb.set_trace()

    pred_keypoints_2d = perspective_projection_cam(pred_joints, pred_cam.reshape(-1, 3))
    conf = target_k2d_flat[:,:, -1].unsqueeze(-1).clone()
#     loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_j2d[:, :, :-1])).mean()
    diff = pred_keypoints_2d -  target_k2d_flat[:, :, :-1]
    MSE_2D_PT = (conf * diff.pow(2)).sum()/pred_keypoints_2d.shape[0]

    return MSE_2D_PT

def KLD_func(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    return KLD

def common_loss_func(cfg, **kwargs):
    data_specs = cfg.data_specs
    loss_specs = cfg.loss_specs
    loss_weights = loss_specs['loss_weights']
    loss_names = loss_specs['loss_names']
    root_dim = data_specs.get('root_dim', 6)

    if "MSE" in loss_names:
        X_r = kwargs["X_r"]
        X = kwargs["X"]
        MSE = MSE_func(X_r, X) 
    if "MSE_RT" in loss_names:
        X_r = kwargs["X_r"]
        X = kwargs["X"]
        MSE_RT = MSE_RT_func(X_r, X, root_dim = root_dim) 

    if "MSE_TP" in loss_names:
        X_r = kwargs["X_r"]
        X = kwargs["X"]
        MSE_TP = MSE_TP_func(X_r, X) 
    
    if "MSE_SMPL_Pts" in loss_names:
        X_r = kwargs["X_r"]
        X = kwargs["X"]
        MSE_SMPL_PT = MSE_SMPL_Pt_func(X_r, X) 

    if "KLD" in loss_names:
        mu = kwargs["mu"]
        logvar = kwargs["logvar"]
        KLD = KLD_func(mu, logvar) 

    if "MSE_2D_PT" in loss_names:
        body_pose = kwargs['body_pose']
        betas = kwargs['betas']
        pred_cam =kwargs['pred_cam']
        target_k2d = kwargs['target_k2d']
        smpl = kwargs['smpl']
        MSE_2D_PT = MSE_2D_PT_func(body_pose, betas, pred_cam, target_k2d, smpl)
        
    
    all_loss = []
    for name in loss_names:
        all_loss.append(loss_weights[name.lower()] * eval(name))

    loss_r = sum(all_loss)
    return loss_r, np.array([t.item() for t in all_loss])
 

def get_loss_func(cfg):
    # model_specs = cfg.model_specs
    # data_specs =cfg.data_specs
    # model_name = model_specs['model_name']
    return common_loss_func