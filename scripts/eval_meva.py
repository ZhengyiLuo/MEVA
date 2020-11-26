import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import joblib
import numpy as np
from tqdm import tqdm 
import h5py
from collections import defaultdict

from meva.lib.meva_model import MEVA
from meva.lib.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from torch.utils.data import DataLoader
from meva.utils.video_config import parse_args, MEVA_DATA_DIR, VIBE_DB_DIR

from meva.utils.kp_utils import convert_kps
from meva.utils.image_utils import normalize_2d_kp, split_into_chunks, transfrom_keypoints, assemble_videos
from meva.utils.transform_utils import *
from skimage.util.shape import view_as_windows
import pickle as pk
from meva.utils.tools import get_chunk_with_overlap
from meva.utils.eval_utils import *
from meva.utils.video_config import parse_args
# from smplx import SMPL


PCK_THRESH = 150.0
AUC_MIN = 0.0
AUC_MAX = 200.0

def db_2_dataset(dataset_data):

    return dataset_data

 
if __name__ == "__main__":
    cfg, cfg_file = parse_args()
    SMPL_MAJOR_JOINTS = np.array([1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21])
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    smpl = SMPL(
                SMPL_MODEL_DIR,
                batch_size=64,
                create_transl=False, 
            )

    meva_model = MEVA(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
        cfg = cfg.VAE_CFG,
    ).to(device)

    meva_dir = 'results/meva/train_meva_2/model_best.pth.tar'
    # meva_dir = 'results/meva/16-11-2020_14-49-27_meva/model_best.pth.tar'
    checkpoint = torch.load(meva_dir)
    best_performance = checkpoint['performance']
    meva_model.load_state_dict(checkpoint['gen_state_dict'])
    meva_model.eval()
    print(f'==> Loaded pretrained model from {meva_dir}...')
    # print(f'Performance on 3DPW test/val  {best_performance}')

    dtype = torch.float
    image_size = 400
    J_regressor = torch.from_numpy(np.load(osp.join(MEVA_DATA_DIR, 'J_regressor_h36m.npy'))).float()
    joints_map = np.array([0, 1, 2, 4, 5, 16, 17, 18, 19])

    ################## Data ##################
    t_total = 90
    overlap = 10
    dataset_data = joblib.load("/hdd/zen/data/video_pose/vibe_db/3dpw_test_db.pt")
    # dataset_data = joblib.load("/hdd/zen/data/video_pose/vibe_db/h36m_test_db.pt")
    # dataset_data = joblib.load("/hdd/zen/data/video_pose/vibe_db/mpii3d_test_db.pt")
    out_dir = "/hdd/zen/data/video_pose/3dpw/meva_res/res"
    full_res = defaultdict(list)
    

    valid_names = dataset_data['vid_name']
    unique_names = np.unique(valid_names)
    data_keyed = {}
    for u_n in unique_names:
        indexes = valid_names == u_n
        if "valid" in dataset_data:
            valids = dataset_data['valid'][indexes].astype(bool)
        else:
            valids = np.ones(dataset_data['features'][indexes].shape[0]).astype(bool)

        data_keyed[u_n] = {
            "features": dataset_data['features'][indexes][valids], 
            "joints3D": dataset_data['joints3D'][indexes][valids]
        }
    dataset_data = data_keyed

    
    with torch.no_grad():
        pbar = tqdm(dataset_data.keys())
        for seq_name in pbar:
            curr_feats = dataset_data[seq_name]['features']
            res_save = {}
            curr_feat = torch.tensor(curr_feats).to(device)
            num_frames = curr_feat.shape[0]
            if num_frames < t_total:
                if num_frames < t_total:
                    print(f"Video < {t_total} frames")
                    
                # print(f"video too short, padding..... {num_frames}")
                # curr_feat = torch.from_numpy(np.repeat(curr_feats, t_total//num_frames + 1, axis = 0)[:t_total].copy()).to(device)
                # chunk_idxes = np.array(list(range(0, t_total)))[None, ]
                # chunck_selects = [(0, num_frames)]
                
            else:
                chunk_idxes, chunck_selects = get_chunk_with_overlap(num_frames, window_size = t_total, overlap=overlap)

            meva_theta, meva_j3d= [], []
            for curr_idx in range(len(chunk_idxes)):
                chunk_idx = chunk_idxes[curr_idx]
                cl = chunck_selects[curr_idx]
                meva_preds = meva_model(curr_feat[None, chunk_idx, :], J_regressor = J_regressor)
                
                meva_theta.append(meva_preds[-1]['theta'][0,cl[0]:cl[1],3:75].cpu().numpy())
                meva_j3d.append(meva_preds[-1]['kp_3d'][0, cl[0]:cl[1]].cpu().numpy())
                
            meva_theta = np.vstack(meva_theta)

            meva_j3d = np.vstack(meva_j3d)
            gt_j3d = dataset_data[seq_name]['joints3D']

            if gt_j3d.shape[1] == 49:
                gt_j3d = convert_kps(gt_j3d, src='spin', dst='common')

            mpjpe, mpjpe_pa, error_vel, error_acc, errors_pck, mat_procs = compute_errors(meva_j3d * 1000, gt_j3d * 1000)
            pck_final = compute_pck(errors_pck, PCK_THRESH) * 100.
            auc_range = np.arange(AUC_MIN, AUC_MAX)
            pck_aucs = []
            for pck_thresh_ in auc_range:
                err_pck_tem = compute_pck(errors_pck, pck_thresh_)
                pck_aucs.append(err_pck_tem)

            auc_final = compute_auc(auc_range / auc_range.max(), pck_aucs)


            full_res['mpjpe'].append(mpjpe)
            full_res['mpjpe_pa'].append(mpjpe_pa)
            full_res['vel_err'].append(error_vel)
            full_res['acc_err'].append(error_acc)
            full_res['auc'].append([auc_final])
            full_res['pck'].append([pck_final])


            pbar.set_description(f"{np.mean(mpjpe):.3f}")


        np.set_printoptions(precision=4, suppress=1)
        full_res = {k: np.mean(np.concatenate(v)) for k, v in full_res.items()}
        print(full_res)
        
