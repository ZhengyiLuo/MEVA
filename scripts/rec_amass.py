import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import math
import pickle as pk
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import joblib

from khrylib.utils import *
from meva.utils.config import Config
from meva.lib.model import *
from meva.utils.transform_utils import *
from meva.utils.image_utils import *
from meva.lib.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from meva.utils.video_config import MEVA_DATA_DIR 
from meva.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
    smpl_to_joints, 
    compute_metric_on_seqs
)
# from copycat.smpllib.smpl_mujoco import SMPL_M_Renderer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--image_size", action="store_true", default=400)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--iter", type=int, default=-2)
    args = parser.parse_args()

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg_name = args.cfg
    cfg = Config(args.cfg)
    gpu_index = args.gpu_index
    device = torch.device('cuda', index=gpu_index)
    image_size = args.image_size

    has_smpl_root = cfg.data_specs['has_smpl_root']
    model, _, run_batch = get_models(cfg, iter = args.iter)
    model.to(device)
    model.eval()

    smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=50,
            create_transl=False,
            dtype = dtype
        ).to(device)
    J_regressor = torch.from_numpy(np.load(osp.join(MEVA_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    output_base = "/hdd/zen/data/ActmixGenenerator/output/3dpw"
    output_path = osp.join(output_base, cfg_name)
    if not osp.isdir(output_path): os.makedirs(output_path)

    # dataset_3dpw = joblib.load("/hdd/zen/data/ActBound/AMASS/3dpw_train_res.pkl")
    # dataset_3dpw = joblib.load("/hdd/zen/data/ActBound/AMASS/3dpw_val_res.pkl")
    # dataset_3dpw = joblib.load("/hdd/zen/data/ActBound/AMASS/3dpw_test_res.pkl")

    # dataset_amass = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_take7.pkl")
    dataset_amass = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_copycat_take5_test.pkl")
    image_size = 400
    total = cfg.data_specs['t_total']

    eval_recs =[]
    # eval_vibe =[]
    idx = 0
    for k, v in tqdm(dataset_amass.items()):
        # if idx > 10: break
        curr_name = v
      
        
        with torch.no_grad():
            pose_6d = torch.from_numpy(v['pose_6d'].reshape(-1, 52, 6))
            pose_aa = torch.from_numpy(v['pose_aa'].reshape(-1, 52, 3))
            batch_size, _,  _ = pose_aa.shape
            pose_6d = torch.cat([pose_6d[:, :22, :], torch.zeros([batch_size, 2, 6])], dim=1)[:90]
            pose_aa = torch.cat([pose_aa[:, :22, :], torch.zeros([batch_size, 2, 3])], dim=1)[:90]
            pose_aa_6d = convert_aa_to_orth6d(pose_aa).reshape(-1, 144).to(device)
            pose_aa_6d = pose_aa_6d[None, :].permute(1, 0, 2)
            X_r, mu, logvar = model(pose_aa_6d)
            X_r = X_r.permute(1,0,2)
            ref_pose_curr_rl = convert_orth_6d_to_aa(X_r.squeeze())
            joblib.dump({"rec": ref_pose_curr_rl.cpu().numpy(), "gt": pose_aa.numpy()}, "test.pkl")
        

