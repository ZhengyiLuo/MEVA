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
import copy

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
from meva.utils.tools import get_chunk_with_overlap
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
    h36m_data = joblib.load("/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_60_fitted.p")

    overlap = 10
    total = cfg.data_specs['t_total']
    t_total = total
    if args.render:
        # renderer = SMPL_Renderer(device = device, image_size = 400, camera_mode="look_at")
        renderer = SMPL_M_Renderer(render_size = (image_size, image_size))
    eval_recs =[]
    # eval_vibe =[]
    filtered_res = {}
    idx = 0
    for k, v in tqdm(h36m_data.items()):
        curr_name = v
        mocap_thetas = v['pose']
        
        with torch.no_grad():
            mocap_pose = torch.tensor(mocap_thetas).squeeze().to(device)

            mocap_pose_6d = convert_aa_to_orth6d(mocap_pose).reshape(-1, 144)
            mocap_pose_6d = mocap_pose_6d[None, :].permute(1, 0, 2)
            num_frames = mocap_pose.shape[0]

            chunk_idxes, chunck_selects = get_chunk_with_overlap(num_frames, window_size = t_total, overlap=overlap)
            X_r_acc = []
            for curr_idx in range(len(chunk_idxes)):
                chunk_idx = chunk_idxes[curr_idx]
                cl = chunck_selects[curr_idx]
                try:
                    X_r, mu, logvar = model(mocap_pose_6d[chunk_idx, :, :])
                    
                except Exception as e:
                    import pdb; pdb.set_trace()
                X_r_acc.append(X_r[cl[0]:cl[1], :])


            X_r = torch.cat(X_r_acc)
            X_r = X_r.permute(1,0,2)
            ref_pose_curr_rl = convert_orth_6d_to_aa(X_r.squeeze())

            zeros = torch.zeros((mocap_pose.shape[0], 10)).to(device)
            # eval_acc = compute_metric_on_seqs(ref_pose_curr_rl, zeros, mocap_pose, zeros, smpl, J_regressor=J_regressor)
            # eval_recs.append(eval_acc)
            
            filtered_res[k] = copy.deepcopy(v)
            filtered_res[k]['pose'] = ref_pose_curr_rl.detach().cpu().numpy().copy()

            del ref_pose_curr_rl
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    joblib.dump(filtered_res, "/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_60_filtered.p")
            
    # print(np.mean(eval_recs, axis = 0))    
