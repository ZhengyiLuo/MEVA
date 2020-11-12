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
from copycat.smpllib.smpl_mujoco import SMPL_M_Renderer



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

    dataset_3dpw = joblib.load("/hdd/zen/data/ActBound/AMASS/3dpw_train_res.pkl")
    # dataset_3dpw = joblib.load("/hdd/zen/data/ActBound/AMASS/3dpw_val_res.pkl")
    # dataset_3dpw = joblib.load("/hdd/zen/data/ActBound/AMASS/3dpw_test_res.pkl")

    image_size = 400
    total = cfg.data_specs['t_total']

    if args.render:
        # renderer = SMPL_Renderer(device = device, image_size = 400, camera_mode="look_at")
        renderer = SMPL_M_Renderer(render_size = (image_size, image_size))
    eval_recs =[]
    # eval_vibe =[]
    idx = 0
    for k, v in tqdm(dataset_3dpw.items()):
        
        curr_name = v
        mocap_thetas = v['target_traj']
        vibe_thetas = v['traj']
        vis_feats = v['feat']
        mocap_betas = v['target_beta']
        vibe_betas = v['traj_beta']
        
        with torch.no_grad():
            vibe_pose = torch.tensor(vibe_thetas).squeeze().to(device)
            mocap_pose = torch.tensor(mocap_thetas).squeeze().to(device)
            vis_feats = torch.tensor(vis_feats).squeeze().to(device)
            vibe_betas = torch.tensor(vibe_betas).squeeze().to(device)
            mocap_betas = torch.tensor(mocap_betas).squeeze().to(device)

            mocap_pose_6d = convert_aa_to_orth6d(mocap_pose).reshape(-1, 144)
            mocap_pose_6d = mocap_pose_6d[None, :].permute(1, 0, 2)
            vibe_pose_6d = convert_aa_to_orth6d(vibe_pose).reshape(-1, 144)
            vibe_pose_6d = vibe_pose_6d[None, :].permute(1, 0, 2)
            vis_feats = vis_feats[None, :].permute(1, 0, 2)

            mocap_pose_6d_chunks = torch.split(mocap_pose_6d, total, dim=0)
            vibe_pose_6d_chunks = torch.split(vibe_pose_6d, total, dim=0)
            vis_feats_chunks = torch.split(vis_feats, total, dim=0)

            X_r_acc = []

            for i in range(len(mocap_pose_6d_chunks)):
                mocap_pose_chunk = mocap_pose_6d_chunks[i]
                vibe_pose_chunk = vibe_pose_6d_chunks[i]
                vis_feats_chunk = vis_feats_chunks[i]


                label_rl = torch.tensor([[1,0]]).to(device).float()

                X_r, mu, logvar = model(mocap_pose_chunk)
                X_r_acc.append(X_r[:mocap_pose_chunk.shape[0]])

            X_r = torch.cat(X_r_acc)
            X_r = X_r.permute(1,0,2)
            ref_pose_curr_rl = convert_orth_6d_to_aa(X_r.squeeze())

            ######## Rendering...... ########
            if args.render:
                mocap_pose = vertizalize_smpl_root(mocap_pose).cpu().numpy()
                ref_pose_curr_rl = vertizalize_smpl_root(ref_pose_curr_rl).cpu().numpy()

                tgt_images = renderer.render_smpl(mocap_pose)
                ref_images = renderer.render_smpl(ref_pose_curr_rl)

                grid_size = [1,2]
                videos = [tgt_images, ref_images]
                descriptions = ["Mocap", "VAE"]
                output_name = "{}/output_vae{:02d}.mp4".format(output_path, idx)
                assemble_videos(videos, grid_size, descriptions, output_name)
                print(output_name)
                idx += 1
            else:
                eval_acc = compute_metric_on_seqs(ref_pose_curr_rl, mocap_betas, mocap_pose, mocap_betas, smpl, J_regressor=J_regressor)
                eval_recs.append(eval_acc)
            
    print(np.mean(eval_recs, axis = 0))    
