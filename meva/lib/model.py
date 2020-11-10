import glob
import os
import sys
import pdb
import os.path as osp
import pickle
sys.path.append(os.getcwd())
import numpy as np
from torch import nn
from torch.nn import functional as F
from khrylib.models.mlp import MLP
from khrylib.models.rnn import RNN
from khrylib.utils.torch import *
from meva.lib.vae_recs import *
from meva.utils.config import Config




def get_vis_refiner(cfg, traj_dim = 144, iter = -1, vis_dim = 2048, vae_iter = -1):
    model_specs = cfg.model_specs
    data_specs = cfg.data_specs
    vae_cfg = Config(model_specs['vae_cfg'])

    vae, _, _ = get_models(vae_cfg, iter = vae_iter)
    refiner = RefinerV1(data_specs['traj_dim'],  data_specs['t_total'], model_specs)
    vae.eval() ## VAE should not be trained at this point
    concat_shceme = model_specs['concat_shceme']
    vis_refiner = Vis_Motion_Refiner(vae, refiner, concat_scheme = concat_shceme)

    return vis_refiner

def get_models(cfg, iter = -1):
    model_specs = cfg.model_specs
    data_specs =cfg.data_specs
    model_name = model_specs['model_name']

    traj_dim = data_specs['traj_dim']
    t_total = data_specs['t_total']

    run_epoch = None
    run_batch = None
    model = None

    if model_name == 'VAErec':
        model = VAErec(traj_dim, t_total, model_specs)
        run_epoch = run_epoch_vae_rec
        run_batch = run_batch_vae_rec
    elif model_name == 'VAEclfv1':
        model = VAEclfv1(traj_dim, data_specs['nc'], t_total,  model_specs)
        run_epoch = run_epoch_vae_cnd
        run_batch = run_batch_vae_cnd
    elif model_name == 'VAEclfv2':
        model = VAEclfv2(traj_dim, data_specs['nc'], t_total, model_specs)
        run_epoch = run_epoch_vae_cnd
        run_batch = run_batch_vae_cnd
    elif model_name == 'RefinerV1':
        model = get_vis_refiner(cfg, iter = iter, vae_iter=model_specs['vae_iter'])
        run_epoch = run_epoch_refiner
        run_batch = run_batch_refiner
    
    if iter > 0:
        cp_path = cfg.model_path % iter
        print("loading {} model from checkpoint: {}".format(model_name, cp_path))
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp["model_dict"])
    elif iter == -1:
        pass
    elif iter  == -2:
        cp_path = sorted(glob.glob(osp.join(cfg.model_dir, "*")), reverse= True)[0]
        print("loading {} model from checkpoint: {}".format(model_name, cp_path))
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp["model_dict"])

    
    return model, run_epoch, run_batch


def get_vae_model(cfg, traj_dim):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', 'VAEv1')
    print("loading model: {}".format(model_name) )
    
    if model_name == 'VAEv1':
        vae = VAE(traj_dim, traj_dim, cfg.nz, cfg.t_pred, specs)
    elif model_name == 'VAEv2':
        vae = VAEv2(traj_dim, traj_dim, cfg.nz, cfg.t_pred, specs)
    elif model_name == 'VAErec':
        vae = VAErec(traj_dim, cfg.nz, cfg.t_total, specs)
    elif model_name == 'VAEclfv1':
        vae = VAEclfv1(traj_dim, cfg.nc, cfg.nz, cfg.t_total, specs)
    elif model_name == 'VAEclfv2':
        vae = VAEclfv2(traj_dim, cfg.nc, cfg.nz, cfg.t_total, specs)

    return 



def get_dsf_model(cfg, traj_dim):
    specs = cfg.dsf_specs
    model_name = specs.get('model_name', 'DSFv1')
    if model_name == 'DSFv1':
        return DSF(traj_dim, cfg.nz, cfg.nc, cfg.dsf0_classes, specs)

def get_clf_model(cfg, traj_dim):
    clf_specs = cfg.clf_specs
    model_name = clf_specs.get('model_name', 'CLFv1')
    if model_name == 'CLFv1':
        return CLFv1(traj_dim, clf_specs)
    elif model_name == 'CLFv2':
        return CLFv2(traj_dim, cfg.nc, clf_specs)
    elif model_name == 'CLFvae':
        return CLFvae(traj_dim, cfg.nc, clf_specs, cfg)

def get_discriminator(cfg, traj_dim):
    specs = cfg.disc_specs
    model_name = specs.get('discriminator_name', 'DISCv1')
    if model_name == 'DISCv1':
        return Discriminator(traj_dim, specs)

def get_tcn_model(cfg):
    specs = cfg.tcn_specs
    model_name = specs.get("model_name", "TCNv1")
    if model_name == "TCNv1":
        return TCNv1(150, [144, 180, 256, 180, 150])
    elif model_name == "TCNv2":
        return TCNv2(24, 6, 24, [3,3,3])


if __name__ == '__main__':
    # model = VAEv2(63, 63, 128, 90, {})
    # X = torch.ones(30, 8, 63)
    # Y = torch.zeros(90, 8, 63)
    # out = model(X, Y)[0]
    # print(out.shape)
    model = VAErec(72, 512, 300, {})
    print(model)
    X = torch.ones(30, 8, 72)

    out = model(X)[0]
    print(out.shape)
    