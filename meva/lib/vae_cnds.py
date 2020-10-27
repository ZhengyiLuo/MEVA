import os
import sys
sys.path.append(os.getcwd())

import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
from torch.nn import functional as F

import numpy as np
from torch import nn
from torch.nn import functional as F
from khrylib.models.mlp import MLP
from khrylib.models.rnn import RNN
from khrylib.utils.torch import *

class VAEclfv1(nn.Module):
    ## V1 has condition on both sides
    def __init__(self, nx, nc, horizon, specs):
        super(VAEclfv1, self).__init__()
        self.nx = nx
        self.nc = nc
        self.nz = nz = specs['nz']
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.e_birnn = e_birnn = specs.get('e_birnn', False)
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.nx_rnn = nx_rnn = specs.get('nx_rnn', 256)
        self.nlab_mlp = nlab_mlp = specs.get('nlab_mlp', [300, 200])
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.dropout = specs.get('dropout', 0.0)
        # encode
        self.e_x_rnn = RNN(nx, nx_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.e_lab_mlp = MLP(nc, nlab_mlp)
        self.e_mlp = MLP( nx_rnn + nlab_mlp[-1], nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        # decode
        self.d_lab_mlp = MLP(nc, nlab_mlp)
        self.d_rnn = RNN(self.d_lab_mlp.out_dim + nz + nx, nx_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nx_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, nx)
        self.d_rnn.set_mode('step')

    def encode_x(self, x):
        if self.e_birnn:
            h_x = self.e_x_rnn(x).mean(dim=0)
        else:
            h_x = self.e_x_rnn(x)[-1]
        return h_x
    
    def encode_z(self, x, lab):
        mu, logvar = self.encode(x, lab)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return z

    def encode(self, x, lab):
        e_x = self.encode_x(x)
        e_lab = self.e_lab_mlp(lab)
        e_feat = torch.cat([e_x, e_lab], dim=1)
        h = self.e_mlp(e_feat)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, lab):
        d_lab = self.d_lab_mlp(lab)
        self.d_rnn.initialize(batch_size=z.shape[0])
        x_rec = []
        x_p = x[0, :] # Feeding in the first frame of the input
        for i in range(self.horizon):
            rnn_in = torch.cat([x_p, z, d_lab], dim=1)
            
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            x_i = self.d_out(h)
            # if self.additive:
                # x_i[..., :-6] += y_p[..., :-6]
            x_rec.append(x_i)
            x_p = x_i
        x_rec = torch.stack(x_rec)
        
        return x_rec

    def forward(self, x, lab):
        mu, logvar = self.encode(x, lab)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(x, z, lab), mu, logvar

    def sample_prior(self, x, lab):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z, lab)

class VAEclfv2(nn.Module):
    ## V2 has condition on one side
    def __init__(self, nx, nc, horizon, specs):
        super(VAEclfv2, self).__init__()
        self.nx = nx
        self.nc = nc
        self.nz = nz = specs['nz']
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.e_birnn = e_birnn = specs.get('e_birnn', False)
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.nx_rnn = nx_rnn = specs.get('nx_rnn', 256)
        self.nlab_mlp = nlab_mlp = specs.get('nlab_mlp', [300, 200])
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.dropout = specs.get('dropout', 0.0)
        # encode
        self.e_x_rnn = RNN(nx, nx_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(nx_rnn , nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        # decode
        self.d_lab_mlp = MLP(nc, nlab_mlp)
        self.d_rnn = RNN(self.d_lab_mlp.out_dim + nz + nx, nx_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nx_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, nx)
        self.d_rnn.set_mode('step')

    def encode_x(self, x):
        if self.e_birnn:
            h_x = self.e_x_rnn(x).mean(dim=0)
        else:
            h_x = self.e_x_rnn(x)[-1]
        return h_x
    
    def encode_z(self, x, lab = None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return z

    def encode(self, x, lab = None):
        e_x = self.encode_x(x)
        h = self.e_mlp(e_x)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, lab):
        d_lab = self.d_lab_mlp(lab)
        self.d_rnn.initialize(batch_size=z.shape[0])
        x_rec = []
        x_p = x[0, :] # Feeding in the first frame of the input
        for i in range(self.horizon):
            rnn_in = torch.cat([x_p, z, d_lab], dim=1)
            
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            x_i = self.d_out(h)
            # if self.additive:
                # x_i[..., :-6] += y_p[..., :-6]
            x_rec.append(x_i)
            x_p = x_i
        x_rec = torch.stack(x_rec)
        
        return x_rec

    def forward(self, x, lab):
        mu, logvar = self.encode(x, lab)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(x, z, lab), mu, logvar

    def sample_prior(self, x, lab):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z, lab)

def run_epoch_vae_cnd(cfg, dataset, model, loss_func, optimizer, device, dtype = torch.float, scheduler = None, mode = "train", options = dict()):
    t_s = time.time()
    train_losses = 0
    total_num_batch = 0
    if mode == 'train':
        generator = dataset.sampling_generator(batch_size=cfg.batch_size, num_samples=50000)
    elif mode == 'mode':
        generator = dataset.iter_generator(batch_size=cfg.batch_size)
        
    pbar = tqdm(generator)
    for data in pbar:
        traj_x = data["traj"]
        label = data["label"]
        
        traj_x = torch.tensor(traj_x) if not torch.is_tensor(traj_x) else traj_x
        label = torch.tensor(label) if not torch.is_tensor(label) else label

        traj_x = traj_x.type(dtype).to(device).permute(1, 0, 2).contiguous()
        label = label.type(dtype).to(device)

        X_r, mu, logvar = model(traj_x, label)
        loss, losses = loss_func(cfg, X_r = X_r, X = traj_x, mu = mu, logvar =  logvar)

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        train_losses += losses
        total_num_batch += 1
        pbar.set_description("Runing Loss {:.3f}".format(np.mean(losses)))

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_batch

    return train_losses, dt

def run_batch_vae_cnd():
    pass


