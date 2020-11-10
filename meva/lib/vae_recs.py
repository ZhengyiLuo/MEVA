import os
import sys
sys.path.append(os.getcwd())

import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
from torch.nn import functional as F

from meva.khrylib.models.mlp import MLP
from meva.khrylib.models.rnn import RNN
from meva.khrylib.utils.torch import *

class VAErec(nn.Module):
    def __init__(self, nx, t_total, specs):
        super(VAErec, self).__init__()
        self.nx = nx
        self.nz = nz =  specs['nz']
        self.t_total = t_total
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.e_birnn = e_birnn = specs.get('e_birnn', False)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', True)
        self.nx_rnn = nx_rnn = specs.get('nx_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.additive = specs.get('additive', False)
        # encode
        self.e_rnn = RNN(nx, nx_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(nx_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nx_rnn, nh_mlp + [nx_rnn], activation='tanh')
        self.d_rnn = RNN(nx + nx_rnn, nx_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nx_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, nx)
        self.d_rnn.set_mode('step')

    def encode_x(self, x):
        if self.e_birnn:
            h_x = self.e_rnn(x).mean(dim=0)
        else:
            h_x = self.e_rnn(x)[-1]
        return h_x

    def encode_x_all(self, x):
        h_x = self.encode_x(x)
        h = self.e_mlp(h_x)
        return h_x, self.e_mu(h), self.e_logvar(h)


    def encode(self, x):

        # self.e_rnn.initialize(batch_size=x.shape[0])
        h_x = self.encode_x(x)
        h = self.e_mlp(h_x)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        
        self.d_rnn.initialize(batch_size=z.shape[0])
        x_rec = []
        x_p = x[0, :] # Feeding in the first frame of the input
        
        for i in range(self.t_total):
            rnn_in = torch.cat([x_p, z], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            x_i = self.d_out(h)
            # if self.additive:
                # x_i[..., :-6] += y_p[..., :-6]
            x_rec.append(x_i)
            x_p = x_i
        x_rec = torch.stack(x_rec)
        
        return x_rec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(x, z), mu, logvar

    def sample_prior(self, x):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z)
    
    def step(self, model):
        pass


class VAErecV2(nn.Module):
    def __init__(self, nx, t_total, specs):
        super(VAErecV2, self).__init__()
        self.nx = nx
        self.nz = nz =  specs['nz']
        self.t_total = t_total
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.e_birnn = e_birnn = specs.get('e_birnn', False)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', True)
        self.nx_rnn = nx_rnn = specs.get('nx_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.additive = specs.get('additive', False)
        # encode
        self.e_rnn = RNN(nx, nx_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(nx_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nx_rnn, nh_mlp + [nx_rnn], activation='relu')
        self.d_rnn = RNN(nx + nx_rnn, nx_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nx_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, nx)
        self.d_rnn.set_mode('step')

        self.init_pose_mlp = MLP(nx_rnn, nh_mlp, activation='relu')
        self.init_pose_out = nn.Linear(self.init_pose_mlp.out_dim, nx)

    def encode_x(self, x):
        if self.e_birnn:
            h_x = self.e_rnn(x).mean(dim=0)
        else:
            h_x = self.e_rnn(x)[-1]
        return h_x

    # def encode_x_all(self, x):
    #     h_x = self.encode_x(x)
    #     h = self.e_mlp(h_x)
    #     return h_x, self.e_mu(h), self.e_logvar(h)


    def encode(self, x):
        # self.e_rnn.initialize(batch_size=x.shape[0])
        h_x = self.encode_x(x)
        h = self.e_mlp(h_x)
        return self.e_mu(h), self.e_logvar(h)
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def encode_hx(self, h_x):
    #     h_init_pose = self.init_pose_mlp(h_x)
    #     h_init_pose = self.init_pose_out(h_init_pose)
    #     h = self.e_mlp(h_x)
    #     return self.e_mu(h), self.e_logvar(h), h_init_pose


    # def decode_hx(self, h_x):
    #     mu, logvar, h_init_pose = self.encode_hx(h_x)
    #     z = mu
    #     return self.decode(h_init_pose[None, ], z), mu, logvar

    def decode(self, z, x_p = None):
        if x_p == None:
            h_init_pose = self.init_pose_mlp(z)
            x = self.init_pose_out(h_init_pose)
            x_p = x # Feeding in the first frame of the predicted input
        
        self.d_rnn.initialize(batch_size=z.shape[0])
        x_rec = []
        
        for i in range(self.t_total):
            rnn_in = torch.cat([x_p, z], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            x_i = self.d_out(h)
            # if self.additive:
                # x_i[..., :-6] += y_p[..., :-6]
            x_rec.append(x_i)
            x_p = x_i
        x_rec = torch.stack(x_rec)
        
        return x_rec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar

    def sample_prior(self, x):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(z)
    
    def step(self, model):
        pass



def run_epoch_vae_rec(cfg, dataset, model, loss_func, optimizer, device, dtype = torch.float, scheduler = None, mode = "train", options = dict()):
    t_s = time.time()
    train_losses = 0
    total_num_batch = 0
    if mode == 'train':
        generator = dataset.sampling_generator(batch_size=cfg.batch_size, num_samples = cfg.num_samples)
    elif mode == 'mode':
        generator = dataset.iter_generator(batch_size=cfg.batch_size)
    pbar = tqdm(generator)
    for data in pbar:
        traj_np = data["traj"]
        if torch.is_tensor(traj_np):
            traj_x = traj_np.type(dtype).to(device).permute(1, 0, 2).contiguous()
        else:
            traj_x = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        X_r, mu, logvar = model(traj_x)
        loss, losses = loss_func(cfg, X_r = X_r, X = traj_x, mu = mu, logvar = logvar)

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
    

def run_batch_vae_rec():
    pass


