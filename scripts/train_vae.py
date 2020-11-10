import os
import pdb
import sys
import math
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.getcwd())
from meva.khrylib.utils import *
from meva.utils.config import Config
from meva.dataloaders.data_loaders import *
from meva.lib.model import *
from meva.lib.loss_funcs import *

from meva.lib.smpl import SMPL_MODEL_DIR
from smplx import SMPL


def log_info(epoch):
    lr = optimizer.param_groups[0]["lr"]
    loss_names = cfg.loss_specs['loss_names']
    losses_str = " ".join(["{}: {:.4f}".format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info(
        "====> Epoch: {} Time: {:.2f} {} lr: {:.5f} exp: {}".format(i, dt, losses_str, lr, exp_name)
    )
    if mode == 'train':
        for name, loss in zip(loss_names, train_losses):
            tb_logger.add_scalar("model_" + name, loss, i)        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--dataset_root", type=str, default="data")
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float 
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    exp_name = args.cfg
    mode = args.mode
    cfg = Config(args.cfg)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == "train" else None
    logger = create_logger(os.path.join(cfg.log_dir, "log.txt"))
        
    """dataset"""
    dataset = get_dataset_cfg(cfg)

    """model"""
    model, run_epoch, _ = get_models(cfg, args.iter)

    """loss function"""
    loss_func = get_loss_func(cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = get_scheduler(
        optimizer,
        policy="lambda",
        nepoch_fix=cfg.num_epoch_fix,
        nepoch=cfg.num_epoch,
    )

    if mode == "train":
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_epoch):
            train_losses, dt = run_epoch(cfg, dataset, model, loss_func, optimizer, device, dtype = dtype, scheduler=scheduler, mode = mode)
            log_info(i)
            
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.model_path % (i + 1)
                    model_cp = {
                        "model_dict": model.state_dict(),
                        # "meta": {"std": dataset.std, "mean": dataset.mean},
                    }
                    pickle.dump(model_cp, open(cp_path, "wb"))

    elif mode == "eval":
        model.to(device)
        model.train()   
        with torch.no_grad():
            train_losses, dt = run_epoch(cfg, dataset, model, loss_func, optimizer, device, dtype = dtype, scheduler=scheduler, mode = mode)
            log_info(i)

