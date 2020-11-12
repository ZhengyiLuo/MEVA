# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import argparse
from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
VIBE_DB_DIR = '/hdd/zen/data/video_pose/vibe_db'
AMASS_DIR = '/hdd/zen/data/video_pose/amass'
INSTA_DIR = '/hdd/zen/data/video_pose/insta_variety'
MPII3D_DIR = '/hdd/zen/data/video_pose/mpi_inf_3dhp'
H36M_DIR = '/hdd/zen/data/video_pose/h36m/raw_data/'
THREEDPW_DIR = '/hdd/zen/data/video_pose/3dpw'
MOVI_DIR = '/hdd/zen/data/video_pose/movi'
PENNACTION_DIR = '/hdd/zen/data/video_pose/pennaction'
POSETRACK_DIR = '/hdd/zen/data/video_pose/posetrack'
SURREAL_DIR = '/hdd/zen/data/video_pose/surreal/SURREAL'
MEVA_DATA_DIR = 'data/meva_data'

# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = True
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 8
cfg.DEBUG_FREQ = 1000
cfg.SEED_VALUE = 1

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.VAE_CFG = "vae_rec_1"
cfg.TRAIN = CN()
cfg.TRAIN.DATASETS_2D = ['Insta']
cfg.TRAIN.DATASETS_3D = ['MPII3D']
cfg.TRAIN.DATASET_EVAL = 'ThreeDPW'
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.DATA_2D_RATIO = 0.5
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 5
cfg.TRAIN.PRETRAINED_REGRESSOR = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.PRETRAINED_VIBE = ''
cfg.TRAIN.RESUME = ''
cfg.TRAIN.NUM_ITERS_PER_EPOCH = 1000
cfg.TRAIN.LR_PATIENCE = 5

# <====== generator optimizer
cfg.TRAIN.GEN_OPTIM = 'Adam'
cfg.TRAIN.GEN_LR = 1e-4
cfg.TRAIN.GEN_WD = 1e-4
cfg.TRAIN.GEN_MOMENTUM = 0.9

# <====== motion discriminator optimizer
cfg.TRAIN.MOT_DISCR = CN()
cfg.TRAIN.MOT_DISCR.OPTIM = 'SGD'
cfg.TRAIN.MOT_DISCR.LR = 1e-2
cfg.TRAIN.MOT_DISCR.WD = 1e-4
cfg.TRAIN.MOT_DISCR.MOMENTUM = 0.9
cfg.TRAIN.MOT_DISCR.UPDATE_STEPS = 1
cfg.TRAIN.MOT_DISCR.FEATURE_POOL = 'concat'
cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE = 1024
cfg.TRAIN.MOT_DISCR.NUM_LAYERS = 1
cfg.TRAIN.MOT_DISCR.ATT = CN()
cfg.TRAIN.MOT_DISCR.ATT.SIZE = 1024
cfg.TRAIN.MOT_DISCR.ATT.LAYERS = 1
cfg.TRAIN.MOT_DISCR.ATT.DROPOUT = 0.1

cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 20
cfg.DATASET.OVERLAP = 0.5

cfg.LOSS = CN()
cfg.LOSS.KP_2D_W = 60.
cfg.LOSS.KP_3D_W = 30.
cfg.LOSS.SHAPE_W = 0.001
cfg.LOSS.POSE_W = 1.0
cfg.LOSS.D_MOTION_LOSS_W = 1.

cfg.MODEL = CN()

cfg.MODEL.TEMPORAL_TYPE = 'gru'

# GRU model hyperparams
cfg.MODEL.TGRU = CN()
cfg.MODEL.TGRU.NUM_LAYERS = 1
cfg.MODEL.TGRU.ADD_LINEAR = False
cfg.MODEL.TGRU.RESIDUAL = False
cfg.MODEL.TGRU.HIDDEN_SIZE = 2048
cfg.MODEL.TGRU.BIDIRECTIONAL = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = osp.join("meva/cfg", f"{args.cfg}.yml")
    print(f"loading from {cfg_file}")
    if args.cfg is not None:
        cfg = update_cfg(cfg_file)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file
