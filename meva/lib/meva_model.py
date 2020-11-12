import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from meva.utils.video_config import MEVA_DATA_DIR
from meva.lib.spin import Regressor, hmr
from meva.lib.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from meva.utils.transform_utils import convert_orth_6d_to_mat, convert_orth_6d_to_aa, convert_mat_to_6d, rotation_matrix_to_angle_axis

from meva.utils.config import Config
from meva.lib.model import *

def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            output_size = 2048,
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, output_size)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, output_size)
        self.use_residual = use_residual
        self.output_size = output_size

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = torch.tanh(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n, self.output_size)

        if self.use_residual and y.shape[-1] == self.output_size:
            y = y + x

        y = y.permute(1,0,2) # TNF -> NTF
        return y


class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(Regressor, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
    
    def set_gender(self, gender="neutral", use_smplx = False):
        if use_smplx:
            from smplx import SMPL
            self.smpl = SMPL(
                SMPL_MODEL_DIR,
                batch_size=64,
                create_transl=False, 
                gender = gender
            ).to(next(self.smpl.parameters()).device)
        else:
            from meva.lib.smpl import SMPL
            self.smpl = SMPL(
                SMPL_MODEL_DIR,
                batch_size=64,
                create_transl=False, 
                gender = gender
            ).to(next(self.smpl.parameters()).device)


    def iter_refine(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        
        return pred_pose, pred_shape, pred_cam
        

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None):
        batch_size = x.shape[0]
        pred_pose, pred_shape, pred_cam =  self.iter_refine(x, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, n_iter=n_iter, J_regressor=J_regressor)
        pred_rotmat = convert_orth_6d_to_mat(pred_pose).view(batch_size , 24, 3, 3)

        ############### SMOOTH ###############
        # from meva.utils.geometry import smooth_pose_mat
        # pred_rotmat = torch.tensor(smooth_pose_mat(pred_rotmat.cpu().numpy(), ratio = 0.3)).float().to(pred_rotmat.device)
        ############### SMOOTH ###############

        return self.smpl_to_kpts(pred_rotmat, pred_shape, pred_cam, J_regressor)


    def smpl_to_kpts(self, pred_rotmat, pred_shape, pred_cam, J_regressor):
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output


class MEVA(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            cfg = "vae_rec_1"
    ):
        super(MEVA, self).__init__()

        self.vae_cfg = vae_cfg = Config(cfg)
        self.seqlen = seqlen
        self.batch_size = batch_size
        
        self.vae_model, _, _ = get_models(vae_cfg, iter = -2)
        for param in self.vae_model.parameters():
            param.requires_grad = False

        self.feat_encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        vae_hidden_size = 512
        self.motion_encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=512,
            bidirectional=bidirectional,
            add_linear=True,
            output_size = vae_hidden_size, 
            use_residual=False,
        )
        
        # if self.vae_cfg.model_specs['model_name'] == "VAErec":
        fc1 = nn.Linear(vae_hidden_size, 256)
        act = nn.Tanh()
        fc2 = nn.Linear(256, 144)
        self.vae_init_mlp = nn.Sequential(fc1, act, fc2)

        self.regressor = Regressor()
        mean_params = np.load(SMPL_MEAN_PARAMS)
        
        self.first_in_flag = True

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )
        self.set_gender()

    def set_gender(self, gender="neutral", use_smplx = False):
        self.regressor.set_gender(gender, use_smplx)
    
    def forward(self, input, J_regressor=None):
        
        
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        feature = self.feat_encoder(input)
        # feature = input

        motion_z = self.motion_encoder(feature).mean(dim = 1)

        if self.vae_cfg.model_specs['model_name'] == "VAErec":
            # smpl_output = self.regressor(feature[:, 0, :], J_regressor=J_regressor)
            # vae_init_pose = convert_mat_to_6d(smpl_output[0]['rotmat']).reshape(batch_size, 144)
            vae_init_pose = self.vae_init_mlp(motion_z)
            X_r = self.vae_model.decode(vae_init_pose[None, :, :], motion_z)
        elif self.vae_cfg.model_specs['model_name'] == "VAErecV2":
            X_r = self.vae_model.decode(motion_z)

        X_r = X_r.permute(1, 0, 2)[:,:seqlen,:]
        

        feature = feature.reshape(-1, feature.size(-1))
        init_pose = X_r.reshape(-1, X_r.shape[-1])

        ## Official
        smpl_output = self.regressor(feature, J_regressor=J_regressor, init_pose = init_pose)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output


class MEVA_demo(MEVA):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=1048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(MEVA_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
            cfg = "vae_rec_1"
    ):
        super().__init__(seqlen, batch_size, n_layers, hidden_size, add_linear, \
            bidirectional, use_residual, cfg)

        self.pretrained = pretrained
        
    def forward(self, input, J_regressor=None):
        self.hmr_model = hmr()
        checkpoint = torch.load(self.pretrained)
        self.hmr_model.load_state_dict(checkpoint['model'], strict=False)
        self.hmr_model.to(input.device)
        self.hmr_model.eval()

        batch_size, seqlen, nc, h, w = input.shape
        feature = self.hmr_model.feature_extractor(input.reshape(-1, nc, h, w))
        feature = feature.reshape(batch_size, seqlen, -1)

        return super().forward(feature, J_regressor = J_regressor)
    
    def extract_feature(self, input):
        self.hmr_model = hmr()
        checkpoint = torch.load(self.pretrained)
        self.hmr_model.load_state_dict(checkpoint['model'], strict=False)
        self.hmr_model.to(input.device)
        self.hmr_model.eval()

        batch_size, seqlen, nc, h, w = input.shape
        feature = self.hmr_model.feature_extractor(input.reshape(-1, nc, h, w))
        feature = feature.reshape(batch_size, seqlen, -1)
        return feature
    



if __name__ == "__main__":
    from kinematic_synthesis.utils.config import Config
    from kinematic_synthesis.lib.model import *
    from meva.dataloaders import *
    from torch.utils.data import DataLoader
    
    meva_model = MEVA(90)
    db = PennAction(seqlen=90, debug=False)

    test_loader = DataLoader(
        dataset=db,
        batch_size=32,
        shuffle=False,
        num_workers=1,
    )

    for i in test_loader:
        kp_2d = i['kp_2d']
        input = i['features']
        meva_model(input)

