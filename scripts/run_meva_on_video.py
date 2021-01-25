import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader


from meva.lib.meva_model import MEVA, MEVA_demo
from meva.utils.renderer import Renderer
from meva.utils.kp_utils import convert_kps
from meva.dataloaders.inference import Inference
from meva.utils.video_config import parse_args, update_cfg
from meva.utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

def main(args):
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    video_file = args.vid_file

    # ========= [Optional] download the youtube video ========= #
    if video_file.startswith('https://www.youtube.com'):
        print(f'Donwloading YouTube video \"{video_file}\"')
        video_file = download_youtube_clip(video_file, '/tmp')

        if video_file is None:
            exit('Youtube url is not valid!')

        print(f'YouTube Video has been downloaded to {video_file}...')

    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')

    filename = os.path.splitext(os.path.basename(video_file))[0]
    output_path = os.path.join(args.output_folder, filename)
    os.makedirs(output_path, exist_ok=True)
 
    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]
    

    # ========= MEVA Model ========= #
    pretrained_file = f"results/meva/{args.exp}/model_best.pth.tar"

    config_file = osp.join("meva/cfg", f"{args.cfg}.yml")
    cfg = update_cfg(config_file)
    model = MEVA_demo(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
        cfg = cfg.VAE_CFG,
    ).to(device)

    
    ckpt = torch.load(pretrained_file)
    # print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    # ========= MEVA Model ========= #

    
    # ========= Run MEVA on each person ========= #
    bbox_scale = 1.2
    print(f'Running MEVA on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']
        if len(frames) < 90:
            print(f"!!!tracklet < 90 frames: {len(frames)} frames")
            continue

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames

        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16, shuffle = False)

        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []
            data_chunks = dataset.iter_data() 

            for idx in range(len(data_chunks)):
                batch = data_chunks[idx]
                batch_image = batch['batch'].unsqueeze(0)
                cl = batch['cl']
                batch_image = batch_image.to(device)

                batch_size, seqlen = batch_image.shape[:2]
                output = model(batch_image)[-1]

                pred_cam.append(output['theta'][0, cl[0]: cl[1], :3])
                pred_verts.append(output['verts'][0, cl[0]: cl[1]])
                pred_pose.append(output['theta'][0,cl[0]: cl[1],3:75])
                pred_betas.append(output['theta'][0, cl[0]: cl[1],75:])
                pred_joints3d.append(output['kp_3d'][0, cl[0]: cl[1]])


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch_image


        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    del model


    end = time.time()
    fps = num_frames / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    print(f'Saving output results to \"{os.path.join(output_path, "meva_output.pkl")}\".')

    joblib.dump(vibe_results, os.path.join(output_path, "meva_output.pkl"))

    # vibe_results = joblib.load(os.path.join(output_path, "meva_output.pkl"))

    if not args.no_render or not len(vibe_results) == 0:
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

        output_img_folder = f'{image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)
            # img = np.zeros(img.shape)

            if args.sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mc = mesh_color[person_id]

                mesh_filename = None

                if args.save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )
                
                frame_cam = np.array([ 0.5,  1., 0,  0])
                if args.sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        mesh_filename=mesh_filename,
                        # angle=270,
                        # axis=[0,1,0],
                    )

            if args.sideview:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        vid_name = os.path.basename(video_file)
        save_name = f'{vid_name.replace(".mp4", "")}_meva_result.mp4'
        save_name = os.path.join(output_path, save_name)
        print(f'Saving result video to {save_name}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)

    shutil.rmtree(image_folder)
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, help='input video path')
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--exp', type=str, help='experience name')
    parser.add_argument('--output_folder', type=str, help='output folder to write results')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'], help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416, help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12, help='batch size of object detector used for bbox tracking')

    parser.add_argument('--vibe_batch_size', type=int, default=90, help='batch size of VIBE')

    parser.add_argument('--display', action='store_true', help='visualize the results of each step during demo')

    parser.add_argument('--no_render', action='store_true', help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_false', default=True, 
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='process the video list with the specified GPU id')

    args = parser.parse_args()

    main(args)
