import os
import numpy as np
import json
import glob
import sys
import pickle
import joblib
import subprocess
sys.path.append(os.getcwd())

from collections import defaultdict
from utils.pose import interpolated_traj_new
from h36m.utils.h36m_global import *


def save_pose_data(subject_id):
    print(f'saving pose data for subject {subject_id}')
    subject_data = {'action_seq': [], 'cam': []}
    video_dir = f'{h36m_folder}/S{subject_id}/Videos'
    smpl_file = f'{h36m_folder}/smpl_param/Human36M_subject{subject_id}_smpl_param.json'
    cam_file = f'{h36m_folder}/annotations/Human36M_subject{subject_id}_camera.json'
    joint_file = f'{h36m_folder}/annotations/Human36M_subject{subject_id}_joint_3d.json'

    cam_file = f'{h36m_folder}/annotations/Human36M_subject{subject_id}_camera.json'
    with open(cam_file) as f:
        cam_dict = json.load(f)
    for cam_id in range(1, 5):
        cam = cam_dict[str(cam_id)]
        for key in cam.keys():
            cam[key] = np.array(cam[key])
        subject_data['cam'].append(cam)

    with open(smpl_file) as f:
        smpl_param = json.load(f)

    with open(joint_file) as f:
        joint = json.load(f)

    video_files = sorted(glob.glob(f'{video_dir}/[!_]*'))
    assert len(video_files) == 120
    action_names = [os.path.basename(x).split('.')[0] for x in video_files[::4]]
    action_names = get_ordered_action_names(action_names, subject_id)

    for action_id in range(2, 17):
        for sub_action in range(1, 3):
            seq = defaultdict(list)
            seq['action_name'] = action_names[2 * (action_id - 2) + (sub_action - 1)]
            print(seq['action_name'])
            # get gt
            gt_jpos_dict = joint[str(action_id)][str(sub_action)]
            for i, key in enumerate(gt_jpos_dict.keys()):
                assert str(i) == key
                if i % 2 == 0:
                    seq['gt_jpos'].append(np.array(gt_jpos_dict[key]) * 0.001)

            smpl_param_dict = smpl_param[str(action_id)][str(sub_action)]
            frames = np.array(sorted([int(x) for x in smpl_param_dict.keys()]))
            diff = np.diff(frames)
            ind = np.where(diff != 5)[0]
            print(f'irregular frames:', [(x, y) for x, y in zip(ind, diff[ind])])
            for i in sorted([int(x) for x in smpl_param_dict.keys()]):
                key = str(i)
                seq['fitted_jpos'].append(np.array(smpl_param_dict[key]['fitted_3d_pose']) * 0.001)
                seq['trans'].append(np.array(smpl_param_dict[key]['trans'][0]))
                seq['pose'].append(np.array(smpl_param_dict[key]['pose']))
                seq['shape'].append(np.array(smpl_param_dict[key]['shape']))
            
            for key in ['gt_jpos', 'fitted_jpos', 'trans', 'pose', 'shape']:
                seq[key] = np.stack(seq[key])

            for key in ['fitted_jpos', 'trans', 'pose', 'shape']:
                seq[key]  = interpolated_traj_new(seq[key], frames * 0.02, np.arange(seq['gt_jpos'].shape[0]) * 0.04, 'expmap' if key == 'pose' else 'lin')

            seq['len'] = seq['gt_jpos'].shape[0]
            subject_data['action_seq'].append(seq)

    subject_data['len'] = sum([seq['len'] for seq in subject_data['action_seq']])
    subject_data['mean_shape'] = np.concatenate([seq['shape'] for seq in subject_data['action_seq']]).mean(axis=0)
    
    out_file = f'{h36m_out_folder}/poses/S{subject_id}.pkl'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    pickle.dump(subject_data, open(out_file, 'wb'))


def save_bbox_data(subject_id):
    print(f'saving bbox data for subject {subject_id}')
    bbox_data = defaultdict(list)
    last_frame_id = defaultdict(lambda: -2)
    video_dir = f'{h36m_folder}/S{subject_id}/Videos'
    data_file = f'{h36m_folder}/annotations/Human36M_subject{subject_id}_data.json'

    video_files = sorted(glob.glob(f'{video_dir}/[!_]*'))
    assert len(video_files) == 120
    action_names = [os.path.basename(x).split('.')[0] for x in video_files[::4]]
    action_names = get_ordered_action_names(action_names, subject_id)

    with open(data_file) as f:
        raw_data = json.load(f)
        images_dict = {x['id']: x for x in raw_data['images']}
        annotations_dict = raw_data['annotations']

    for annotation in annotations_dict:
        bbox = annotation['bbox']
        info = images_dict[annotation['image_id']]
        frame_id = info['frame_idx']
        if frame_id % 2 == 1:
            continue
        cam_id = info['cam_idx'] - 1
        action_name = action_names[2 * (info['action_idx'] - 2) + (info['subaction_idx'] - 1)]
        seq_name = f'{action_name}_cam{cam_id}'
        assert frame_id - last_frame_id[seq_name] == 2
        last_frame_id[seq_name] = frame_id
        bbox_data[seq_name].append(bbox)

    for key, bboxes in bbox_data.items():
        frame_dir = f'{h36m_out_folder}/frames/S{subject_id}/{key}'
        num_frames = len(glob.glob(f'{frame_dir}/*.png'))
        bbox_np = np.array(bboxes)
        if num_frames - bbox_np.shape[0] > 5:
            print(key, num_frames, bbox_np.shape[0])
        assert bbox_np.shape[0] <= num_frames
        out_file = f'{h36m_out_folder}/bbox/S{subject_id}/{key}.npy'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        np.save(out_file, bbox_np)


def save_video_data(subject_id, export_video=False, export_frame=False, export_vibe=False):
    print(f'saving video data for subject {subject_id}')
    video_dir = f'{h36m_folder}/S{subject_id}/Videos'
    video_files = sorted(glob.glob(f'{video_dir}/[!_]*'))
    assert len(video_files) == 120
    video_names = [os.path.splitext(os.path.basename(x))[0] for x in video_files]

    for video_file, video_name in zip(video_files, video_names):
        action_name, cam_name = tuple(video_name.split('.')[:2])
        cam_id = cam_mapping[cam_name]
        # export frames
        frame_dir = f'{h36m_out_folder}/frames/S{subject_id}/{action_name}_cam{cam_id}'
        out_file = f'{h36m_out_folder}/videos/S{subject_id}/{action_name}_cam{cam_id}.mp4'
        if export_frame:
            os.makedirs(frame_dir, exist_ok=True)
            cmd = ['ffmpeg', '-i', video_file, '-r', '25', '-start_number', '0', f'{frame_dir}/%04d.png']
            subprocess.call(cmd)
        # export frames
        if export_video:
            if not (args.skip and os.path.exists(out_file)):
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                cmd = ['ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0', '-i',
                    f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '8', '-pix_fmt', 'yuv420p', out_file]
                subprocess.call(cmd)
        # export vibe
        if export_vibe:
            print('calling VIBE')
            out_dir = f'{h36m_out_folder}/{args.vibe_folder}/S{subject_id}'
            if not (args.skip and os.path.exists(f'{out_dir}/{action_name}_cam{cam_id}/meva_output.pkl')):
                os.makedirs(out_dir, exist_ok=True)
                cmd = ['/mnt/home/yeyuan/anaconda3/envs/vibe-env/bin/python', '/mnt/home/yeyuan/repo/VIBE/demo.py',
                    '--vid_file', out_file, '--output_folder', out_dir, '--no_render', '--bbox_scale', str(args.bbox_scale), '--shift', str(args.shift)]
                subprocess.call(cmd, cwd=r'/mnt/home/yeyuan/repo/VIBE')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', default="1,5,6,7,8,9,11")
    parser.add_argument('--pose', action='store_true', default=False)
    parser.add_argument('--video', action='store_true', default=False)
    parser.add_argument('--frame', action='store_true', default=False)
    parser.add_argument('--bbox', action='store_true', default=False)
    parser.add_argument('--vibe', action='store_true', default=False)
    parser.add_argument('--vibe_folder', default=None)
    parser.add_argument('--shift', type=int, default=2)
    parser.add_argument('--bbox_scale', type=float, default=0.9)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--skip', action='store_true', default=False)
    args = parser.parse_args()

    subjects = [int(x) for x in args.subjects.split(',')]

    if args.pose:
        for sub in subjects:
            save_pose_data(sub)

    if args.video or args.frame or args.vibe:
        for sub in subjects:
            save_video_data(sub, args.video, args.frame, args.vibe)

    if args.bbox:
        for sub in subjects:
            save_bbox_data(sub)