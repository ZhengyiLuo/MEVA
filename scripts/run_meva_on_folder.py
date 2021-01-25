import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())


if __name__ == "__main__":
    video_base = "/hdd/zen/data/video_pose/meva/pip_175k_stabilized_0/videos/"
    
    for action in os.listdir(video_base):
        video_files =  glob.glob(f"{video_base}{action}/*.mp4")
        for video_file in video_files:
            cmd = f" python scripts/run_meva_on_video.py \
                --cfg train_meva_2 \
                --vid_file  {video_file}\
                --output_folder results/output/{action} \
                --exp train_meva_2 \
                --no_render"
            # print(cmd)
            try:
                os.system(cmd)
            except KeyboardInterrupt:
                exit()