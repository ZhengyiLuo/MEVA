import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())


if __name__ == "__main__":
    action = "person_abandons_package"
    video_files =  glob.glob(f"/hdd/zen/data/video_pose/meva/pip_175k_stabilized_0/videos/{action}/*.mp4")
    for video_file in video_files:
        cmd = f" python scripts/run_meva_on_video.py \
            --cfg train_meva_2 \
            --vid_file  {video_file}\
            --output_folder results/output/{action} \
            --exp train_meva_2"
        # print(cmd)
        os.system(cmd)