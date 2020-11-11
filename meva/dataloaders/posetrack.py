# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

from meva.dataloaders import Dataset2D
from meva.utils.video_config import POSETRACK_DIR


class PoseTrack(Dataset2D):
    def __init__(self, seqlen, overlap=0.75, folder=None, debug=False):
        db_name = 'posetrack'
        super(PoseTrack, self).__init__(
            seqlen = seqlen,
            folder=POSETRACK_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
