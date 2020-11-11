# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

from meva.dataloaders import Dataset3D
from meva.utils.video_config import MOVI_DIR

class MoVi(Dataset3D):
    def __init__(self, split, seqlen, overlap=0.5, debug=False):
        db_name = 'movi'

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        is_train = split == 'train' or split == 'all'
        overlap = overlap if is_train else 0.
        super(MoVi, self).__init__(
            split=split,
            folder=MOVI_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )