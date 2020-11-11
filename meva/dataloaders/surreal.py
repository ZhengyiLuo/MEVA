# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

from meva.dataloaders import Dataset3D
from meva.utils.video_config import SURREAL_DIR

class Surreal(Dataset3D):
    def __init__(self, split, seqlen, overlap=0.5, debug=False):
        db_name = 'surreal'

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        split = "all"
        is_train = split == 'train' or split == 'all'
        overlap = overlap if is_train else 0.
        super(Surreal, self).__init__(
            split=split,
            folder=SURREAL_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )