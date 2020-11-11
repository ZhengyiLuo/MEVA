# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

from meva.dataloaders import Dataset3D
from meva.utils.video_config import H36M_DIR


class H36M(Dataset3D):
    def __init__(self, split, seqlen, overlap=0.5, debug=False):
        db_name = 'h36m'

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        # set = "all"
        is_train = split == 'train' or split == 'all'
        overlap = overlap if is_train else 0.
        # print('H36M Dataset overlap ratio: ', overlap)
        super(H36M, self).__init__(
            split = split,
            folder=H36M_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        

if __name__ == "__main__":
    pass