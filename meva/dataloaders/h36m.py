# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

from lib.dataset import Dataset3D
from meva.utils.video_config import H36M_DIR


class H36M(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.5, debug=False):
        db_name = 'h36m'

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        # set = "all"
        is_train = set == 'train' or set == 'all'
        overlap = overlap if is_train else 0.
        print('H36M Dataset overlap ratio: ', overlap)
        super(H36M, self).__init__(
            set = set,
            folder=H36M_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')

if __name__ == "__main__":
    pass