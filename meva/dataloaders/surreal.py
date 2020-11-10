# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

from lib.dataset import Dataset3D
from meva.utils.video_config import SURREAL_DIR

class Surreal(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.5, debug=False):
        db_name = 'surreal'

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        set = "all"
        is_train = set == 'train' or set == 'all'
        overlap = overlap if is_train else 0.
        print(f'{db_name} Dataset overlap ratio: ', overlap)
        super(Surreal, self).__init__(
            set=set,
            folder=SURREAL_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')