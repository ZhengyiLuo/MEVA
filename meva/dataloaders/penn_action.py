# This script is borrowed from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script

from meva.dataloaders import Dataset2D
from meva.utils.video_config import PENNACTION_DIR


class PennAction(Dataset2D):
    def __init__(self, seqlen, overlap=0.75, debug=False):
        db_name = 'pennaction'

        super(PennAction, self).__init__(
            seqlen = seqlen,
            folder=PENNACTION_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
