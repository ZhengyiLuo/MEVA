from meva.khrylib.utils import *
import re
from skimage.util.shape import view_as_windows

def normalize_sentence(sent):
    sent = sent.replace('-', ' ')
    return re.sub(r'[^\w\s]', '', sent).lower()


def batch_get_traj(traj_arr, dt=1.0/30.0, init_pos=None, init_heading=None):
    traj_int = []
    for traj in traj_arr:
        traj_int.append(get_traj_from_state_pred(traj, dt, init_pos, init_heading))
    traj_int = np.stack(traj_int)
    return traj_int


def get_chunk_selects(chunk_idxes, last_chunk, window_size = 80, overlap = 10):
    shift = window_size - int(overlap/2)
    chunck_selects = []
    for i in range(len(chunk_idxes)):
        chunk_idx = chunk_idxes[i]
        if i == 0:
            chunck_selects.append((0, shift))
        elif i == len(chunk_idxes) - 1:
            chunck_selects.append((-last_chunk, window_size))
        else:
            chunck_selects.append((int(overlap/2), shift))
    return chunck_selects 

def get_chunk_with_overlap(num_frames, window_size = 80, overlap = 10):
    assert overlap % 2 == 0
    step = window_size - overlap 
    chunk_idxes = view_as_windows(np.array(range(num_frames)), window_size, step= step)
    chunk_supp = np.linspace(num_frames - window_size, num_frames-1, num = window_size).astype(int)
    chunk_idxes = np.concatenate((chunk_idxes, chunk_supp[None, ]))
    last_chunk = chunk_idxes[-1][:step][-1] - chunk_idxes[-2][:step][-1] + int(overlap/2)
    chunck_selects = get_chunk_selects(chunk_idxes, last_chunk, window_size= window_size, overlap=overlap)
    
    return chunk_idxes, chunck_selects