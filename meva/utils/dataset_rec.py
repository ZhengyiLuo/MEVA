import numpy as np


class DatasetRec:

    def __init__(self, mode, rec_length):
        self.mode = mode
        self.t_total = rec_length
        self.prepare_data()
        self.std, self.mean = None, None
        self.normalized = False

    def prepare_data(self):
        raise NotImplementedError

    def normalize_data(self, mean=None, std=None):
        raise NotImplementedError

    def __len__(self):
        t_data = next(iter(self.data.values()))
        return sum([d.shape[0] for d in t_data.values()])

    def sample(self):
        sample = {key: [] for key in self.data.keys()}
        t_data = next(iter(self.data.values()))
        seq_id = np.random.choice(list(t_data.keys()))
        seq_len = t_data[seq_id].shape[0]
        fr_start = np.random.randint(seq_len - self.t_total) if seq_len - self.t_total != 0 else 0
        fr_end = fr_start + self.t_total
        for key in self.data.keys():
            # import pdb
            # pdb.set_trace()
            if key == "label":
                sample[key] = self.data[key][seq_id]
            elif key == "entry_name":
                sample[key] = np.array([self.data[key][seq_id]])[None,:]
            else:
                sample[key] = self.data[key][seq_id][None, fr_start: fr_end]
        return sample

    def sampling_generator(self, num_samples=1000, batch_size=8):
        for i in range(num_samples // batch_size):
            sample = {key: [] for key in self.data.keys()}
            for i in range(batch_size):
                sample_i = self.sample()
                for key in self.data.keys():
                    sample[key].append(sample_i[key])
            for key in self.data.keys():
                sample[key] = np.concatenate(sample[key], axis=0)
            yield sample

    def iter_generator(self, batch_size = 8):
        t_data = next(iter(self.data.values()))
        seq_keys = list(t_data.keys())
        
        for i in range(self.seq_len // batch_size):
            sample = {key: [] for key in self.data.keys()}
            for j in range(batch_size):
                idx = i * batch_size + j
                if idx >= self.seq_len:
                    break
                seq_key = seq_keys[idx]
                for key in self.data.keys():
                    if key == "label" :
                        sample[key].append(self.data[key][seq_key])
                    elif key == "entry_name":
                        sample[key].append(np.array([self.data[key][seq_key]])[None,:])
                    else:
                        sample[key].append(self.data[key][seq_key][None, :self.t_total])

            for key in self.data.keys():
                sample[key] = np.concatenate(sample[key], axis = 0)
                
            yield sample

