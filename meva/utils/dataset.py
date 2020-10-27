import numpy as np


class Dataset:

    def __init__(self, mode, t_his, t_pred):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
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
        sample = {}
        t_data = next(iter(self.data.values()))
        seq_id = np.random.choice(list(t_data.keys()))
        seq_len = t_data[seq_id].shape[0]
        fr_start = np.random.randint(seq_len - self.t_total)
        fr_end = fr_start + self.t_total
        for key in self.data.keys():
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

    def iter_generator(self, step=30):
        t_data = next(iter(self.data.values()))
        for seq_id in t_data.keys():
            seq_len = t_data[seq_id].shape[0]
            for i in range(0, seq_len - self.t_total, step):
                sample = {}
                for key in self.data.keys():
                    sample[key] = self.data[key][seq_id][None, i: i + self.t_total]
                yield sample



