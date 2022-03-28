import numpy as np


class bballDataSet(object):
    def __init__(self, data_path, n_players=10):
        # setting for court
        self.NORMALIZATION_COEF = 7
        self.PLAYER_CIRCLE_SIZE = 12 / self.NORMALIZATION_COEF
        self.BALL_CIRCLE_SIZE = 12 / self.NORMALIZATION_COEF
        self.INTERVAL = 33
        self.DIFF = 6
        self.X_MIN = 0
        self.X_MAX = 100
        self.Y_MIN = 0
        self.Y_MAX = 50
        self.NROM_LEN = 50
        self.COL_WIDTH = 0.3
        self.SCALE = 1.65
        self.FONTSIZE = 6
        self.X_CENTER = self.X_MAX / 2 - self.DIFF / 1.5 + 0.10
        self.Y_CENTER = self.Y_MAX - self.DIFF / 1.5 - 0.35
        self.DELTA_T = 0.16

        self.n_players = n_players

        self.locs = np.load(data_path)

        self.preprocess()

    def __len__(self):
        return self.locs.shape[0]

    def preprocess(self):
        self.nsamples, self.timestamp = self.locs.shape[:2]
        self.locs = self.locs.reshape(self.nsamples, self.timestamp, -1, 2)

        # choose offensive players or all players
        self.locs = self.locs[:, :, :self.n_players+1, :]

        # calculate the velocity
        # now self.locs.shape = (nsamples, timestamp, nplayers, 2)

        self.vels = np.zeros_like(self.locs)
        self.vels[:, :-1, :, :] = (self.locs[:, 1:, :, :] -
                                   self.locs[:, :-1, :, :]) / self.DELTA_T
        self.vels[:, -1, :, :] = self.vels[:, -2, :, :]


class bballDataLoader(bballDataSet):
    def __init__(self, data_path, train_len=40):
        super().__init__(data_path)
        # data_npz = np.load(data_path)
        # self.timestamps = data_npz['timestamps']
        self.n_batches = self.locs.shape[0]
        self.train_len = train_len
        self.start = 0
        self.shuffle = False
        self.indexs = np.array(range(0, self.n_batches))

    def __getitem__(self, idx):
        past = (self.locs[idx, :self.train_len],
                self.vels[idx, :self.train_len])
        future = (self.locs[idx, self.train_len:],
                  self.vels[idx, self.train_len:])
        return past, future

    def _get_batch(self, batch_size):
        indexs = np.random.choice(
            range(self.n_batches), size=batch_size, replace=False)
        return self[indexs]

    def __iter__(self):
        return self

    def __next__(self):
        # (batch_size, time_steps, num_agents, 4)
        if self.start < self.n_batches:
            idxs = self.indexs[self.start: self.start + self.batch_size]
            self.start += self.batch_size
            return self[idxs]
        else:
            raise StopIteration()

    def __call__(self, batch_size=100, shuffle=True):
        self.batch_size = batch_size
        self.indexs = np.array(range(0, self.n_batches))
        if shuffle:
            np.random.shuffle(self.indexs)
        self.start = 0
        self.shuffle = shuffle

        return self


class ChargedLoader(object):
    def __init__(self, data_path, train_len) -> None:
        super().__init__()
        data_npz = np.load(data_path)
        self.locs = data_npz['loc']
        self.vels = data_npz['vel']
        self.n_batches = self.locs.shape[0]
        self.n_steps = self.locs.shape[1]
        self.n_balls = self.locs.shape[-1]
        self.train_len = train_len
        self.delta_T = 0.001
        self.start = 0
        self.batch_size = None
        self.shuffle = True
        self.indexs = None

    def __getitem__(self, idx):
        past = (self.locs[idx, :self.train_len],
                self.vels[idx, :self.train_len])
        future = (self.locs[idx, self.train_len:],
                  self.vels[idx, self.train_len:])
        return past, future

    def _get_batch(self, batch_size):
        indexs = np.random.choice(
            range(self.n_batches), size=batch_size, replace=False)
        return self[indexs]

    def __iter__(self):
        return self

    def __len__(self):
        return self.locs.shape[0]

    def __next__(self):
        # (batch_size, time_steps, num_agents, 4)
        if self.start < self.n_batches:
            idxs = self.indexs[self.start: self.start + self.batch_size]
            self.start += self.batch_size
            return self[idxs]
        else:
            raise StopIteration()

    def __call__(self, batch_size=100, shuffle=True):
        self.batch_size = batch_size
        self.indexs = np.array(range(0, self.n_batches))
        if shuffle:
            np.random.shuffle(self.indexs)
        self.start = 0
        self.shuffle = shuffle
        return self
