import numpy as np
from numpy.random import Generator, PCG64


class ChargedParticlesSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0., seed=None, charge_prob=[1. / 2, 0, 1. / 2], one_uncharged=False):

        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._charge_types = np.array([-1., 0, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T
        self._rg = Generator(PCG64(seed))
        self.charge_prob = charge_prob
        self.charges = self._rg.choice(self._charge_types,
                                       size=(self.n_balls, 1),
                                       p=self.charge_prob)
        if one_uncharged:
            self.charges[np.random.randint(low=0, high=self.n_balls), 0] = 0
        self.loc_init = self._rg.standard_normal(
            size=(self.n_balls, 2)) * self.loc_std
        self.vel_init = self._rg.standard_normal(size=(self.n_balls, 2))

    def _l2(self, A, B):
        """
        Input: A is a N x d matrix
               B is a M x d matrix
        Output: dist is a N x M matrix where dist[i,j] is the square norm
                between A[i, :] and B[j, :]
        i.e. dist[i, j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * \
                            edges[i, j] / dist
        return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, T=100, sample_freq=200):
        # Output: (batch_size, time_steps , 2, num_balls)

        # graph: (0: neutral, 1: attract, 2: exclude)
        n = self.n_balls
        # assert (T % sample_freq == 0)
        # T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        # Sample edges

        # edges = charges.dot(charges.transpose())

        # Initialize location and velocity
        loc_next, vel_next = self.loc_init.copy(), self.vel_init.copy()
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        # Write initial loc and velocity
        counter = 0
        charges = self.charges
        edges = charges.dot(charges.transpose())

        # loc[0, :, :],  vel[0, :, :] = self._clamp(loc_next, vel_next)
        loc_list = []
        vel_list = []
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrop
            l2_dist_power3 = np.power(
                self._l2(loc_next, loc_next), 3. / 2.
            )
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # set self-forces are zero
            # print(forces_size)
            #assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            # non-diag values must larger than 1e-10
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[:, 0],
                                       loc_next[:, 0]).reshape(1, n, n),
                     np.subtract.outer(loc_next[:, 1],
                                       loc_next[:, 1]).reshape(1, n, n)
                 ))).sum(axis=-1).T
            F = F.clip(-self._max_F, self._max_F)
            vel_next += self._delta_T * F
            # run leapfrop
            # for i in range(1, T):
            i = 0
            while True:
                i += 1
                # update location
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)
                if i % sample_freq == 0:
                    # loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    # print(loc_next[0,0])
                    if counter == T:
                        break
                    loc_list.append(loc_next.copy())
                    vel_list.append(vel_next.copy())
                    counter += 1
                l2_dist_power3 = np.power(
                    self._l2(loc_next, loc_next),
                    3. / 2.
                )
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[:, 0],
                                           loc_next[:, 0]).reshape(1, n, n),
                         np.subtract.outer(loc_next[:, 1],
                                           loc_next[:, 1]).reshape(1, n, n)
                     ))).sum(axis=-1).T
                F = F.clip(-self._max_F, self._max_F)
                vel_next += self._delta_T * F
        loc = np.stack(loc_list, axis=0)
        vel = np.stack(vel_list, axis=0)
        return loc, vel, charges


if __name__ == '__main__':
    from multiprocessing import Pool
    import numpy as np
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sample", default=1000, type=int)
    parser.add_argument("--filename", default="train.npz", type=Path)
    parser.add_argument("--sample_freq", default=200, type=int)
    parser.add_argument("--seq_len", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    np.random.seed(args.seed)  # fixed seed for reproducability
    seeds = np.random.choice(range(100000), size=args.num_sample)
    save_path = Path("./dataset/charged")
    save_path.mkdir(exist_ok=True, parents=True)
    # Multi-threading, in default it will use all threads in the cpu.

    def sample(i):
        sim = ChargedParticlesSim(seed=seeds[i])
        loc, vel, charge = sim.sample_trajectory(
            T=args.seq_len, sample_freq=args.sample_freq)
        return [loc, vel, charge]
    with Pool() as p:
        results = list(p.map(sample, range(args.num_sample)))
    print(len(results))
    loc, vel, charge = [np.stack(x) for x in zip(*results)]
    np.savez_compressed(save_path / args.filename,
                        loc=loc, vel=vel, charge=charge)
