import h5py
import numpy as np
import healpy as hp
import os
import copy
from multiprocessing import Pool

class ScanFields:
    def __init__(self):
        self.ss = {}
        self.hitmap = []
        self.h = []
        self.spins = []
        self.mean = []
        self.std = []
        self.all_channels = [
            'L1-040','L2-050','L1-060','L3-068','L2-068','L4-078','L1-078','L3-089','L2-089','L4-100','L3-119','L4-140',
            'M1-100','M2-119','M1-140','M2-166','M1-195',
            'H1-195','H2-235','H1-280','H2-337','H3-402'
        ]

    @classmethod
    def load_det(cls, base_path, filename):
        assert filename.endswith('.h5'), "Invalid file path. Must be a .h5 file"
        instance = cls()
        with h5py.File(os.path.join(base_path, filename), 'r') as f:
            instance.ss = {key: value[()] for key, value in zip(f['ss'].keys(), f['ss'].values()) if key != "quat"}
            instance.hitmap = f['hitmap'][:]
            instance.h = f['h'][:, 0, :]
            instance.h[np.isnan(instance.h)] = 1.0
            quantify_group = f['quantify']
            instance.spins = quantify_group['n'][()]
            instance.mean = quantify_group['mean'][()]
            instance.std = quantify_group['std'][()]
        return instance

    @classmethod
    def load_channel(cls, base_path, channel):
        dirpath = os.path.join(base_path, channel)
        filenames = os.listdir(dirpath)
        first_sf = cls.load_det(dirpath, filenames[0])
        instance = cls()
        instance.hitmap = np.zeros_like(first_sf.hitmap)
        instance.h = np.zeros_like(first_sf.h)
        for filename in filenames:
            sf = cls.load_det(dirpath, filename)
            instance.hitmap += sf.hitmap
            instance.h += sf.hitmap[:, np.newaxis] * sf.h
        instance.h /= instance.hitmap[:, np.newaxis]
        instance.mean = np.array([np.mean(instance.hitmap), np.mean(instance.h, axis=0)[0]])
        instance.std = np.array([np.std(instance.hitmap), np.std(instance.h, axis=0)[0]])
        instance.spins = first_sf.spins
        return instance

    @classmethod
    def _load_channel_task(cls, args):
        base_path, ch = args
        return cls.load_channel(base_path, ch)

    @classmethod
    def load_full_FPU(cls, base_path, channel_list, max_workers=None):
        if max_workers is None:
            max_workers = os.cpu_count()
        print(f"Using {max_workers} processes")
        with Pool(processes=max_workers) as pool:
            crosslink_channels = pool.map(cls._load_channel_task, [(base_path, ch) for ch in channel_list])
        instance = cls()
        hitmap = np.zeros_like(crosslink_channels[0].hitmap)
        h = np.zeros_like(crosslink_channels[0].h)
        for sf in crosslink_channels:
            hitmap += sf.hitmap
            h += sf.hitmap[:, np.newaxis] * sf.h
        instance.hitmap = hitmap
        instance.h = h / hitmap[:, np.newaxis]
        instance.mean = np.array([np.mean(hitmap), np.mean(h, axis=0)[0]])
        instance.std = np.array([np.std(hitmap), np.std(h, axis=0)[0]])
        instance.spins = crosslink_channels[0].spins
        return instance

    def get_xlink(self, spin_n):
        if spin_n == 0:
            return np.ones_like(self.h[:, 0]) + 1j * np.zeros_like(self.h[:, 0])
        if spin_n < 0:
            return self.h[:, np.abs(spin_n) - 1].conj()
        else:
            return self.h[:, spin_n - 1]

    def get_covmat_3D(self):
        covmat = np.array([
            [self.get_xlink(0)     , self.get_xlink(-2)/2.0, self.get_xlink(2)/2.0],
            [self.get_xlink(2)/2.0 , self.get_xlink(0)/4.0 , self.get_xlink(4)/4.0],
            [self.get_xlink(-2)/2.0, self.get_xlink(-4)/4.0, self.get_xlink(0)/4.0]
            ])
        return covmat

    def get_covmat_2D(self):
        covmat = np.array([
            [self.get_xlink(0)/4.0 , self.get_xlink(4)/4.0],
            [self.get_xlink(-4)/4.0, self.get_xlink(0)/4.0]
            ])
        return covmat

    def t2b(self):
        """Transform Top detector cross-link to Bottom detector cross-link
        Top and bottom detector make a orthogonal pair.
        """
        class_copy = copy.deepcopy(self)
        class_copy.h *= np.exp(-1j * self.spins * (np.pi / 2))
        return class_copy

    def __add__(self, other):
        """Add hitmap and h of two Scanfield instances"""
        if not isinstance(other, ScanFields):
            return NotImplemented
        result = copy.deepcopy(self)
        result.hitmap += other.hitmap
        result.h = (self.h*self.hitmap[:, np.newaxis] + other.h*other.hitmap[:, np.newaxis])/result.hitmap[:, np.newaxis]
        result.mean = np.array([np.mean(result.hitmap), np.mean(result.h, axis=0)[0]])
        result.std = np.array([np.std(result.hitmap), np.std(result.h, axis=0)[0]])
        return result
