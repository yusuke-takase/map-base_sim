import h5py
import numpy as np
import healpy as hp
import os
import copy
from multiprocessing import Pool

class ScanFields:
    """ Class to store the scan fields data of detectors """
    def __init__(self):
        """ Initialize the class with empty data

        ss (dict):  of the scanning strategy parameters
        hitmap (np.ndarray): hitmap of the detector
        h (np.ndarray): cross-link (orientation function) of the detector
        spins (np.ndarray): array of spin numbers
        std (np.ndarray): standard deviation of the hitmap and h
        mean (np.ndarray): mean of the hitmap and h
        all_channels (list): list of all the channels in the LiteBIRD

        """
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
    def load_det(cls, base_path: str, filename: str):
        """ Load the scan fields data of a detector from a .h5 file

        Args:
            base_path (str): path to the directory containing the .h5 file

            filename (str): name of the .h5 file containing the scan fields data simulated by Falcons.jl
            The fileformat requires cross-link_2407-dataset's format.
            The file should contain the following groups:
                - ss: scanning strategy parameters
                - hitmap: hitmap of the detector
                - h: cross-link (orientation function) of the detector
                - quantify: group containing the following datasets
                    - n: number of spins
                    - mean: mean of the hitmap and h
                    - std: standard deviation of the hitmap and h
        Returns:
            instance (ScanFields): instance of the ScanFields class containing the scan fields data of the detector
        """
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
    def load_channel(cls, base_path: str, channel: str):
        """Load the scan fields data of a channel from the directory containing the .h5 files

        Args:
            base_path (str): path to the directory containing the .h5 files

            channel (str): name of the channel to load the scan fields data from

        Returns:
            instance (ScanFields): instance of the ScanFields class containing the scan fields data of the channel
        """
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
    def load_full_FPU(cls, base_path: str, channel_list: list, max_workers=None):
        """ Load the scan fields data of all the channels in the FPU from the directory containing the .h5 files

        Args:
            base_path (str): path to the directory containing the .h5 files

            channel_list (list): list of channels to load the scan fields data from

            max_workers (int): number of processes to use for loading the scan
                               fields data of the channels. Default is None, which
                               uses the number of CPUs in the system

        Returns:
            instance (ScanFields): instance of the ScanFields class containing the scan
                                   fields data of all the channels in the FPU
        """
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

    def get_xlink(self, spin_n: int):
        """Get the cross-link of the detector for a given spin number

        Args:
            spin_n (int): spin number for which the cross-link is to be obtained

            If spin_n is 0, the cross-link for the spin number 0 is returned, i.e,
            the map which has one in the real part and zero in the imaginary part.

        Returns:
            xlink (1d-np.ndarray): cross-link of the detector for the given spin number
        """

        if spin_n == 0:
            return np.ones_like(self.h[:, 0]) + 1j * np.zeros_like(self.h[:, 0])
        if spin_n < 0:
            return self.h[:, np.abs(spin_n) - 1].conj()
        else:
            return self.h[:, spin_n - 1]

    def get_covmat_3D(self):
        """Get the covariance matrix of the detector in 3x3 matrix form"""
        covmat = np.array([
            [self.get_xlink(0)     , self.get_xlink(-2)/2.0, self.get_xlink(2)/2.0],
            [self.get_xlink(2)/2.0 , self.get_xlink(0)/4.0 , self.get_xlink(4)/4.0],
            [self.get_xlink(-2)/2.0, self.get_xlink(-4)/4.0, self.get_xlink(0)/4.0]
            ])
        return covmat

    def get_covmat_2D(self):
        """Get the covariance matrix of the detector in 2x2 matrix form
        It is used for the defferencial signal analisys.
        """
        covmat = np.array([
            [self.get_xlink(0)/4.0 , self.get_xlink(4)/4.0],
            [self.get_xlink(-4)/4.0, self.get_xlink(0)/4.0]
            ])
        return covmat

    def t2b(self):
        """Transform Top detector cross-link to Bottom detector cross-link
        It assume top and bottom detector make a orthogonal pair.
        """
        class_copy = copy.deepcopy(self)
        class_copy.h *= np.exp(-1j * self.spins * (np.pi / 2))
        return class_copy

    def __add__(self, other):
        """Add hitmap and h of two Scanfield instances
        For the hitmap, it adds the hitmap of the two instances
        For h, it adds the cross-link of the two instances weighted by the hitmap
        """
        if not isinstance(other, ScanFields):
            return NotImplemented
        result = copy.deepcopy(self)
        result.hitmap += other.hitmap
        result.h = (self.h*self.hitmap[:, np.newaxis] + other.h*other.hitmap[:, np.newaxis])/result.hitmap[:, np.newaxis]
        result.mean = np.array([np.mean(result.hitmap), np.mean(result.h, axis=0)[0]])
        result.std = np.array([np.std(result.hitmap), np.std(result.h, axis=0)[0]])
        return result

class Field:
    """ Class to store the field data of detectors """
    def __init__(self, field: np.ndarray, spin: int):
        """Initialize the class with field and spin data

        Args:
            field (np.ndarray): field data of the detector
            spin (int): spin number of the detector
        """
        if all(isinstance(x, float) for x in field):
            self.field = field + 1j*np.zeros(len(field))
        else:
            self.field = field
        self.spin = spin

class SignalFields:
    """ Class to store the signal fields data of detectors """
    def __init__(self, *fields: Field):
        """ Initialize the class with field data

        Args:
            fields (Field): field (map) data of the signal
        """
        self.fields = sorted(fields, key=lambda field: field.spin)
        self.spins = np.array([field.spin for field in self.fields])

def couple_fields(scan_fields: ScanFields, signal_fields: SignalFields, spin_out: int):
    """ Multiply the scan fields and signal fields to get the detected fields by given cross-linking

    Args:
        scan_fields (ScanFields): scan fields data of the detector

        signal_fields (SignalFields): signal fields data of the detector

        spin_out (int): spin number of the output field

    Returns:
        results (np.ndarray): detected fields by the given cross-linking
    """
    results = []
    for i in range(len(signal_fields.spins)):
        n = spin_out - signal_fields.spins[i]
        print(f"n-n': {n}, n: {signal_fields.spins[i]}")
        results.append(scan_fields.get_xlink(n) * signal_fields.fields[i].field)
    return np.array(results)
