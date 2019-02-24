from collections import OrderedDict
from os import path

import numpy as np
from natsort import natsorted
from pyll.base import TorchDataset
from pyll.utils.timer import Timer


class ProteinLoc(TorchDataset):
    def __init__(self, data_directory_path: str, label_file: str = None, transforms=None, subset: float = 1.,
                 num_classes: int = None, verbose: bool = False, patching: bool = False, patch_size: int = 1024):
        """ Read samples from cyto dataset."""
        self.verbose = verbose
        self.patching = patching
        self.patch_size = patch_size
        self.n_classes = num_classes
        self.classes = ["Actin filaments", "Centrosome", "Cytosol", "Endoplasmic reticulum", "Golgi apparatus",
                        "Intermediate filaments", "Microtubules", "Mitochondria", "Nuclear membrane", "Nucleoli",
                        "Nucleus", "Plasma membrane", "Vesicles"]
        assert (path.exists(data_directory_path))

        # Load labels
        if label_file is not None:
            self.log("Collecting label data...")
            labels = self.load_labels(label_file)
            label_keys = list(natsorted(labels.keys()))
            if subset != 1.:
                label_keys = label_keys[:int(len(label_keys) * subset)]
            self.labels = labels
            self.label_keys = label_keys
            self.n_samples = len(label_keys)

        # Load sample paths
        sample_paths = self.load_sample_list(data_directory_path)
        if len(sample_paths) == 0:
            raise Exception("Empty dataset!")
        else:
            self.log("Found {} samples".format(len(sample_paths)))

        # expose everything important
        self.data_directory = data_directory_path
        self.samples = sample_paths
        self.transforms = transforms

        # load first sample and check shape
        i = 0
        sample = self[i]

        while sample['input'] is None and i < len(self):
            sample = self[i]
            i += 1

        if sample['input'] is not None:
            self.data_shape = sample['input'].shape
        else:
            self.data_shape = "Unknown"
        self.log("Discovered {} samples (subset={}) with shape {}".format(self.n_samples, subset, self.data_shape))

    def __len__(self):
        if hasattr(self, "label_keys"):
            return len(self.label_keys)
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        if hasattr(self, "label_keys"):
            sample_key = self.label_keys[idx]
        else:
            sample_key = list(self.samples.items())[idx][0]
        sample = self.read_sample(sample_key)

        if self.patching:
            input_dim = sample['input'].shape[1]
            patches_per_dim = np.ceil(input_dim / self.patch_size)
            # get all image patches
            step_size = int(
                self.patch_size - (self.patch_size * patches_per_dim - input_dim) / (patches_per_dim - 1))
            sample['input'] = sample['input'].permute(
                1, 2, 0).unfold(
                0, size=self.patch_size, step=step_size).unfold(
                1, size=self.patch_size, step=step_size).contiguous().view(
                -1, 4, self.patch_size, self.patch_size)
        return sample

    @property
    def shape(self):
        return self.data_shape

    @property
    def num_classes(self):
        return self.n_classes

    def log(self, message):
        if self.verbose:
            print(message)

    def read_sample(self, key):
        with Timer("Read Sample", verbose=self.verbose):
            X = self.load_sample(key=key)
            if self.transforms:
                X = self.transforms(X)
            if hasattr(self, "labels"):
                label = self.labels[key]
                return dict(input=X, target=label, ID=key)
            else:
                return dict(input=X, ID=key)

    def get_sample_keys(self):
        # TODO check
        if hasattr(self, "labels"):
            return self.label_keys.copy()
        else:
            return self.samples.keys().copy()

    def get_label_encoded(self, labels):
        label = np.zeros(shape=(self.n_classes), dtype=np.float32)
        for i in range(self.n_classes):
            label[i] = 1 if self.classes[i] in labels else 0
        return label

    def load_sample(self, key):
        """Load all npz"""
        npz = np.load(self.samples[key])
        return npz["sample"]

    def load_sample_list(self, directory: str):
        """Load all keys and file directory paths"""
        from glob import glob
        file_paths = glob(path.join(directory, '*.npz'))
        file_paths.sort()

        sample_paths = OrderedDict()
        for file_path in file_paths:
            key = path.splitext(path.basename(file_path))[0]
            sample_paths[key] = file_path
        return sample_paths

    def load_labels(self, file_path: str):
        """Load all keys and values"""
        labels = OrderedDict()
        with open(file_path, 'r') as in_file:
            for line in in_file:
                line = line.replace("\n", "")
                values = line.split(", ")
                key = values[0]
                labels[key] = self.get_label_encoded(values[1:])
        return labels
