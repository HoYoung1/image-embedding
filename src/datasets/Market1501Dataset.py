import os

from skimage import io
from torch.utils.data import Dataset

"""
Market1501 dataset
"""


class Market1501Dataset(Dataset):

    def __init__(self, raw_directory):
        self.raw_directory = raw_directory

        self._len = None
        self._files = [os.path.join(self.raw_directory, f) for f in os.listdir(self.raw_directory)]

        # The market 1501 dataset files have the naming convention target_camerasite_..., e.g. 1038_c2s2_131202_03.jpeg
        self._target_raw_labels = [os.path.basename(f).split("_")[0] for f in self._files]
        self._zero_indexed_labels = {}
        for rc in self._target_raw_labels:
            self._zero_indexed_labels[rc] = self._zero_indexed_labels.get(rc, len(self._zero_indexed_labels))

    def __len__(self):
        if self._len is None:
            self._len = len(self._files)

        return self._len

    def __getitem__(self, index):
        target = self._zero_indexed_labels[self._target_raw_labels[index]]
        return io.imread(self._files[index]), target
