import logging
import os

from PIL import Image
from skimage import io
from torchvision.transforms import transforms

from datasets.CustomDataset import CustomDataset

"""
Market1501 dataset
"""


class Market1501Dataset(CustomDataset):

    @property
    def num_classes(self):
        return len(self._zero_indexed_labels)

    def __init__(self, raw_directory, min_img_size_h=214, min_img_size_w=214):
        self.min_img_size_w = min_img_size_w
        self.min_img_size_h = min_img_size_h
        self.raw_directory = raw_directory

        self._len = None
        self._files = [os.path.join(self.raw_directory, f) for f in os.listdir(self.raw_directory) if f.endswith("jpg")]

        # The market 1501 dataset files have the naming convention target_camerasite_..., e.g. 1038_c2s2_131202_03.jpeg
        self._target_raw_labels = [os.path.basename(f).split("_")[0] for f in self._files]
        self._zero_indexed_labels = {}
        for rc in self._target_raw_labels:
            self._zero_indexed_labels[rc] = self._zero_indexed_labels.get(rc, len(self._zero_indexed_labels))

    def __len__(self):
        if self._len is None:
            self._len = len(self._files)

        return self._len

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __getitem__(self, index):
        target = self._zero_indexed_labels[self._target_raw_labels[index]]
        self.logger.debug("preprocessing imaage {}".format(self._files[index]))
        return self._pre_process_image(io.imread(self._files[index])), target

    def _pre_process_image(self, image):
        # pre-process data
        image = Image.fromarray(image)
        # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
        transform_pipeline = transforms.Compose([transforms.Resize((self.min_img_size_h, self.min_img_size_w)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      # torch image: C X H X W
                                                                      std=[0.229, 0.224, 0.225])])
        img_tensor = transform_pipeline(image)
        # Add batch [N, C, H, W]
        # img_tensor = img.unsqueeze(0)
        return img_tensor
