from datasets.custom_dataset_factorybase import CustomDatasetFactoryBase
from datasets.market1501_dataset import Market1501Dataset


class Market1501Factory(CustomDatasetFactoryBase):

    def __init__(self):
        pass

    def get(self, images_dir):
        # Market150 dataset size is 64 width, height is 128, so we maintain the aspect ratio
        dataset = Market1501Dataset(images_dir, min_img_size_h=256, min_img_size_w=128)

        return dataset
