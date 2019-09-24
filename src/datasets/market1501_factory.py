from datasets.custom_dataset_factorybase import CustomDatasetFactoryBase
from datasets.market1501_dataset import Market1501Dataset


class Market1501Factory(CustomDatasetFactoryBase):

    def __init__(self):
        pass

    def get(self, images_dir):
        dataset = Market1501Dataset(images_dir)

        return dataset
