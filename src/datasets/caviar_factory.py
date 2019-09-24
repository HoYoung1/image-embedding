from datasets.caviar_dataset import CaviarDataset
from datasets.custom_dataset_factorybase import CustomDatasetFactoryBase


class CaviarFactory(CustomDatasetFactoryBase):

    def get(self, images_dir):
        dataset = CaviarDataset(images_dir)

        return dataset
