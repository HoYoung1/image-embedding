from torch.utils.data import DataLoader

from datasets.Market1501Dataset import Market1501Dataset


class Market1501DatasetFactory:

    @staticmethod
    def dataset_name():
        return "Market1501"

    def get(self, images_dir):
        batch_size = 32
        dataset = Market1501Dataset(images_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return dataloader
