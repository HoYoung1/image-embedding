from torch.utils.data import DataLoader

from datasets.CaviarDataset import CaviarDataset


class DatasetFactory:

    def get(self, images_dir):
        batch_size = 32
        dataset = CaviarDataset(images_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return dataloader
