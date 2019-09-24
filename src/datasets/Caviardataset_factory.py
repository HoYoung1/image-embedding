from datasets.CaviarDataset import CaviarDataset


class CaviarDatasetFactory:

    @staticmethod
    def dataset_name():
        return "Caviar"

    def get(self, images_dir):
        dataset = CaviarDataset(images_dir)

        return dataset
