from datasets.Market1501Dataset import Market1501Dataset


class Market1501DatasetFactory:

    @staticmethod
    def dataset_name():
        return "Market1501"

    def get(self, images_dir):
        dataset = Market1501Dataset(images_dir)

        return dataset
