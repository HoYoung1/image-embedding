from datasets.evaluation_dataset_factorybase import EvaluationDatasetFactoryBase
from datasets.market1501_dataset import Market1501Dataset


class Market1501Factory(EvaluationDatasetFactoryBase):

    def __init__(self):
        pass

    def get(self, query_images, gallery_images=None):
        # Market150 dataset size is 64 width, height is 128, so we maintain the aspect ratio
        # NOTE: for some reason oly 224 / 224 works, any other shape results in NAN

        gallery_images = gallery_images or query_images
        query_dataset = Market1501Dataset(query_images, min_img_size_h=256, min_img_size_w=128)
        gallery_dataset = Market1501Dataset(gallery_images, min_img_size_h=256, min_img_size_w=128,
                                            initial_label_map=query_dataset.label_number_map)

        return query_dataset, gallery_dataset
