from datasets.caviar_dataset import CaviarDataset
from datasets.evaluation_dataset_factorybase import EvaluationDatasetFactoryBase


class CaviarFactory(EvaluationDatasetFactoryBase):

    def get(self, query_images, gallery_images=None):
        gallery_images = gallery_images or query_images
        query_dataset = CaviarDataset(query_images, min_img_size_h=256, min_img_size_w=128)
        gallery_dataset = CaviarDataset(gallery_images, min_img_size_h=256, min_img_size_w=128,
                                        initial_label_map=query_dataset.label_number_map)

        return query_dataset, gallery_dataset
