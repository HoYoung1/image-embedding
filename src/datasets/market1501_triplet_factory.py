from datasets.custom_dataset_factorybase import CustomDatasetFactoryBase
from datasets.market1501_triplet_dataset import Market1501TripletDataset
from image_preprocessor import ImagePreprocessor


class Market1501TripletFactory(CustomDatasetFactoryBase):

    def __init__(self):
        pass

    def get(self, images_dir):
        # Market150 dataset size is 64 width, height is 128, so we maintain the aspect ratio
        # NOTE: for some reason oly 224 / 224 works, any other shape results in NAN
        dataset = Market1501TripletDataset(images_dir, min_img_size_h=256, min_img_size_w=128)
        processor = ImagePreprocessor(min_img_size_h=dataset.min_img_size_h, min_img_size_w=dataset.min_img_size_w,
                                      original_height=dataset.original_height, original_width=dataset.original_width)

        dataset.preprocessor = processor
        return dataset
