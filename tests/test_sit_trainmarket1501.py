import os
import tempfile
from unittest import TestCase

from dataset_factory import DatasetFactory
from train_factory import TrainFactory


class TestSitTrainMarket1501(TestCase):

    def test_run(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "imagesMarket1501")
        # get dataset
        dataset_factory = DatasetFactory().get_datasetfactory("Market1501Factory")
        dataset = dataset_factory.get(img_dir)

        # get train factory
        train_factory = TrainFactory(num_workers=1, epochs=2, batch_size=2, early_stopping=True, patience_epochs=2)
        output_dir = tempfile.mkdtemp()

        # Act
        pipeline = train_factory.get(dataset)
        pipeline.run(dataset, dataset, output_dir)
