import os
import tempfile
from unittest import TestCase

from datasets.Market1501Dataset import Market1501Dataset
from train_factory import TrainFactory


class TestSitTrainMarket1501(TestCase):

    def test_run(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "imagesMarket1501")
        dataset = Market1501Dataset(img_dir)
        factory = TrainFactory(num_workers=1, epochs=2, batch_size=2, early_stopping=True, patience_epochs=2)
        output_dir = tempfile.mkdtemp()

        # Act
        pipeline = factory.get(dataset)
        pipeline.run(dataset, dataset, output_dir)
