import os
import tempfile
from unittest import TestCase

import torch

from Predictor import Predictor
from datasets.Market1501Dataset import Market1501Dataset
from train_factory import TrainFactory


class TestSitPredictor(TestCase):

    def test___call__(self):
        # Arrange
        output_dir = tempfile.mkdtemp()
        self._run_train(output_dir)
        sut = Predictor(output_dir)
        img_name = os.path.join(os.path.dirname(__file__), "imagesMarket1501", "0007_c2s3_070952_01.jpg")

        # Act
        result = sut(img_name)

        # Assert
        self.assertIsInstance(result, torch.Tensor)

    def _run_train(self, output_dir):
        img_dir = os.path.join(os.path.dirname(__file__), "imagesMarket1501")
        dataset = Market1501Dataset(img_dir)
        factory = TrainFactory(num_workers=1, epochs=2, batch_size=2, early_stopping=True, patience_epochs=2)
        pipeline = factory.get(dataset)
        pipeline.run(dataset, dataset, output_dir)
