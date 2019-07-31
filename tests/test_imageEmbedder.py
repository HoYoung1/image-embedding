import os
from unittest import TestCase

import torch

from resnetembedding import ResnetEmbedder


class TestImageEmbedder(TestCase):

    def test___call__(self):
        # Arrange
        img_name = os.path.join(os.path.dirname(__file__), "imagesLFW", "AJ_Cook_001.jpg")
        batch_size = 32
        embedder = ResnetEmbedder()
        input = torch.rand((batch_size, 3, 224, 224))

        # Act
        actual = list(embedder(input))

        # Assert just n record with length
        self.assertEqual(input.shape[0], len(actual))
