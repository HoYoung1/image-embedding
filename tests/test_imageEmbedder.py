import os
from unittest import TestCase

from skimage import io

from resnetembedding import ResnetEmbedder


class TestImageEmbedder(TestCase):

    def test___call__(self):
        # Arrange
        img_name = os.path.join(os.path.dirname(__file__), "imagesLFW", "AJ_Cook_001.jpg")
        image = io.imread(img_name)
        embedder = ResnetEmbedder()

        # Act
        actual = embedder(image)

        # Assert just 1 record with length
        self.assertEqual(actual.shape[0], 1)
        self.assertEqual(len(actual.shape), 2)
