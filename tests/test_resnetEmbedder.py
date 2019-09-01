from unittest import TestCase

import torch

from resnetembedding import ResnetEmbedder


class TestResnetEmbedder(TestCase):

    def test___call__(self):
        # Arrange
        batch_size = 32
        embedder = ResnetEmbedder()
        input = torch.rand((batch_size, 3, 224, 224))

        # Act
        actual = list(embedder(input))

        # Assert just n record with length
        self.assertEqual(input.shape[0], len(actual))
