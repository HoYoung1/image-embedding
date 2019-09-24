from unittest import TestCase

import torch

from euclidean_pairwise_distance import EuclideanPairwiseDistance


class TestEuclideanPairwiseDistance(TestCase):

    def test___call__(self):
        # Arrange
        x = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=torch.float32)
        sut = EuclideanPairwiseDistance()
        expected = torch.tensor([[0.0000, 1.7321, 3.4641, 5.1962],
                                 [1.7321, 0.0000, 1.7321, 3.4641],
                                 [3.4641, 1.7321, 0.0000, 1.7321],
                                 [5.1962, 3.4641, 1.7321, 0.0000]]).round()

        # Act
        actual = sut(x).round()

        # Assert
        self.assertTrue(torch.equal(actual, expected))
