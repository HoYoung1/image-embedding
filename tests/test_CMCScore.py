from unittest import TestCase

import torch

from CMCScore import CMCScore


class TestCMCScore(TestCase):
    def test_score(self):
        # Arrange
        pairwise_distance = torch.tensor([[0, 2, 1], [2, 0, 3], [1, 3, 0]])
        sut = CMCScore()
        expected = 66.67

        # Act
        actual = sut.score(pairwise_distance, [2, 1, 2], 2)

        # Assert
        self.assertEqual(round(actual, 2), expected)
