from __future__ import absolute_import

import torch

"""
Computes pairwise euclidean distance
"""


class EuclideanPairwiseDistance():

    def __call__(self, x, y=None):
        """
Computes pairwise euclidean distance
        :param x: n x f float matrix  ( n samples and f features)
        :param y: optional y matrix
        :return: pair wise euclidean distance
        """
        assert len(x.shape) == 2, "Requires a 2D matrix"

        if y is None:
            y = x

        x = x.unsqueeze(1)
        result = torch.pow(x - y, 2)
        result = torch.sqrt(result.sum(dim=2))

        return result
