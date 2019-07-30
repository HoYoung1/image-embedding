from __future__ import absolute_import

import torch

"""
Computes pairwise euclidean distance
"""


class EuclideanPairwiseDistance():

    def __call__(self, x, y=None):
        """


        :param x: n x f float matrix  ( n samples and f features)
        :param y: optional y matrix
        :return: pair wise euclidean distance
        """
        assert len(x.shape) == 2, "Requires a 2D matrix"

        if y is None:
            y = x

        dist = torch.sqrt(torch.pow(x.unsqueeze(1) - x, 2).sum(dim=2))

        return dist
