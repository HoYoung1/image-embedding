from __future__ import absolute_import

import torch


class EuclideanPairwiseDistance():
    """
    Computes pairwise euclidean distance
    """

    def __call__(self, x):
        """
Computes pairwise euclidean distance
        :param x: n x f float matrix  ( n samples and f features)
        :param y: optional y matrix
        :return: pair wise euclidean distance
        """
        assert len(x.shape) == 2, "Requires a 2D matrix"

        # Note: not using (x-x.unsqueeze(1))^2 as it creates a very large matrix

        squared_x = torch.pow(x, 2).sum(1)

        y = torch.t(x)
        xy = x @ y
        result = torch.sqrt(squared_x + squared_x.unsqueeze(1) - 2 * xy)

        return result
