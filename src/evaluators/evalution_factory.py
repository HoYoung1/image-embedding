from cmc_score import CMCScore
from euclidean_pairwise_distance import EuclideanPairwiseDistance
from evaluators.evaluator import Evaluator
from evaluators.evaluator_factory_base import EvaluatorFactoryBase


class EvaluationFactory(EvaluatorFactoryBase):
    """
    Creates a evaluator
    """

    def __init__(self, k_threshold=5):
        self.k_threshold = k_threshold

    def get_evaluator(self):
        distance_metric = EuclideanPairwiseDistance()
        scorer = CMCScore()

        evaluator = Evaluator(distance_metric, scorer, k_threshold=self.k_threshold)

        return evaluator
