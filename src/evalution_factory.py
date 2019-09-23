from CMCScore import CMCScore
from EuclideanPairwiseDistance import EuclideanPairwiseDistance
from evaluator import Evaluator


class EvaluationFactory:

    def get_evaluator(self):
        distance_metric = EuclideanPairwiseDistance()
        scorer = CMCScore()
        k_threshold = 5

        evaluator = Evaluator(distance_metric, scorer, k_threshold=k_threshold)

        return evaluator
