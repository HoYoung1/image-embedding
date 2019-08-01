from CMCScore import CMCScore
from EuclideanPairwiseDistance import EuclideanPairwiseDistance
from evaluator import Evaluator
from resnetembedding import ResnetEmbedder


class EvaluationFactory:

    def get_evaluator(self):
        model = ResnetEmbedder()
        distance_metric = EuclideanPairwiseDistance()
        scorer = CMCScore()
        k_threshold = 5

        evaluator = Evaluator(model, distance_metric, scorer, k_threshold=k_threshold)

        return evaluator
