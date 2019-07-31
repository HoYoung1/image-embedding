from CMCScore import CMCScore
from EuclideanPairwiseDistance import EuclideanPairwiseDistance
from evaluator import Evaluator
from resnetembedding import ResnetEmbedder


class EvaluationFactory:

    def get_evaluator(self):
        model = ResnetEmbedder()
        distance_metric = EuclideanPairwiseDistance()
        scorer = CMCScore()
        batch_size = 32
        k_threshold = 1

        evaluator = Evaluator(model, distance_metric, scorer, batch_size=batch_size, k_threshold=k_threshold)

        return evaluator
