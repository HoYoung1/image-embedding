from CMCScore import CMCScore
from EuclideanPairwiseDistance import EuclideanPairwiseDistance
from evaluator import Evaluator
from resnetembedding import ResnetEmbedder


class EvaluationFactory:

    def get_evaluator(self):
        model = ResnetEmbedder()
        distance_metric = EuclideanPairwiseDistance()
        scorer = CMCScore()

        evaluator = Evaluator(model, distance_metric, scorer)

        return evaluator
