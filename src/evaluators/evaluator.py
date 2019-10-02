from evaluators.evaluator_base import EvaluatorBase


class Evaluator(EvaluatorBase):
    def __init__(self, distance_measurer, scorer, k_threshold=1):
        self.k_threshold = k_threshold
        self.distance_metric = distance_measurer
        self.scorer = scorer

    def __call__(self, actual_embedding, target_class):
        # Compute pairwise
        pairwise_distance = self.distance_metric(actual_embedding)

        score = self.scorer.score(pairwise_distance, target_class, k_threshold=self.k_threshold)
        return score
