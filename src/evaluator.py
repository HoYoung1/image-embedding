from evaluator_base import EvaluatorBase


class Evaluator(EvaluatorBase):
    def __init__(self, distance_measurer, scorer, k_threshold=1, batch_size=32):
        self.batch_size = batch_size
        self.k_threshold = k_threshold
        self.distance_metric = distance_measurer
        self.scorer = scorer

    def evaluate(self, actual_embedding, target_class):
        # Compute pairwise
        pairwise_distance = self.distance_metric(actual_embedding)

        score = self.scorer.score(pairwise_distance, target_class, self.k_threshold)
        return score
