from evaluators.evaluator_base import EvaluatorBase


class Evaluator(EvaluatorBase):
    def __init__(self, distance_measurer, scorer, k_threshold=1):
        self.k_threshold = k_threshold
        self.distance_metric = distance_measurer
        self.scorer = scorer

    def __call__(self, gallery_embedding, gallery_target_class, query_embedding=None, query_target_class=None):
        # Compute pairwise
        pairwise_distance = self.distance_metric(gallery_embedding, query_embedding)

        score = self.scorer.score(pairwise_distance, target_label_x_gallery=gallery_target_class,
                                  target_label_y_query=query_target_class, k_threshold=self.k_threshold)
        return score
