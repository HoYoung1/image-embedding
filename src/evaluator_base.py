class EvaluatorBase:
    """
    Abstract base class for evalutaor
    """
    def evaluate(self, actual_embedding, target_class):
        raise NotImplementedError
