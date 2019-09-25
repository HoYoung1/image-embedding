class EvaluatorBase:
    """
    Abstract base class for evalutaor
    """

    def __call__(self, actual_embedding, target_class):
        raise NotImplementedError
