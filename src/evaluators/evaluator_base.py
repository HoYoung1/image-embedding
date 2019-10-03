class EvaluatorBase:
    """
    Abstract base class for evalutaor
    """

    def __call__(self, query_embedding, query_target_class, gallery_embedding=None, gallery_target_class=None):
        raise NotImplementedError
