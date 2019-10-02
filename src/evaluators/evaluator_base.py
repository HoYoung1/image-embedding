class EvaluatorBase:
    """
    Abstract base class for evalutaor
    """

    def __call__(self, gallery_embedding, gallery_target_class, query_embedding=None, query_target_class=None):
        raise NotImplementedError
