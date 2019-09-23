class EvaluatorFactoryBase:
    """
    Abstract base class for evalutaor factor
    """

    def get_evaluator(self):
        raise NotImplementedError
