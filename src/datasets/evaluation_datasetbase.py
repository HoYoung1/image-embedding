from torch.utils.data import Dataset


class EvaluationDatasetBase(Dataset):
    """
    This is the base class for a custom dataset
    """

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def label_number_map(self):
        raise NotImplementedError
