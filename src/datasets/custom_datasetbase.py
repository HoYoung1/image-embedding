from torch.utils.data import Dataset


class CustomDatasetBase(Dataset):
    """
    This is the base class for a custom dataset
    """

    @property
    def num_classes(self):
        raise NotImplementedError
