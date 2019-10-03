from torch.utils.data import Dataset


class TripletDatasetBase(Dataset):
    """
    This is the base class for a custom dataset that retruns a triplet of the features (p,q,n) target
    """

    @property
    def num_classes(self):
        raise NotImplementedError
