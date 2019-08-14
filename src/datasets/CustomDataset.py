from torch.utils.data import Dataset


class CustomDataset(Dataset):

    @property
    def num_classes(self):
        raise NotImplementedError
