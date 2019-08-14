from torch.nn import NLLLoss
from torch.optim import SGD

from ModelResnet import ModelResnet
from train import Train
from train_pipeline import TrainPipeline


class TrainFactory:

    def __init__(self, epochs=50, early_stopping=True, patience_epochs=10, batch_size=32, num_workers=None, **kwargs):
        self.batch_size = batch_size
        self.patience_epochs = patience_epochs
        self.early_stopping = early_stopping
        self.epochs = epochs

        self.learning_rate = kwargs.get("learning_rate", .01)
        self.num_workers = num_workers

    def get(self, train_dataset):
        trainer = Train(patience_epochs=self.patience_epochs, early_stopping=self.early_stopping)
        model = ModelResnet(n_classes=train_dataset.num_classes)
        optimiser = SGD(lr=self.learning_rate, params=model.parameters())
        train_pipeline = TrainPipeline(epochs=self.epochs,
                                       batch_size=self.batch_size,
                                       optimiser=optimiser,
                                       trainer=trainer,
                                       num_workers=self.num_workers,
                                       loss_func=NLLLoss(),
                                       model=model)

        return train_pipeline
