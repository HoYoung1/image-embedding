import logging
import os

from torch import nn
from torch.optim import Adam

from model_resnet import ModelResnet
from train import Train
from train_pipeline import TrainPipeline


class TrainFactory:

    def __init__(self, epochs=50, early_stopping=True, patience_epochs=10, batch_size=32, num_workers=None,
                 additional_args=None):

        if num_workers is None and os.cpu_count() > 1:
            self.num_workers = os.cpu_count() - 1
        else:
            self.num_workers = 0

        self.batch_size = batch_size
        self.patience_epochs = patience_epochs
        self.early_stopping = early_stopping
        self.epochs = epochs
        self.additional_args = additional_args or {}

        self.learning_rate = float(self._get_value(self.additional_args, "learning_rate", ".01"))

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self.logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

    def get(self, train_dataset):
        trainer = Train(patience_epochs=self.patience_epochs, early_stopping=self.early_stopping, epochs=self.epochs)
        model = ModelResnet(n_classes=train_dataset.num_classes)
        # optimiser = SGD(lr=self.learning_rate, params=model.parameters(), momentum=0.9)
        optimiser = Adam(lr=self.learning_rate, params=model.parameters())
        train_pipeline = TrainPipeline(batch_size=self.batch_size,
                                       optimiser=optimiser,
                                       trainer=trainer,
                                       num_workers=self.num_workers,
                                       loss_func=nn.CrossEntropyLoss(),
                                       model=model)

        return train_pipeline
