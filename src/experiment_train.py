import argparse
import logging
import os
import sys

from dataset_factory_service_locator import DatasetFactoryServiceLocator
from train_factory import TrainFactory


class ExperimentTrain:

    def __init__(self):
        pass

    def run(self, dataset_factory_name, train_dir, val_dir, out_dir, batch_size=32, epochs=10, patience_epoch=2,
            additional_args=None):
        # Set up dataset
        datasetfactory = DatasetFactoryServiceLocator().get_factory(dataset_factory_name)
        train_dataset = datasetfactory.get(train_dir)
        val_dataset = datasetfactory.get(val_dir)

        # Get trainpipeline
        factory = TrainFactory(num_workers=None, epochs=epochs, batch_size=batch_size, early_stopping=True,
                               patience_epochs=patience_epoch, additional_args=additional_args)
        pipeline = factory.get(train_dataset)

        # Start training
        pipeline.run(train_dataset, val_dataset, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        help="The type of dataset",
                        choices=DatasetFactoryServiceLocator().factory_names, required=True)

    parser.add_argument("--traindir",
                        help="The input train  data", default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--valdir",
                        help="The input val data", default=os.environ.get("SM_CHANNEL_VAL", None))

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_MODEL_DIR", None))

    parser.add_argument("--batchsize", help="The batch size", type=int, default=32)

    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)

    parser.add_argument("--patience", help="The number of patience epochs", type=int, default=10)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ExperimentTrain().run(args.dataset,
                          args.traindir,
                          args.valdir,
                          args.outdir,
                          args.batchsize,
                          args.epochs,
                          args.patience)
