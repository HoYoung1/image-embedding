import argparse
import logging
import os
import sys

from datasets.Dataset_factory import DatasetFactory
from evalution_factory import EvaluationFactory

sys.path.append(os.path.abspath('./src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset",
                        help="The type of dataset",
                        choices=DatasetFactory.get_datasets())
    parser.add_argument("rawimagesdir",
                        help="The directory path cpontaining market dataset")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    evaluator = EvaluationFactory().get_evaluator()
    datasetfactory = DatasetFactory.get_datasetfactory(args.dataset)
    dataset = datasetfactory.get(args.rawimagesdir)
    result = evaluator.evaluate(dataset)

    print("Dataset {} , Result : {} ".format(type(dataset), result))
