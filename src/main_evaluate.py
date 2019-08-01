import argparse
import os
import sys

from dataset_factory import DatasetFactory
from evalution_factory import EvaluationFactory

sys.path.append(os.path.abspath('./src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rawimagesdir",
                        help="The directory path cpontaining market dataset")

    args = parser.parse_args()

    print("Runnning...")

    evaluator = EvaluationFactory().get_evaluator()
    dataset = DatasetFactory().get(args.rawimagesdir)
    result = evaluator.evaluate(dataset)

    print("Dataset {} , Result : {} ".format(type(dataset), result))
