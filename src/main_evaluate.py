import argparse
import os
import sys

from datasets.Market1501Dataset import Market1501Dataset
from evalution_factory import EvaluationFactory

sys.path.append(os.path.abspath('./src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rawimagesdir",
                        help="The directory path cpontaining market dataset")

    args = parser.parse_args()

    print("Runnning...")

    evaluator = EvaluationFactory().get_evaluator()
    dataset = Market1501Dataset(args.rawimagesdir)
    result = evaluator.evaluate(dataset)

    print("Dataset {} , Result : {} ".format(type(dataset), result))
