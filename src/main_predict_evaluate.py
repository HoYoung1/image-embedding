import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from dataset_factory_service_locator import DatasetFactoryServiceLocator
from evaluator_factory_service_locator import EvalutorFactoryServiceLocator
from predictor import Predictor


class PredictEvaluate:

    def __call__(self, dataset_factory_name, model_path, rawimagesdir):
        evalfactory = EvalutorFactoryServiceLocator().get_factory("EvaluationFactory")
        evaluator = evalfactory.get_evaluator()
        datasetfactory = DatasetFactoryServiceLocator().get_factory(dataset_factory_name)

        dataset = datasetfactory.get(rawimagesdir)
        batch_size = min(len(dataset), 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = Predictor(model_path)

        embeddings = []
        class_person = []
        for person_img, target in dataloader:
            embedding = model(person_img)
            embeddings.extend(embedding)
            class_person.extend(target)

        embeddings = torch.stack(embeddings)
        class_person = torch.stack(class_person)

        result = evaluator(embeddings, class_person)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        help="The type of dataset",
                        choices=DatasetFactoryServiceLocator().factory_names, required=True)
    parser.add_argument("--modelpath",
                        help="The model path", required=True)

    parser.add_argument("--rawimagesdir",
                        help="The directory path cpontaining market dataset", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    result = PredictEvaluate()(args.dataset, args.modelpath, args.rawimagesdir)
    print("Score is {}".format(result))
