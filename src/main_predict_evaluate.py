# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from dataset_factory_service_locator import DatasetFactoryServiceLocator
from evaluator_factory_service_locator import EvalutorFactoryServiceLocator
from predictor import Predictor


class PredictEvaluate:

    def __call__(self, dataset_factory_name, model_path, gallery_images_dir, query_images_dir=None,
                 eval_factory_name="EvaluationFactory"):
        evalfactory = EvalutorFactoryServiceLocator().get_factory(eval_factory_name)
        evaluator = evalfactory.get_evaluator()
        datasetfactory = DatasetFactoryServiceLocator().get_factory(dataset_factory_name)

        class_person_gallery, embeddings_gallery = self._get_predictions(gallery_images_dir, datasetfactory, model_path)

        class_person_query, embeddings_query = None, None
        if query_images_dir is not None:
            class_person_query, embeddings_query = self._get_predictions(query_images_dir, datasetfactory, model_path)

        result = evaluator(embeddings_gallery, class_person_gallery, query_embedding=embeddings_query,
                           query_target_class=class_person_query)
        return result

    def _get_predictions(self, rawimagesdir, datasetfactory, model_path):
        dataset_query = datasetfactory.get(rawimagesdir)
        batch_size = min(len(dataset_query), 32)
        dataloader_query = DataLoader(dataset_query, batch_size=batch_size, shuffle=False)
        model = Predictor(model_path)
        embeddings = []
        class_person = []
        for person_img, target in dataloader_query:
            embedding = model(person_img)
            embeddings.extend(embedding)
            class_person.extend(target)

        embeddings = torch.stack(embeddings)
        class_person = torch.stack(class_person)
        return class_person, embeddings


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
