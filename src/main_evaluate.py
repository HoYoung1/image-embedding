import argparse
import os
import sys

import numpy as np
import torch

from CMCScore import CMCScore
from EuclideanPairwiseDistance import EuclideanPairwiseDistance

sys.path.append(os.path.abspath('./src'))
from resnetembedding import ResnetEmbedder
from datasets.Market1501Dataset import Market1501Dataset


def run(images_dir):
    images_dataset = Market1501Dataset(images_dir)

    embeddings = []
    class_person = []
    for person_img, target in images_dataset:
        embedder = ResnetEmbedder()
        # TODO : Fix this numpy stuff.. this is really bad
        embedding = embedder(person_img).to(device="cpu").data.numpy()[0]
        embeddings.append(embedding)
        class_person.append(target)

    embeddings = np.stack(embeddings)

    # Compute pairwise
    distance_metric = EuclideanPairwiseDistance()
    pairwise_distance = distance_metric(torch.tensor(embeddings))

    cmc_scorer = CMCScore()
    score = cmc_scorer.score(pairwise_distance, class_person, 2)
    print(score)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rawimagesdir",
                        help="The directory path cpontaining market dataset")

    args = parser.parse_args()

    print("Runnning...")
    run(args.rawimagesdir)
