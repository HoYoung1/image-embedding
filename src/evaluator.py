import torch
from torch.utils.data import DataLoader

from evaluator_base import EvaluatorBase


class Evaluator(EvaluatorBase):
    def __init__(self, model, distance_measurer, scorer, k_threshold=1, batch_size=32):
        self.batch_size = batch_size
        self.k_threshold = k_threshold
        self.model = model
        self.distance_metric = distance_measurer
        self.scorer = scorer

    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        embeddings = []
        class_person = []

        for person_img, target in dataloader:
            embedding = self.model(person_img)
            embeddings.extend(embedding)
            class_person.extend(target)

        embeddings = torch.stack(embeddings)
        class_person = torch.stack(class_person)

        # Compute pairwise
        pairwise_distance = self.distance_metric(embeddings)

        score = self.scorer.score(pairwise_distance, class_person, self.k_threshold)
        return score
