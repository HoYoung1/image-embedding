import torch
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(self, model, distance_measurer, scorer, batch_size=32, k_threshold=1):
        self.k_threshold = k_threshold
        self.batch_size = batch_size
        self.model = model
        self.distance_metric = distance_measurer
        self.scorer = scorer

    def evaluate(self, dataset):
        embeddings = []
        class_person = []
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for person_img, target in dataloader:
            embedding = self.model(person_img)
            embeddings.extend(embedding)
            class_person.extend(target)

        embeddings = torch.stack(embeddings)
        class_person = torch.stack(class_person)

        # clean up gpu memory
        del self.model
        torch.cuda.empty_cache()

        # Compute pairwise
        pairwise_distance = self.distance_metric(embeddings)

        score = self.scorer.score(pairwise_distance, class_person, self.k_threshold)
        return score
