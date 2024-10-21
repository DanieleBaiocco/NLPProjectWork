import collections

import numpy as np

from preprocessing.data import DataProcessor


class ClassificationAndCounting:
    def __init__(self, learner, processor: DataProcessor):
        self.learner = learner
        self.processor = processor
        self.name = "bert_classify_and_count"

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            labels_proba=None
    ):
        loss, emissions, path = self.learner(input_ids, attention_mask, token_type_ids, labels, labels_proba)
        proba_dists = self.aggregate(path, attention_mask)
        return loss, emissions, path, proba_dists

    def aggregate(self, predictions, attention_mask):
        all_probabilities = []
        for sequence, mask in zip(predictions, attention_mask):
            label_proba = self.processor.compute_label_probadist(sequence, mask=mask)
            all_probabilities.append(label_proba)
        return np.array(all_probabilities)

    # Wrapper methods mimicking PyTorch model behavior
    def train(self, mode=True):
        self.learner.train(mode)

    def eval(self):
        self.learner.eval()

    def state_dict(self, *args, **kwargs):
        return self.learner.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.learner.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.learner.to(*args, **kwargs)

    def parameters(self):
        return self.learner.parameters()

    def named_parameters(self):
        return self.learner.named_parameters()

    def zero_grad(self):
        self.learner.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
