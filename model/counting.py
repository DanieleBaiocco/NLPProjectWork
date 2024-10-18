import collections

import numpy as np

from preprocessing.data import DataProcessor


class ClassificationAndCounting:
    def __init__(self, learner, processor: DataProcessor):
        self.learner = learner
        self.labels = ["X", "B-Claim", "I-Claim", "B-Premise", "I-Premise", 'O']
        self.label_map = self._create_label_map()
        self.num_labels = len(processor.labels)

    def _create_label_map(self):
        label_map = collections.OrderedDict()
        for i, label in enumerate(self.labels):
            label_map[label] = i
        return label_map

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        loss, emissions, path = self.learner(input_ids, attention_mask, token_type_ids, labels)
        proba_dists = self.aggregate(path, attention_mask)
        return loss, emissions, path, proba_dists

    def aggregate(self, predictions, attention_mask):
        all_probabilities = []
        for sequence, mask in zip(predictions, attention_mask):
            # Count occurrences of each class in the sequence
            sequence = sequence[mask == 1][1:]
            # Qui dovresti avere una sequenza SENZA la classe 0 se tutto e' andato al meglio
            # Per questo fai MENO UNO, perche' in questo modo hai le classi SENZA la classe X.
            # Posso ora mergiare in posizione 0 (classe corrispondente a B-CLAIM) con la posizione 1 (classe corrispondente a I-CLAIM)
            # Nota che senza il -1, ho che B-Claim e' legato al numero 1.
            #sequence = sequence - 1

            counts = np.bincount(sequence, minlength=self.num_labels)
            # Compute the probability distribution
            probabilities = counts / len(sequence)
            probabilities = self.merge_BI(probabilities)
            all_probabilities.append(probabilities)
        return np.array(all_probabilities)

    def merge_BI(self, probabilities):
        b_claim_idx = self.label_map["B-Claim"]
        i_claim_idx = self.label_map["I-Claim"]
        probabilities = self._combine_probas(probabilities, b_claim_idx, i_claim_idx)
        b_evidence_idx = self.label_map["B-Premise"]
        i_evidence_idx = self.label_map["I-Premise"]
        probabilities = self._combine_probas(probabilities, b_evidence_idx, i_evidence_idx)
        return probabilities

    def _combine_probas(self, arr, idx1, idx2):
        # Ensure the indices are valid and different
        if idx1 == idx2:
            raise ValueError("Indices must be different")

        # Copy the array to avoid modifying the original
        new_arr = np.array(arr)

        # Sum the values at idx1 and idx2
        combined_value = new_arr[idx1] + new_arr[idx2]

        # Remove the higher index first to maintain valid positions
        new_arr = np.delete(new_arr, idx2)  # Delete idx2 value
        new_arr[idx1] = combined_value  # Assign the sum to idx1
        return new_arr