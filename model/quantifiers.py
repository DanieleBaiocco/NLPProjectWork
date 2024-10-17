import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier, ACC, CC

from model.sequence_tagging import BertForSequenceTagging
from preprocessing.data import DataProcessor


class BertForSequenceTaggingCaC(CC):
    def __init__(self, classifier: BertForSequenceTagging, processor: DataProcessor):
        super().__init__(classifier=classifier)

        self.processor = processor

    @property
    def classes_(self):
        return np.array(["X", "I-Claim", "I-Premise", 'O'])

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        return super().fit(data, False, None)

    def classify(self, instances):
        return self.classifier(**instances)

    def classifier_fit_predict(self, data: LabelledCollection, fit_classifier=True, predict_on=None):
        return self.classifier(data)

    def aggregate(self, classif_predictions: np.ndarray):
        bclaimidx = self.processor.label_map["B-Claim"]
        iclaimidx = self.processor.label_map["I-Claim"]
        labels_predictions = [np.int64(iclaimidx) if label == bclaimidx else label for label in classif_predictions[2].detach().numpy().flatten()]
        bevidenceidx = self.processor.label_map["B-Premise"]
        ievidenceidx = self.processor.label_map["I-Premise"]
        labels_predictions = [np.int64(ievidenceidx) if label == bevidenceidx else label for label in labels_predictions]
        super().aggregate(np.array(labels_predictions))
