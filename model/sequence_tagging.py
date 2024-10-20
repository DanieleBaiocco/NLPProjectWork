import torch
from sklearn.base import BaseEstimator
from torch import nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel


class BertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.rnn = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.crf = CRF(config.num_labels, batch_first=True)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            labels_proba=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        rnn_out, _ = self.rnn(sequence_output)
        emissions = self.classifier(rnn_out)
        mask = attention_mask.bool()
        log_likelihood = self.crf(emissions, labels, mask=mask)
        path = self.crf.decode(emissions)
        path = torch.LongTensor(path)
        return -1 * log_likelihood, emissions, path


class BertForLabelDistribution(BertPreTrainedModel):
    def __init__(self, config, loss_fn):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.rnn = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        self.loss_fn = loss_fn

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            labels_proba=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        rnn_out, _ = self.rnn(sequence_output)  # [batch_size, seq_len, 2*hidden_size]
        emissions = self.classifier(rnn_out)  # [batch_size, seq_len, num_labels]
        cls_emissions = emissions[:, 0, :]  # [batch_size, num_labels]
        # posos calcolare la loss e dopo la softmax
        cls_probs = self.softmax(cls_emissions)  # [batch_size, num_labels]
        loss = self.loss_fn(cls_probs, labels_proba)
        return loss, emissions, None, cls_probs
