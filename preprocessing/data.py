import collections
import os

import torch
from torch.utils.data import TensorDataset
from transformers import BasicTokenizer


class InputExample(object):
    def __init__(self, guid, text, labels=None):
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    def __init__(self):
        self.labels = ["X", "B-Claim", "I-Claim", "B-Premise", "I-Premise", 'O']
        self.label_map = self._create_label_map()
        self.replace_labels = {
            'B-MajorClaim': 'B-Claim',
            'I-MajorClaim': 'I-Claim',
        }

    def _create_label_map(self):
        label_map = collections.OrderedDict()
        for i, label in enumerate(self.labels):
            label_map[label] = i
        return label_map

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_conll(os.path.join(data_dir, "train.conll"), replace=self.replace_labels), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_conll(os.path.join(data_dir, "dev.conll"), replace=self.replace_labels), "dev")

    def get_test_examples(self, data_dir, setname="test.conll"):
        """See base class."""
        return self._create_examples(
            self.read_conll(os.path.join(data_dir, setname), replace=self.replace_labels), "test")

    def get_labels(self):
        """ See base class."""
        return self.labels

    def convert_labels_to_ids(self, labels):
        idx_list = []
        for label in labels:
            idx_list.append(self.label_map[label])
        return idx_list

    def convert_ids_to_labels(self, idx_list):
        labels_list = []
        for idx in idx_list:
            labels_list.append([key for key in self.label_map.keys() if self.label_map[key] == idx][0])
        return labels_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, str(i))
            text = line[0]
            labels = line[-1]
            examples.append(
                InputExample(guid=guid, text=text, labels=labels))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     cls_token='[CLS]',
                                     sep_token='[SEP]'):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a, labels = tokenizer.tokenize_with_label_extension(example.text, example.labels,
                                                                       copy_previous_label=True)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                labels = labels[:(max_seq_length - 2)]
            labels = ["X"] + labels + ["X"]

            tokens = [cls_token] + tokens_a + [sep_token]
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            label_ids = self.convert_labels_to_ids(labels)
            label_ids += padding
            assert len(label_ids) == max_seq_length
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
        return features

    @classmethod
    def features_to_dataset(cls, feature_list):
        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    @classmethod
    def read_conll(cls, input_file, token_number=0, token_column=1, label_column=4, replace=None):
        """Reads a conll type file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            sentences = []
            tokenizer = BasicTokenizer()
            sent_tokens = []
            sent_labels = []

            for idx, line in enumerate(lines):

                line = line.split('\t')
                # Skip the lines in which there is /n
                if len(line) < 2:
                    continue

                # Controllare se sono all'ID numero 1, assicurandomi nel mentre che NON sia la prima iterazione
                if idx != 0 and int(line[token_number]) == 0:
                    assert len(sent_tokens) == len(sent_labels)
                    if replace:
                        sent_labels = [replace[label] if label in replace.keys() else label for label in sent_labels]
                    sentences.append([' '.join(sent_tokens), sent_labels])
                    sent_tokens = []
                    sent_labels = []

                token = line[token_column]
                label = line[label_column].replace('\n', '')
                tokenized = tokenizer.tokenize(token)

                if len(tokenized) > 1:
                    for i in range(len(tokenized)):
                        sent_tokens.append(tokenized[i])
                        sent_labels.append(label)
                else:
                    sent_tokens.append(tokenized[0])
                    sent_labels.append(label)
            if sent_tokens != []:
                assert len(sent_tokens) == len(sent_labels)
                if replace:
                    sent_labels = [replace[label] if label in replace.keys() else label for label in sent_labels]
                sentences.append([' '.join(sent_tokens), sent_labels])
        return sentences


def load_examples(processor: DataProcessor, data_dir, max_seq_length, tokenizer, evaluate=False, isval=False):
    if evaluate:
        examples = processor.get_test_examples(data_dir)
    elif isval:
        examples = processor.get_dev_examples(data_dir)
    else:
        examples = processor.get_train_examples(data_dir)
    features = processor.convert_examples_to_features(examples, max_seq_length=max_seq_length, tokenizer=tokenizer)
    dataset = processor.features_to_dataset(features)
    return dataset
