# This is a sample Python script.
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

from model.counting import ClassificationAndCounting
from model.sequence_tagging import BertForSequenceTagging
from preprocessing.data import DataProcessor, load_examples
from preprocessing.tokenization import ExtendedBertTokenizer
from train import train, evaluate


# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataprocessor = DataProcessor()
    print(dataprocessor.label_map)
    data_dir = "./data/neoplasm"
    max_seq_length = 510
    model_name_or_path = 'allenai/scibert_scivocab_uncased'
    do_lower_case = True
    base_tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
    tokenizer = ExtendedBertTokenizer(base_tokenizer)
    train_ds = load_examples(processor=dataprocessor, data_dir=data_dir, max_seq_length=max_seq_length,
                             tokenizer=tokenizer)
    val_ds = load_examples(processor=dataprocessor, data_dir=data_dir, max_seq_length=max_seq_length,
                          tokenizer=tokenizer, isval=True)
    test_ds = load_examples(processor=dataprocessor, data_dir=data_dir, max_seq_length=max_seq_length,
                            tokenizer=tokenizer, evaluate=True)

    num_labels = len(dataprocessor.get_labels())

    config = BertConfig.from_pretrained(
       model_name_or_path,
       num_labels=num_labels,
       finetuning_task="seqtag")
    model = BertForSequenceTagging.from_pretrained(
        model_name_or_path,
        config=config
     )
    model.to(device)
    #history = train(device= device, train_dataset= train_ds, model = model, eval_dataset= val_ds)
    #_ , test_metric = evaluate(device = device, model = model,eval_dataset= test_ds, eval_batch_size= 32)

    eval_dataloader = DataLoader(test_ds, batch_size=2)

    wrapper = ClassificationAndCounting(learner = model, processor = dataprocessor)
    for batch in eval_dataloader:
        wrapper.learner.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            outputs = wrapper.forward(**inputs)
            loss, emissions, path, proba_dist = outputs
            print(loss, proba_dist)




def compute_class_probabilities(predictions, num_classes=7):
    """
    Compute the probability distribution of each class for each sequence of predictions.

    Args:
        predictions (list of list of int): A list of sequences of predicted class indices.
        num_classes (int): The number of possible classes (default is 6, corresponding to classes 0 to 5).

    Returns:
        list of np.array: A list of probability distributions for each sequence.
    """
    all_probabilities = []

    for sequence in predictions:
        # Count occurrences of each class in the sequence
        counts = np.bincount(sequence, minlength=num_classes)
        print(counts)

        # Compute the probability distribution
        probabilities = counts / len(sequence)

        all_probabilities.append(probabilities)

    return all_probabilities


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
