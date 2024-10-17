# This is a sample Python script.
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from model.quantifiers import BertForSequenceTaggingCaC
from model.sequence_tagging import BertForSequenceTagging
from preprocessing.data import DataProcessor, load_examples
from preprocessing.tokenization import ExtendedBertTokenizer
from train import train, evaluate


# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataprocessor = DataProcessor()
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
    #history = train(device= device, train_dataset= train_ds, model = model, eval_dataset= val_ds, )
    #_ , test_metric = evaluate(device = device, model = model,eval_dataset= test_ds, eval_batch_size= 32)
    bertCaCCC= BertForSequenceTaggingCaC(model, processor=dataprocessor)

    # VEDERE COSA PRENDE IN INPUT PRECISAMENTE IL TUO MODELLO QUANDO FA TRAINING (credo prenda un dizionario e gli va bene cois)
    test_dataloader = DataLoader(test_ds, batch_size=1)
    #bertCaCCC.fit(train_ds)
    for batch in test_dataloader:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
        res = bertCaCCC.quantify(inputs)
        break



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
