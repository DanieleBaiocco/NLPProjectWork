# This is a sample Python script.
import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

from data.metrics.metric import acc_and_f1, kldiv_loss
from model.counting import ClassificationAndCounting
from model.sequence_tagging import BertForSequenceTagging, BertForLabelDistribution
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
        num_labels=num_labels)
    model = BertForSequenceTagging.from_pretrained(
        model_name_or_path,
        config=config
    )
    model.to(device)
    #wrapper = ClassificationAndCounting(learner=model, processor=dataprocessor)
    #metrics = [torchmetrics.Accuracy(num_classes=3, average='macro', task = "multiclass"),
    #       torchmetrics.F1Score(num_classes=3, average='macro', task = "multiclass")]
    #history = train(device=device, train_dataset=train_ds, model=wrapper, eval_dataset=val_ds,
    #                generate_preds_labels_fn=classification_preds_labels_fn, metrics=metrics)

    # _ , test_metric = evaluate(device = device, model = model,eval_dataset= test_ds, eval_batch_size= 32)
    config.num_labels = 3
    qua_model = BertForLabelDistribution.from_pretrained(model_name_or_path, config=config, loss_fn = kldiv_loss)
    qua_model.to(device)
    metrics = [torchmetrics.KLDivergence(log_prob=False, reduction="mean"),
               torchmetrics.MeanAbsoluteError(),
               torchmetrics.MeanSquaredError()]
    train(device=device, train_dataset=train_ds, model=qua_model, eval_dataset=val_ds,
                    generate_preds_labels_fn=proba_preds_labels_fn, metrics=metrics)


def proba_preds_labels_fn(inputs, outputs):
    _, _, _, pred_proba = outputs
    pred_proba = pred_proba.detach().cpu().numpy()
    labels_proba = inputs["labels_proba"]
    return pred_proba, labels_proba


def classification_preds_labels_fn(inputs, outputs):
    loss, emissions, path, _ = outputs
    batch_logits = path.detach().cpu().numpy().flatten()
    batch_labels = inputs["labels"].detach().cpu().numpy().flatten()
    attention_mask = inputs["attention_mask"].detach().cpu().numpy().flatten()
    valid_batch_logits = batch_logits[attention_mask == 1]
    valid_batch_labels = batch_labels[attention_mask == 1]
    return valid_batch_logits, valid_batch_labels


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print(torch.__version__)
    main()

# QUINDI COME NEXT STEP HO:
"""
1. Fai in modo che il tuo modello sia trainabile con lo stesso codice di adesso, ma usando la mae/kldiv tra predizione e true label
   per generare una metrica di valutazione.
2. Implementa un altro modello che usa un BertModel finetunnato sulla task di dare come output una probabilita' di distribuzione (all'altezza 
   del CLS token forse)
3. Implementa un modo per avere delle LOSSES DIFFERENTI, che sono sempre specificabili dall'esterno (usando lo stesso train loop). Come loss posso infatti
   avere mae/kldiv/mse/jddiv ecc..
4. Prova a inventarti una rete che abbia una loss multipla (che considera sia la predizione sulle labels che la probabilita' di distribuzione) HARD, solo se e' richiesto
5. Implementa un modo per salvare le weights
6. Prima di runnare il tuo codice, fai in modo che sia FINITO, che ci sia TUTTO all'interno, quindi ora impegnati a finirlo.
"""
