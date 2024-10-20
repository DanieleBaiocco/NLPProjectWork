import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1_micro = f1_score(labels, preds, labels=[1, 2, 3, 4, 5], average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_claim = f1_score(labels, preds, labels=[1, 2], average='micro')
    f1_evidence = f1_score(labels, preds, labels=[3, 4], average='micro')
    return {
        "acc": acc,
        'eval_f1_micro': f1_micro,
        'eval_f1_macro': f1_macro,
        'f1_claim': f1_claim,
        'f1_evidence': f1_evidence,
    }


def mse_loss(pred_probas, label_probas):
    loss = nn.MSELoss(reduction="mean")
    return loss(pred_probas, label_probas)


def mae_loss(pred_probas, label_probas):
    loss = nn.L1Loss(reduction="mean")
    return loss(pred_probas, label_probas)


def kldiv_loss(pred_probas, label_probas):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    pred_probas_log = torch.log(pred_probas)
    output = kl_loss(pred_probas_log, label_probas)
    return output
