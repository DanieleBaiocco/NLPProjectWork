from sklearn.metrics import f1_score


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