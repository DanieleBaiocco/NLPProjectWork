import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup

from data.metrics.metric import acc_and_f1


def evaluate(device, model, eval_dataset, eval_batch_size):
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)
    epoch_loss_sum = 0.0
    epoch_logits, epoch_labels = [], []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            outputs = model(**inputs)
            loss, emissions, path = outputs
            epoch_loss_sum += loss.item()
            batch_logits = path.detach().cpu().numpy().flatten()
            batch_labels = inputs["labels"].detach().cpu().numpy().flatten()
            attention_mask = inputs["attention_mask"].detach().cpu().numpy().flatten()
        # Filter out padding tokens (where attention_mask == 1)
        valid_batch_logits = batch_logits[attention_mask == 1]
        valid_batch_labels = batch_labels[attention_mask == 1]
        epoch_logits = np.append(valid_batch_logits, batch_logits, axis=0)
        epoch_labels = np.append(valid_batch_labels, batch_labels, axis=0)
    eval_loss = epoch_loss_sum / len(eval_dataset)
    eval_metric = acc_and_f1(epoch_logits, epoch_labels)
    return eval_loss, eval_metric


def train(
        device,
        train_dataset,
        model,
        eval_dataset,
        num_train_epochs=1,
        train_batch_size=4,
        eval_batch_size=32,
        weight_decay=0.3,
        learning_rate=2e-5,
        adam_epsilon=1e-8,
        warmup_steps=0
):
    history = {
        'train_loss': [],
        'train_metrics': [],
        'eval_loss': [],
        'eval_metrics': []
    }
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)
    num_steps_per_epoch = len(train_dataloader)
    t_total = num_steps_per_epoch * num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_loss_sum = 0.0
        epoch_logits, epoch_labels = [], []
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            outputs = model(**inputs)
            loss, emissions, path = outputs
            loss.backward()
            epoch_loss_sum += loss.item()
            batch_logits = path.detach().cpu().numpy().flatten()
            batch_labels = inputs["labels"].detach().cpu().numpy().flatten()
            attention_mask = inputs["attention_mask"].detach().cpu().numpy().flatten()
            # Filter out padding tokens (where attention_mask == 1)
            valid_batch_logits = batch_logits[attention_mask == 1]
            valid_batch_labels = batch_labels[attention_mask == 1]
            epoch_logits = np.append(valid_batch_logits, batch_logits, axis=0)
            epoch_labels = np.append(valid_batch_labels, batch_labels, axis=0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        epoch_loss = tr_loss / num_steps_per_epoch
        epoch_metric = acc_and_f1(epoch_logits, epoch_labels)
        eval_loss, eval_metric = evaluate(device=device, model=model, eval_dataset=eval_dataset,
                                          eval_batch_size=eval_batch_size)
        history['train_loss'].append(epoch_loss)
        history['train_metrics'].append(epoch_metric)
        history['eval_loss'].append(eval_loss)
        history['eval_metrics'].append(eval_metric)
        print_epoch_summary(epoch, epoch_loss, epoch_metric, eval_loss, eval_metric)
    return history


def print_epoch_summary(epoch, epoch_loss, epoch_metric, eval_loss, eval_metric):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1} Summary")
    print(f"{'-'*50}")
    print(f"Training Loss: {epoch_loss:.4f}")
    print(f"Training Metrics: Accuracy = {epoch_metric['acc']:.4f}, "
          f"F1 Micro = {epoch_metric['eval_f1_micro']:.4f}, "
          f"F1 Macro = {epoch_metric['eval_f1_macro']:.4f}, "
          f"F1 Claim = {epoch_metric['f1_claim']:.4f}, "
          f"F1 Evidence = {epoch_metric['f1_evidence']:.4f}")
    print(f"Validation Loss: {eval_loss:.4f}")
    print(f"Validation Metrics: Accuracy = {eval_metric['acc']:.4f}, "
          f"F1 Micro = {eval_metric['eval_f1_micro']:.4f}, "
          f"F1 Macro = {eval_metric['eval_f1_macro']:.4f}, "
          f"F1 Claim = {eval_metric['f1_claim']:.4f}, "
          f"F1 Evidence = {eval_metric['f1_evidence']:.4f}")
    print(f"{'='*50}\n")