import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup


def process_batch(batch, device, model, gen_preds_labels_fn, eval=False):
    model.eval() if eval else model.train()
    batch = tuple(t.to(device) for t in batch)
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3],
              "labels_proba": batch[4]}
    with torch.no_grad() if eval else torch.enable_grad():
        outputs = model(**inputs)
    preds, labels = gen_preds_labels_fn(inputs, outputs)
    return outputs, preds, labels


def evaluate(device, model, eval_dataset, eval_batch_size, metrics, gen_preds_labels_fn):
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)
    epoch_loss_sum = 0.0

    # Reset metrics before evaluation
    for metric in metrics:
        metric.reset()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            outputs, preds, labels = process_batch(batch, device, model, gen_preds_labels_fn, eval=True)
            loss = outputs[0]
            epoch_loss_sum += loss.item()

            # Update each metric with current batch's predictions and labels
            for metric in metrics:
                metric.update(preds, labels)

    eval_loss = epoch_loss_sum / len(eval_dataloader)

    # Compute all metrics
    eval_metrics = {metric.__class__.__name__: metric.compute().item() for metric in metrics}

    return eval_loss, eval_metrics


def train(
        device,
        train_dataset,
        model,
        eval_dataset,
        generate_preds_labels_fn,
        metrics,
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
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_loss_sum = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            outputs, preds, labels = process_batch(batch, device, model, generate_preds_labels_fn, eval=False)
            loss = outputs[0]
            loss.backward()
            epoch_loss_sum += loss.item()
            # Update metrics with current batch predictions and labels
            for metric in metrics:
                metric.update(preds, labels)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
        epoch_loss = epoch_loss_sum / num_steps_per_epoch
        history['train_loss'].append(epoch_loss)

        # Retrieve metric values after the epoch
        epoch_metrics = {metric.__class__.__name__: metric.compute().item() for metric in metrics}

        # Perform evaluation at the end of the epoch
        eval_loss, eval_metrics = evaluate(device, model, eval_dataset, eval_batch_size, metrics)

        # Append eval_loss to history
        history['eval_loss'].append(eval_loss)

        # Append eval_metrics to history
        for metric_name, metric_value in eval_metrics.items():
            history['eval_metrics'][metric_name].append(metric_value)

        # Log epoch summary
        print(f"Epoch {epoch + 1}/{num_train_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}")
        for metric_name, metric_value in epoch_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print(f"Eval Loss: {eval_loss:.4f}")
        for metric_name, metric_value in eval_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
    return history


def print_epoch_summary(epoch, epoch_loss, epoch_metric, eval_loss, eval_metric):
    print(f"\n{'=' * 50}")
    print(f"Epoch {epoch + 1} Summary")
    print(f"{'-' * 50}")
    print(f"Training Loss: {epoch_loss:.4f}")
    print("Training Metrics:")
    for key, value in epoch_metric.items():
        print(f"  {key} = {value:.4f}")

    print(f"\nValidation Loss: {eval_loss:.4f}")
    print("Validation Metrics:")
    for key, value in eval_metric.items():
        print(f"  {key} = {value:.4f}")
    print(f"{'=' * 50}\n")
