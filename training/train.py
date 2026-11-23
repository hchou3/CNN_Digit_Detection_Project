import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.Dataset import DigitDataset, seed_worker
from training.models import ModelA, ModelB, count_parameters

MODEL_SETTINGS = {
    "a": (32, "crop", 1.0),
    "b": (64, "imagenet", 0.1),
}

def build_model(name):
    if name == "a":
        return ModelA()
    return ModelB(pretrained=True)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Runs one pass over a dataset. Trains if an optimizer is given.

    Args:
        model (nn.Module): the classifier.
        loader (DataLoader): batches of (crops, labels).
        loss_fn (nn.Module): loss function.
        device (torch.device): cpu or cuda.
        optimizer (torch.optim.Optimizer): None to only evaluate.

    Returns:
        tuple: two-element tuple containing:
            loss (float): average loss per crop.
            acc (float): share of crops classified correctly.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (scores.argmax(1) == y).sum().item()
            count += len(y)

    return total_loss / count, correct / count


def make_loader(path, train, batch_size, num_workers, size, norm):
    """Builds a DataLoader for one .npz file.

    Args:
        path (str): .npz file with X and y.
        train (bool): augment and shuffle if True.
        batch_size (int): crops per batch.
        num_workers (int): background processes loading batches.
        size (int): model input size.
        norm (str): "crop" or "imagenet".

    Returns:
        DataLoader: the loader.
    """
    dataset = DigitDataset(path, train=train, size=size, norm=norm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, worker_init_fn=seed_worker)

def make_optimizer(model, lr, feature_lr_factor):
    """SGD over the trainable weights. Pretrained feature layers get a lower rate.

    Args:
        model (nn.Module): the classifier.
        lr (float): learning rate for the head (and all of Model A).
        feature_lr_factor (float): multiplier for trainable layers in model.features.

    Returns:
        torch.optim.SGD: the optimizer.
    """
    head, features = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("features"):
            features.append(p)
        else:
            head.append(p)

    groups = [{"params": head, "lr": lr}]
    if features:
        groups.append({"params": features, "lr": lr * feature_lr_factor})
    return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=5e-4)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    size, norm, feature_lr_factor = MODEL_SETTINGS[args.model]
    train_loader = make_loader(os.path.join(args.data_dir, "train.npz"), True, args.batch_size, args.workers, size, norm)
    val_loader = make_loader(os.path.join(args.data_dir, "val.npz"), False, args.batch_size, args.workers, size, norm)
    test_loader = make_loader(os.path.join(args.data_dir, "test.npz"), False, args.batch_size, args.workers, size, norm)

    model = build_model(args.model).to(device)
    print("parameters:", count_parameters(model))

    # Model B stage 1: freeze the pretrained convolutions, train only the new head
    if args.freeze_epochs > 0:
        model.freeze_features()
        print("stage 1: head only, trainable parameters:", count_parameters(model))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr, feature_lr_factor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    run_name = "{}_lr{}_bs{}_ep{}".format(args.model, args.lr, args.batch_size, args.epochs)
    os.makedirs(os.path.join(args.out_dir, "history"), exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    history_path = os.path.join(args.out_dir, "history", run_name + ".csv")
    weights_path = os.path.join(args.weights_dir, run_name + ".pt")

    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
                                "test_loss", "test_acc", "seconds"])

    best_val_loss, best_epoch, bad_epochs = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):

        # Model B stage 2: also train the last convolution block, at a lower rate
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            model.unfreeze_last_block()
            optimizer = make_optimizer(model, args.lr, feature_lr_factor)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
            bad_epochs = 0
            print("stage 2: last block unfrozen, trainable parameters:", count_parameters(model))

        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        # Test is logged for the training curves only, never used to make decisions
        test_loss, test_acc = run_epoch(model, test_loader, loss_fn, device)
        seconds = time.time() - start

        print("epoch {}  lr {:.4g}  train_loss {:.4f}  train_acc {:.4f}  val_loss {:.4f}  "
              "val_acc {:.4f}  test_loss {:.4f}  test_acc {:.4f}  {:.0f}s".format(
                  epoch, lr, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, seconds))
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, lr, train_loss, train_acc, val_loss, val_acc,
                                    test_loss, test_acc, round(seconds, 1)])

        scheduler.step(val_loss)

        # Early stopping: only an improvement of at least 0.001 counts
        if val_loss < best_val_loss - 0.001:
            best_val_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save(model.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("stopping: no val_loss improvement for", args.patience, "epochs")
                break

    print("best epoch", best_epoch, "val_loss {:.4f}".format(best_val_loss), "saved", weights_path)

    # One summary row per run, for the comparison tables
    runs_path = os.path.join(args.out_dir, "runs.csv")
    new_file = not os.path.exists(runs_path)
    with open(runs_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["run", "model", "lr", "batch_size", "best_epoch", "best_val_loss", "parameters"])
        writer.writerow([run_name, args.model, args.lr, args.batch_size, best_epoch,
                         round(best_val_loss, 4), sum(p.numel() for p in model.parameters())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="a", choices=list(MODEL_SETTINGS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=os.path.join("data", "processed"))
    parser.add_argument("--out-dir", default=os.path.join("experiments", "results"))
    parser.add_argument("--weights-dir", default="weights")
    main(parser.parse_args())