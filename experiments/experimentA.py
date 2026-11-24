"""
Charts and scores for Model A or Model B, produced on Kaggle.

Example usage:
    python -m experiments.experiment --model a --run a_lr0.01_bs128_ep50
    python -m experiments.experiment --model b --run b_lr0.01_bs128_ep30
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from training.Dataset import DigitDataset
from training.models import ModelA, ModelB

CLASS_NAMES = [str(d) for d in range(10)] + ["bg"]

MODEL_SETTINGS = {
    "a": (ModelA, 32, "crop"),
    "b": (ModelB, 64, "imagenet"),
}


def read_history(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def plot_curves(history, title, out_path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for split in ["train", "val", "test"]:
        axes[0].plot(epochs, 1 - history[split + "_acc"], label=split)
        axes[1].plot(epochs, history[split + "_loss"], label=split)

    axes[0].set_title(title + " error")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("error (1 - accuracy)")
    axes[0].legend()
    axes[1].set_title(title + " loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def predict(model, path, size, norm, batch_size=256):
    dataset = DigitDataset(path, train=False, size=size, norm=norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    y_true, y_pred, seconds = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            start = time.time()
            scores = model(x)
            seconds += time.time() - start
            y_true.append(y.numpy())
            y_pred.append(scores.argmax(1).numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), seconds


def plot_confusion(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(11)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    for i in range(11):
        for j in range(11):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, "{:.2f}".format(cm_norm[i, j]), ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title + " test confusion matrix (row normalized)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_scores(y_true, y_pred, out_path):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(11)), zero_division=0)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "count"])
        for i in range(11):
            writer.writerow([CLASS_NAMES[i], round(precision[i], 4), round(recall[i], 4),
                             round(f1[i], 4), int(support[i])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="a", choices=list(MODEL_SETTINGS))
    parser.add_argument("--run", required=True)
    parser.add_argument("--history-dir", default=os.path.join("experiments", "results", "history"))
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--test", default=os.path.join("data", "processed", "test.npz"))
    parser.add_argument("--out-dir", default=os.path.join("experiments", "results"))
    args = parser.parse_args()

    model_class, size, norm = MODEL_SETTINGS[args.model]
    title = "Model " + args.model.upper()
    os.makedirs(args.out_dir, exist_ok=True)

    # Training curves
    history = read_history(os.path.join(args.history_dir, args.run + ".csv"))
    plot_curves(history, title, os.path.join(args.out_dir, args.run + "_curves.png"))

    # Test predictions on CPU
    model = model_class()
    model.load_state_dict(torch.load(os.path.join(args.weights_dir, args.run + ".pt"), map_location="cpu"))
    y_true, y_pred, seconds = predict(model, args.test, size, norm)

    plot_confusion(y_true, y_pred, title, os.path.join(args.out_dir, args.run + "_confusion.png"))
    save_scores(y_true, y_pred, os.path.join(args.out_dir, args.run + "_test_scores.csv"))
    print("saved", args.run + "_curves.png,", args.run + "_confusion.png,", args.run + "_test_scores.csv")

    # Summary scores
    digits = y_true < 10
    print("test crops:", len(y_true))
    print("overall accuracy:", round(np.mean(y_true == y_pred), 4))
    print("digit accuracy:", round(np.mean(y_true[digits] == y_pred[digits]), 4))
    print("background accuracy:", round(np.mean(y_true[~digits] == y_pred[~digits]), 4))
    print("macro F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
    print("CPU time per 1000 crops: {:.2f} s".format(1000 * seconds / len(y_true)))