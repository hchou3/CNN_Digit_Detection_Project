"""
Charts and Graphs produced on Kaggle
"""

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
from training.models import ModelA

RUN_NAME = "a_lr0.01_bs128"
HISTORY_PATH = os.path.join("experiments", "results", "history", RUN_NAME + ".csv")
WEIGHTS_PATH = os.path.join("weights", RUN_NAME + ".pt")
TEST_PATH = os.path.join("data", "processed", "test.npz")
CHART_DIR = os.path.join("experiments", "results", "charts")
TABLE_DIR = os.path.join("experiments", "results", "tables")

CLASS_NAMES = [str(d) for d in range(10)] + ["bg"]


def read_history(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def plot_curves(history, filename):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for split in ["train", "val", "test"]:
        axes[0].plot(epochs, 1 - history[split + "_acc"], label=split)
        axes[1].plot(epochs, history[split + "_loss"], label=split)

    axes[0].set_title("Model A error")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("error (1 - accuracy)")
    axes[0].legend()
    axes[1].set_title("Model A loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, filename), dpi=120)
    plt.close(fig)


def predict(model, path, batch_size=512):
    loader = DataLoader(DigitDataset(path, train=False), batch_size=batch_size, shuffle=False)
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


def plot_confusion(y_true, y_pred, filename):
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
    ax.set_title("Model A test confusion matrix (row normalized)")

    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, filename), dpi=120)
    plt.close(fig)


def save_scores(y_true, y_pred, filename): 
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(11)), zero_division=0)

    with open(os.path.join(TABLE_DIR, filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "count"])
        for i in range(11):
            writer.writerow([CLASS_NAMES[i], round(precision[i], 4), round(recall[i], 4),
                             round(f1[i], 4), int(support[i])])


if __name__ == "__main__":
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # Training curves
    history = read_history(HISTORY_PATH)
    plot_curves(history, "modelA_curves.png")
    print("saved modelA_curves.png")

    # Test predictions on CPU
    model = ModelA()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    y_true, y_pred, seconds = predict(model, TEST_PATH)

    plot_confusion(y_true, y_pred, "modelA_confusion.png")
    save_scores(y_true, y_pred, "modelA_test_scores.csv")
    print("saved modelA_confusion.png and modelA_test_scores.csv")

    # Summary scores
    digits = y_true < 10
    print("test crops:", len(y_true))
    print("overall accuracy:", round(np.mean(y_true == y_pred), 4))
    print("digit accuracy:", round(np.mean(y_true[digits] == y_pred[digits]), 4))
    print("background accuracy:", round(np.mean(y_true[~digits] == y_pred[~digits]), 4))
    print("macro F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
    print("CPU time per 1000 crops: {:.2f} s".format(1000 * seconds / len(y_true)))