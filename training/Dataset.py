import os

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from preprocess.directories import PROCESSED_DIR, DEBUG_DIR

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))

def normalize_crop(crop):
        """Converts a BGR uint8 crop into a normalized float tensor.

        Args:
            crop (numpy.array): color BGR uint8 crop, shape (h, w, 3).

        Returns:
            torch.Tensor: float32 tensor, shape (3, h, w), RGB order.
        """
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - rgb.mean()) / (rgb.std() + 1e-6)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))

def augment(crop, max_angle=30, max_shift=2, max_noise=20):
        """Applies a random rotation, shift, lighting change, noise and blur.

        Args:
            crop (numpy.array): color BGR uint8 crop, shape (h, w, 3).
            max_angle (float): largest rotation in degrees, either direction.
            max_shift (float): largest shift in pixels, either direction.
            max_noise (float): largest Gaussian noise standard deviation.

        Returns:
            numpy.array: augmented BGR uint8 crop, same shape.
        """
        h, w = crop.shape[:2]

        angle = np.random.uniform(-max_angle, max_angle)
        scale = np.random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += np.random.uniform(-max_shift, max_shift, 2)
        out = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        out = out.astype(np.float32)
        out = out * np.random.uniform(0.6, 1.4) + np.random.uniform(-30, 30)
        out = np.clip(out, 0, 255)
        out = 255.0 * (out / 255.0) ** np.random.uniform(0.7, 1.5)

        if np.random.rand() < 0.5:
            out = out + np.random.normal(0, np.random.uniform(0, max_noise), out.shape)
        if np.random.rand() < 0.2:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        return np.clip(out, 0, 255).astype(np.uint8)

class DigitDataset(Dataset):
    def __init__(self, path, train=False, size=32, norm="crop"):
        data = np.load(path)
        self.X, self.y = data["X"], data["y"]
        self.train = train
        self.size = size
        self.norm = norm

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        crop = self.X[index]
        if self.train:
            crop = augment(crop)
        if self.size != crop.shape[0]:
            crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if self.norm == "imagenet":
            return normalize_imagenet(crop), int(self.y[index])
        return normalize_crop(crop), int(self.y[index])

# Helpers
def seed_worker(worker_id):
    """Gives each DataLoader worker its own NumPy seed.

    Without this, workers can produce the same random augmentations.
    Pass as worker_init_fn when num_workers > 0.
    """
    np.random.seed(torch.initial_seed() % 2 ** 32)


def save_augment_preview(dataset, num_crops, filename):
    """Saves original crops (top row) and augmented versions (bottom row).

    Args:
        dataset (DigitDataset): dataset to take crops from.
        num_crops (int): number of crops to show.
        filename (str): output image name inside DEBUG_DIR.

    Returns:
        None.
    """
    originals = [dataset.X[i] for i in range(num_crops)]
    augmented = [augment(c) for c in originals]    
    grid = np.vstack([np.hstack(originals), np.hstack(augmented)])
    grid = cv2.resize(grid, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, filename), grid)

if __name__ == "__main__":
    np.random.seed(0)
    train_set = DigitDataset(os.path.join(PROCESSED_DIR, "train.npz"), train=True)
    print("train crops:", len(train_set))

    x, label = train_set[0]
    print("one item:", tuple(x.shape), x.dtype, "label", label)
    print("mean", round(float(x.mean()), 3), "std", round(float(x.std()), 3))

    save_augment_preview(train_set, 12, "augment_preview.png")
