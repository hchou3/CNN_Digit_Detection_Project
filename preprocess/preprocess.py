import cv2
import os
import numpy as np
import json

from preprocess import TRAIN_DIR, TEST_DIR, PROCESSED_DIR, DEBUG_DIR, TRAIN_LABELS_PATH, TEST_LABELS_PATH


def crop_digit(image, box, size=(32, 32), margin=0.15):
    side = int(round(max(box["width"], box["height"]) * (1 + margin)))
    cx = box["left"] + box["width"] // 2
    cy = box["top"] + box["height"] // 2

    padded = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_REPLICATE)
    x0 = cx - side // 2 + side
    y0 = cy - side // 2 + side
    crop = padded[y0:y0 + side, x0:x0 + side]

    interp = cv2.INTER_AREA if side > size[0] else cv2.INTER_LINEAR
    return cv2.resize(crop, size, interpolation=interp)

def save_crop_preview(image_dir, labels, num_images, filename):
    crops = []
    for i in range(1, num_images + 1):
        name = "{}.png".format(i)
        image = cv2.imread(os.path.join(image_dir, name))
        for box in labels[name]:
            crop = crop_digit(image, box)
            print(name, "label", box["label"], "box", box["width"], "x", box["height"], "->", crop.shape)
            crops.append(crop)

    strip = np.hstack(crops)
    strip = cv2.resize(strip, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, filename), strip)

def build_digit_crops(image_dir, labels):
    X, y, names = [], [], []
    for name, boxes in labels.items():
        image = cv2.imread(os.path.join(image_dir, name))
        for box in boxes:
            X.append(crop_digit(image, box))
            y.append(box["label"])
            names.append(name)
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64), names

def iou(box_a, box_b):
    """IoU of two boxes.

    Args:
        box_a (dict): box with left, top, width and height values.
        box_b (dict): box with left, top, width and height values.

    Returns:
        float: overlap between 0.0 (none) and 1.0 (identical).
    """
    ax0, ay0 = box_a["left"], box_a["top"]
    ax1, ay1 = ax0 + box_a["width"], ay0 + box_a["height"]
    bx0, by0 = box_b["left"], box_b["top"]
    bx1, by1 = bx0 + box_b["width"], by0 + box_b["height"]

    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    union = box_a["width"] * box_a["height"] + box_b["width"] * box_b["height"] - inter
    return inter / union

def check_overlap(box, digit):
    """Fraction of a digit box that lies inside another box.

    Args:
        box (dict): candidate box with left, top, width and height values.
        digit (dict): digit box with left, top, width and height values.

    Returns:
        float: 0.0 (digit not inside) to 1.0 (digit fully inside).
    """
    inter_w = max(0, min(box["left"] + box["width"], digit["left"] + digit["width"]) - max(box["left"], digit["left"]))
    inter_h = max(0, min(box["top"] + box["height"], digit["top"] + digit["height"]) - max(box["top"], digit["top"]))
    return inter_w * inter_h / (digit["width"] * digit["height"])

def sample_negatives(img, boxes, num_samples=10, threshold=0.3, min_edges=0.03, tries=100):
    """Sample negative crops centered on edges in the image.

    Args:
        img (np.ndarray): image to sample from.
        boxes (list): list of digit boxes to avoid.
        num_samples (int, optional): number of negatives to sample. Defaults to 10.
        threshold (float, optional): maximum fraction of any digit inside the crop,
                                     and of the crop lying on any digit. Defaults to 0.3.
        min_edges (float, optional): minimum fraction of edge pixels in the crop.
                                     Defaults to 0.10.
        tries (int, optional): maximum number of attempts. Defaults to 100.

    Returns:
        tuple: two-element tuple containing:
            crops (list): negative crops, each 32x32x3.
            kept (list): the boxes the crops were cut from.
    """
    h, w = img.shape[:2]
    mean_h = np.mean([box["height"] for box in boxes])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    for b in boxes:
        edges[max(b["top"], 0):b["top"] + b["height"], max(b["left"], 0):b["left"] + b["width"]] = 0
        
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return [], []

    crops, kept = [], []
    for _ in range(tries):
        if len(crops) == num_samples:
            break

        # Square size near the digit size, never tiny compared to the digits
        side = int(round(np.random.normal(mean_h, mean_h / 2)))
        side = max(side, int(0.5 * mean_h), 8)
        side = min(side, h, w)

        # Center the square on a random edge pixel, kept inside the image
        i = np.random.randint(len(xs))
        left = int(np.clip(xs[i] - side // 2, 0, w - side))
        top = int(np.clip(ys[i] - side // 2, 0, h - side))
        box = {"left": left, "top": top, "width": side, "height": side}

        density = np.mean(edges[top:top + side, left:left + side] > 0)
        if density < min_edges:
            continue

        # crop_digit adds a 15% margin, so check the area that will actually be cropped
        grow = int(round(side * 0.15 / 2))
        crop_area = {"left": left - grow, "top": top - grow,
                     "width": side + 2 * grow, "height": side + 2 * grow}

        digit_inside = max(check_overlap(crop_area, b) for b in boxes)   # share of a digit in the crop
        crop_on_digit = max(check_overlap(b, crop_area) for b in boxes)  # share of the crop on a digit
        if digit_inside < threshold and crop_on_digit < threshold:
            if all(iou(box, k) < 0.5 for k in kept):
                kept.append(box)
                crops.append(crop_digit(img, box))

    return crops, kept

def build_negative_samples(image_dir, labels, seed=0):
    """Crops edge-based negatives from every image, one per digit.

    Args:
        image_dir (str): folder with the full images.
        labels (dict): image name -> list of digit boxes.
        seed (int): random seed, so the same crops are made every run.

    Returns:
        tuple: four-element tuple containing:
            X (numpy.array): crops, shape (N, 32, 32, 3), uint8.
            y (numpy.array): labels, all 10, shape (N,).
            names (list): image name each crop came from.
            neg_boxes (list): box each crop was cut from.
    """
    np.random.seed(seed)
    X, names, neg_boxes = [], [], []
    for name, boxes in labels.items():
        image = cv2.imread(os.path.join(image_dir, name))
        crops, kept = sample_negatives(image, boxes, num_samples=len(boxes))
        X.extend(crops)
        names.extend([name] * len(crops))
        neg_boxes.extend(kept)
    X = np.array(X, dtype=np.uint8)
    y = np.full(len(X), 10, dtype=np.int64)
    return X, y, names, neg_boxes

def save_negative_check(image_dir, labels, names, neg_boxes, image_name):
    """Draws digit boxes (green) and negative boxes (red) on one image.

    Args:
        image_dir (str): folder with the full images.
        labels (dict): image name -> list of digit boxes.
        names (list): image name for each negative, from build_negative_samples.
        neg_boxes (list): box for each negative, from build_negative_samples.
        image_name (str): image to draw, e.g. "9.png".

    Returns:
        None.
    """
    vis = cv2.imread(os.path.join(image_dir, image_name))
    for b in labels[image_name]:
        cv2.rectangle(vis, (b["left"], b["top"]),
                      (b["left"] + b["width"], b["top"] + b["height"]), (0, 255, 0), 1)
    for name, b in zip(names, neg_boxes):
        if name == image_name:
            cv2.rectangle(vis, (b["left"], b["top"]),
                          (b["left"] + b["width"], b["top"] + b["height"]), (0, 0, 255), 1)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, "neg_check_" + image_name), vis)

def split_by_image(names, val_fraction=0.1, seed=0):
    """Splits crops into train and validation so each image goes to one side only.

    Args:
        names (list): image name for each crop.
        val_fraction (float): share of images used for validation.
        seed (int): random seed, so the split is the same every run.

    Returns:
        tuple: two-element tuple containing:
            train_mask (numpy.array): True for crops in the training set.
            val_mask (numpy.array): True for crops in the validation set.
    """
    images = sorted(set(names), key=lambda n: int(n.split(".")[0]))
    np.random.seed(seed)
    order = np.random.permutation(len(images))
    n_val = int(len(images) * val_fraction)
    val_images = set(images[i] for i in order[:n_val])

    val_mask = np.array([n in val_images for n in names])
    return ~val_mask, val_mask

if __name__ == "__main__":
    # Preview: uncomment to check a few train crops by eye
    # with open(TRAIN_LABELS_PATH) as f:
    #     train_labels = json.load(f)
    # save_crop_preview(TRAIN_DIR, train_labels, 10, "crops_preview.png")
    
    # Test
    with open(TEST_LABELS_PATH) as f:
        test_labels = json.load(f)

    X_test, y_test, test_names = build_digit_crops(TEST_DIR, test_labels)
    X_neg, y_neg, neg_names, neg_boxes = build_negative_samples(TEST_DIR, test_labels)
    X_test = np.concatenate([X_test, X_neg])
    y_test = np.concatenate([y_test, y_neg])
    print("test", X_test.shape, np.bincount(y_test))

    # Train
    with open(TRAIN_LABELS_PATH) as f:
        train_labels = json.load(f)

    X_train, y_train, train_names = build_digit_crops(TRAIN_DIR, train_labels)
    X_neg, y_neg, neg_names, neg_boxes = build_negative_samples(TRAIN_DIR, train_labels)
    X_train = np.concatenate([X_train, X_neg])
    y_train = np.concatenate([y_train, y_neg])
    train_names = train_names + neg_names
    print("train", X_train.shape, np.bincount(y_train))

    strip = np.hstack(X_neg[:20])
    strip = cv2.resize(strip, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, "negatives_preview_train.png"), strip)

    train_mask, val_mask = split_by_image(train_names)
    X_val, y_val = X_train[val_mask], y_train[val_mask]
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    print("train", X_train.shape, np.bincount(y_train))
    print("val", X_val.shape, np.bincount(y_val))

    np.savez_compressed(os.path.join(PROCESSED_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(PROCESSED_DIR, "val.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(PROCESSED_DIR, "test.npz"), X=X_test, y=y_test)

