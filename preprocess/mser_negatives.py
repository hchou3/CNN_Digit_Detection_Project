import json
import os

import cv2


from directories import TRAIN_DIR, PROCESSED_DIR, DEBUG_DIR, TEST_LABELS_PATH, TRAIN_LABELS_PATH

def mser_boxes(gray):
    mser = cv2.MSER_create()
    _, bboxes = mser.detectRegions(gray)

    boxes = []
    for x, y, w, h in bboxes:
        boxes.append({"left": int(x), "top": int(y), "width": int(w), "height": int(h)})
    return boxes

if __name__ == "__main__":
    with open(TRAIN_LABELS_PATH) as f:
        train_labels = json.load(f)

    name = "1.png"
    image = cv2.imread(os.path.join(TRAIN_DIR, name))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes = mser_boxes(gray)
    print(name, "MSER boxes:", len(boxes))

    vis = image.copy()
    for b in boxes:
        cv2.rectangle(vis, (b["left"], b["top"]),
                      (b["left"] + b["width"], b["top"] + b["height"]), (255, 0, 0), 1)
    for b in train_labels[name]:
        cv2.rectangle(vis, (b["left"], b["top"]),
                      (b["left"] + b["width"], b["top"] + b["height"]), (0, 255, 0), 2)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, "mser_" + name), vis)