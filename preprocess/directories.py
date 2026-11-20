import os


DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DEBUG_DIR = os.path.join("output", "debug")

TRAIN_LABELS_PATH = os.path.join(PROCESSED_DIR, "boxes_train.json")
TEST_LABELS_PATH = os.path.join(PROCESSED_DIR, "boxes_test.json")