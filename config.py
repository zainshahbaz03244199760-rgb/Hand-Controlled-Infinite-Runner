""""
config.py
----------
Single place to define gesture classes, dataset filenames, and paths.
Edit this file if you rename files or add/remove gestures -- every other
script imports from here, so you only change it in one place.
"""

import os

# ---------------------------------------------------------------------------
# Gesture classes
# index : (csv_filename_without_extension, human_label, game_action)
# ---------------------------------------------------------------------------
GESTURES = {
    0: {"file": "closed_hand",    "label": "Closed Hand",   "action": "none"},
    1: {"file": "v_sign",         "label": "V Sign",        "action": "right"},
    2: {"file": "thumb_out",      "label": "Thumb Out",     "action": "left"},
    3: {"file": "opened_hand",    "label": "Opened Hand",   "action": "down"},
    4: {"file": "index_finger",   "label": "Index Finger",  "action": "up"},
    5: {"file": "three_fingers",  "label": "Three Fingers",  "action": "powerup"},
}

NUM_CLASSES = len(GESTURES)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

# Your dataset is ONE combined CSV with a header row:
#   x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
# (63 landmark feature columns + 1 label column = 64 columns total).
# Put your file at dataset/gesture_data.csv, or change the filename below.
DATASET_FILE = os.path.join(DATASET_DIR, "gesture_data.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

# ---------------------------------------------------------------------------
# Real-time inference tuning
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75   # ignore predictions below this confidence
SMOOTHING_WINDOW = 5          # majority vote over last N frames
GESTURE_COOLDOWN_SECONDS = 0.35  # min time between repeated game actions

os.makedirs(MODEL_DIR, exist_ok=True)
