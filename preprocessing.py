"""
preprocessing.py
-----------------
Shared feature-extraction / normalization code.

CRITICAL: this exact same logic must run during:
  1) dataset creation   (data_collection.py)
  2) model training     (train_model.py)
  3) real-time inference (realtime_classifier.py / main.py)

If these three ever disagree on how a landmark row turns into a feature
vector, the trained model will perform well in training and badly in
real-time use. Keep this file as the single source of truth.
"""

import numpy as np

NUM_LANDMARKS = 21          # MediaPipe Hands always returns 21 landmarks
COORDS_PER_LANDMARK = 3     # x, y, z
FEATURE_LENGTH = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63


def landmarks_to_array(hand_landmarks):
    """
    Convert a MediaPipe `hand_landmarks` object (from results.multi_hand_landmarks[i])
    into a flat numpy array of length 63: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

    MediaPipe already gives x, y normalized to [0, 1] relative to the image width/height,
    and z roughly relative to the wrist depth. We keep the raw values here and let
    `normalize_landmarks` do the position/scale invariance.
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(raw_features):
    """
    Make a raw 63-length landmark vector invariant to:
      - where the hand is in the frame (translation)
      - how close the hand is to the camera / how large it is in frame (scale)

    Steps:
      1. Reshape to (21, 3)
      2. Subtract the wrist (landmark 0) from every point -> translation invariant
      3. Divide every coordinate by the max absolute value in the sample -> scale invariant
      4. Flatten back to a 63-length vector

    This is the standard trick used in MediaPipe-based gesture classifiers
    (e.g. Kazuhito00's hand-gesture-recognition project) and works well with
    a small MLP classifier.
    """
    raw_features = np.asarray(raw_features, dtype=np.float32)

    if raw_features.size != FEATURE_LENGTH:
        raise ValueError(
            f"Expected {FEATURE_LENGTH} values (21 landmarks x 3 coords), "
            f"got {raw_features.size}. Check your CSV column count / "
            f"your data_collection.py output format."
        )

    pts = raw_features.reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK).copy()

    # 1) translation invariance: make wrist the origin
    wrist = pts[0].copy()
    pts -= wrist

    # 2) scale invariance: normalize by the largest absolute coordinate
    max_val = np.max(np.abs(pts))
    if max_val > 1e-6:
        pts /= max_val

    return pts.flatten().astype(np.float32)


def preprocess_row(raw_row):
    """Convenience wrapper: raw CSV row (or live landmark array) -> model-ready vector."""
    return normalize_landmarks(raw_row)
