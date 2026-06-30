from collections import deque, Counter
from config import CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW
from preprocessing import normalize_landmarks
import numpy as np

class GesturePredictor:
    """Wraps the model with a small majority-vote smoothing window so a single
    noisy frame doesn't flip the predicted gesture."""

    def __init__(self, model, label_map, window=SMOOTHING_WINDOW, threshold=CONFIDENCE_THRESHOLD):
        self.model = model
        self.label_map = label_map
        self.threshold = threshold
        self.history = deque(maxlen=window)

    def predict(self, raw_landmarks_63):
        features = normalize_landmarks(raw_landmarks_63)
        probs = self.model.predict(features.reshape(1, -1), verbose=0)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        if confidence < self.threshold:
            return None, confidence  # not confident enough, ignore this frame

        self.history.append(class_id)
        most_common_id, _ = Counter(self.history).most_common(1)[0]
        return most_common_id, confidence

    def reset(self):
        self.history.clear()