"""
realtime_classifier.py
------------------------
STEP 3: Hand detection + trained model integration for gesture classification.

Opens your webcam, detects your hand with MediaPipe, runs the SAME
normalization used in training, feeds it to the trained model, and shows
the predicted gesture + confidence live on screen.

Run this AFTER train_model.py has produced model/gesture_model.h5.
This script does not control any game yet -- it's purely to verify the
model recognizes your gestures correctly before wiring it into the game
(that's main.py / game_controller.py, step 4).

Press 'q' to quit.
"""

import json
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from config import MODEL_PATH, LABEL_MAP_PATH, CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW
from preprocessing import landmarks_to_array, normalize_landmarks

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r") as f:
        raw_map = json.load(f)
    label_map = {int(k): v for k, v in raw_map.items()}
    return model, label_map


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


def main():
    model, label_map = load_model_and_labels()
    predictor = GesturePredictor(model, label_map)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display_text = "No hand detected"
            color = (0, 0, 255)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                raw = landmarks_to_array(hand_landmarks)
                class_id, confidence = predictor.predict(raw)

                if class_id is not None:
                    display_text = f"{label_map[class_id]}  ({confidence*100:.0f}%)"
                    color = (0, 255, 0)
                else:
                    display_text = f"Unsure ({confidence*100:.0f}%)"
                    color = (0, 165, 255)
            else:
                predictor.reset()

            cv2.putText(frame, display_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, "press q to quit", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Gesture Classifier - Step 3 test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
