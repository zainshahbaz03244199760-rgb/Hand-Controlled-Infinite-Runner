"""
#
main.py
---------
STEP 4 (part B): Full pipeline -- webcam -> hand detection -> trained model
-> gesture smoothing -> game key presses.

This is the script you actually run while playing the game.

How to use:
  1. Open Subway Surfers (browser tab or app window) and make sure it's
     the focused/active window.
  2. Run this script (it will open a second small window showing your
     webcam feed with the recognized gesture overlaid -- useful for
     debugging, you can ignore it once things work).
  3. Make gestures in front of your camera; the corresponding key presses
     are sent to whichever window is focused (the game).
  4. Press 'q' inside the webcam window to stop.

Make sure you've already run train_model.py at least once so that
model/gesture_model.h5 and model/label_map.json exist.
"""

import json
import time
import requests

import cv2
import mediapipe as mp
import tensorflow as tf

from config import MODEL_PATH, LABEL_MAP_PATH
from preprocessing import landmarks_to_array
from realtime_classifier import GesturePredictor
from game_controller import GameController

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
session = requests.Session()
last_prediction_time = 0
prediction_interval = 0.05


def main():
    """print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    predictor = GesturePredictor(model, label_map)"""

    global last_prediction_time

    with open(LABEL_MAP_PATH, "r") as f:
        label_map = {
            int(k): v
            for k, v in json.load(f).items()
        }
    controller = GameController()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    print("Starting in 3 seconds -- click on your game window now so it has focus...")
    time.sleep(3)
    print("Go! Press 'q' in the webcam window to stop.")

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
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

            display_text = "No hand"
            color = (0, 0, 255)
            gesture_id = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                """raw = landmarks_to_array(hand_landmarks)
                gesture_id, confidence = predictor.predict(raw)
                print(gesture_id)

                if gesture_id is not None:
                    display_text = f"{label_map[gesture_id]} ({confidence*100:.0f}%)"
                    color = (0, 255, 0)
                else:
                    display_text = f"Unsure ({confidence*100:.0f}%)"
                    color = (0, 165, 255)"""

                current_time = time.time()

                if current_time - last_prediction_time >= prediction_interval:
                    raw = landmarks_to_array(hand_landmarks)

                    try:
                        response = session.post(
                            "http://localhost:5000/predict",
                            json={
                                "landmarks": raw.tolist()
                            },
                            timeout=0.2
                        )

                        result = response.json()

                        gesture_id = result["gesture_id"]
                        confidence = result["confidence"]

                        last_prediction_time = current_time

                    except Exception as e:
                        print("API Error:", e)

                        gesture_id = None
                        confidence = 0

                    print(gesture_id)

                    if gesture_id is not None:
                        display_text = (
                            f"{label_map[gesture_id]}"
                            f" ({confidence * 100:.0f}%)"
                        )

                        color = (0, 255, 0)

                    else:
                        display_text = (
                            f"Unsure ({confidence * 100:.0f}%)"
                        )

                        color = (0, 165, 255)

                else:
                    # predictor.reset()
                    try:
                        requests.post(
                            "http://localhost:5000/reset"
                        )
                    except:
                        pass

            controller.handle_gesture(gesture_id)

            cv2.putText(frame, display_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, "press q to quit", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Gesture Game Control (debug view)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
