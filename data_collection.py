"""
data_collection.py
--------------------
STEP 1: Hand detection + dataset collection.

You already have your 6 CSV files, so you mostly won't need to re-run this.
It's included so you can:
  - collect MORE samples later if accuracy is low for a specific gesture
  - see exactly what format the CSVs are expected to be in (so you can
    double check your existing files match)

Controls:
  0-5   -> select which gesture you are currently recording
  SPACE -> start / stop recording (while recording, every detected frame
           is appended as one row to that gesture's CSV)
  q     -> quit

Each CSV row = 63 raw numbers: x0,y0,z0, x1,y1,z1, ... x20,y20,z20
(no header, no label column -- the label comes from the filename).
This matches what `preprocessing.py` and `train_model.py` expect.
"""

import os
import csv
import cv2
import mediapipe as mp

from config import GESTURES, DATASET_DIR
from preprocessing import landmarks_to_array

os.makedirs(DATASET_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_csv_path(gesture_id):
    fname = GESTURES[gesture_id]["file"] + ".csv"
    return os.path.join(DATASET_DIR, fname)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    current_gesture = 0
    recording = False
    saved_counts = {gid: 0 for gid in GESTURES}

    # Pre-count existing rows so the on-screen counter is accurate if you're
    # adding to an existing dataset rather than starting fresh.
    for gid in GESTURES:
        path = get_csv_path(gid)
        if os.path.exists(path):
            with open(path, "r") as f:
                saved_counts[gid] = sum(1 for _ in f)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        print("Press 0-5 to choose gesture, SPACE to start/stop recording, q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if recording:
                    row = landmarks_to_array(hand_landmarks)  # raw 63 values
                    with open(get_csv_path(current_gesture), "a", newline="") as f:
                        csv.writer(f).writerow(row.tolist())
                    saved_counts[current_gesture] += 1

            # --- overlay UI ---
            label = GESTURES[current_gesture]["label"]
            count = saved_counts[current_gesture]
            status = "RECORDING" if recording else "paused"
            color = (0, 0, 255) if recording else (0, 255, 0)

            cv2.putText(frame, f"Gesture [{current_gesture}]: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples saved: {count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "0-5: pick gesture | SPACE: rec on/off | q: quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Dataset Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
            if 48 <= key <= 53:  # ascii '0' to '5'
                current_gesture = key - 48
                recording = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
