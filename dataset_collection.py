import cv2
import mediapipe as mp
import time
import pandas as pd
import os

# =========================
# SETTINGS
# =========================

GESTURE_LABEL = 0
CSV_FILE = "gesture_data.csv"

TOTAL_RUNTIME = 12
WAIT_TIME = 2
CAPTURE_INTERVAL = 0.5
MAX_SAMPLES = 20

# =========================
# MEDIAPIPE SETUP
# =========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='hand_landmarker.task'
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

start_time = time.time()
last_capture = 0

dataset = []
sample_count = 0

print("\nStarting dataset collection...")
print(f"Will collect exactly {MAX_SAMPLES} samples\n")

while cap.isOpened():
    print("Cam opened")

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    elapsed = time.time() - start_time

    # Display timer
    cv2.putText(
        frame,
        f"Samples: {sample_count}/{MAX_SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # =========================
    # HAND DETECTION
    # =========================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        # Draw landmarks
        for i, landmark in enumerate(hand):

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(
                frame,
                str(i),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )

        # =========================
        # DATA COLLECTION LOGIC
        # =========================

        if elapsed >= WAIT_TIME:

            if elapsed - last_capture >= CAPTURE_INTERVAL:

                # Stop exactly at 20 samples
                if sample_count >= MAX_SAMPLES:
                    break

                # Extract RAW landmarks (NO NORMALIZATION)
                features = []

                for lm in hand:
                    features.extend([lm.x, lm.y, lm.z])

                row = features + [GESTURE_LABEL]

                dataset.append(row)
                sample_count += 1
                last_capture = elapsed

                print(f"\nSample {sample_count}")
                print(row)

    cv2.imshow("Dataset Collection", frame)

    # Stop if reached max samples
    if sample_count >= MAX_SAMPLES:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# SAVE CSV
# =========================

columns = []
for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])
columns.append("label")

df = pd.DataFrame(dataset, columns=columns)

if os.path.exists(CSV_FILE):
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)
else:
    df.to_csv(CSV_FILE, index=False)

print("\n====================")
print("FINAL DATASET")
print("====================")
print(df)

print("\nShutting down safely...")

try:
    cap.release()
except:
    pass

try:
    landmarker.close()
except:
    pass

cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Cleanup complete.")