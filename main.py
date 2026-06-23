import cv2
import mediapipe as mp
import time

# ---------- MediaPipe setup ----------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='hand_landmarker.task'
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

# ---------- Webcam ----------

cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # Flip webcam for mirror effect
    frame = cv2.flip(frame,1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp = int(time.time()*1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    height,width,_ = frame.shape

    if result.hand_landmarks:

        # Process each detected hand
        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):

            # Get handedness info
            hand_info = result.handedness[hand_index][0]

            hand_label = hand_info.category_name
            confidence = hand_info.score

            print("\n"+"="*40)
            print(
                f"{hand_label} Hand "
                f"(Confidence: {confidence:.2f})"
            )

            # Wrist landmark (point 0)
            wrist = hand_landmarks[0]

            wrist_x = int(wrist.x*width)
            wrist_y = int(wrist.y*height)

            # Draw handedness label
            cv2.putText(
                frame,
                f"{hand_label} ({confidence:.2f})",
                (wrist_x, wrist_y-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

            # Draw landmarks
            for i, landmark in enumerate(hand_landmarks):

                x = int(landmark.x*width)
                y = int(landmark.y*height)

                cv2.circle(
                    frame,
                    (x,y),
                    5,
                    (0,255,0),
                    -1
                )

                cv2.putText(
                    frame,
                    str(i),
                    (x,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255,0,0),
                    1
                )

                print(
                    f"Landmark {i}: "
                    f"x={landmark.x:.3f}, "
                    f"y={landmark.y:.3f}, "
                    f"z={landmark.z:.3f}"
                )

    cv2.imshow(
        "Hand Tracking with Handedness",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()