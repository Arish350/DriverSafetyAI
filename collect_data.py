import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

DATA_FILE = "drowsiness_data.csv"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

mp_face_mesh = mp.solutions.face_mesh


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, indices, width, height):
    points = [
        (landmarks[i].x * width, landmarks[i].y * height)
        for i in indices
    ]

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_ratio(landmarks, width, height):
    upper = (landmarks[13].x * width, landmarks[13].y * height)
    lower = (landmarks[14].x * width, landmarks[14].y * height)
    left = (landmarks[78].x * width, landmarks[78].y * height)
    right = (landmarks[308].x * width, landmarks[308].y * height)

    horizontal = distance(left, right)

    if horizontal == 0:
        return 0.0

    return distance(upper, lower) / horizontal


file_exists = os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit(1)

label = None
last_save = 0
counts = {"alert": 0, "drowsy": 0}

try:
    with open(DATA_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["ear", "mouth_ratio", "label"])

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            print("A = Alert | D = Drowsy | Q = Quit")

            while True:
                success, frame = camera.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                ear = None
                mouth = None

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    left = eye_aspect_ratio(
                        landmarks, LEFT_EYE, width, height
                    )
                    right = eye_aspect_ratio(
                        landmarks, RIGHT_EYE, width, height
                    )
                    ear = (left + right) / 2

                    mouth = mouth_ratio(landmarks, width, height)

                    cv2.putText(
                        frame, f"EAR: {ear:.2f}  Mouth: {mouth:.2f}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2
                    )

                    # Save a sample every 0.2 seconds
                    now = time.monotonic()
                    if label is not None and now - last_save >= 0.2:
                        writer.writerow([ear, mouth, label])
                        file.flush()
                        counts[label] += 1
                        last_save = now

                label_text = label.upper() if label else "NOT SELECTED"
                cv2.putText(
                    frame, f"Label: {label_text}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2
                )

                cv2.putText(
                    frame,
                    f"Alert: {counts['alert']}  Drowsy: {counts['drowsy']}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2
                )

                cv2.putText(
                    frame, "A=Alert | D=Drowsy | Q=Quit",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                cv2.imshow("Collect Training Data", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("a"):
                    label = "alert"
                    last_save = 0
                    print("Collecting ALERT samples")
                elif key == ord("d"):
                    label = "drowsy"
                    last_save = 0
                    print("Collecting DROWSY samples")
                elif key == ord("q"):
                    break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Dataset saved to {DATA_FILE}")
print("Samples collected this session:", counts)