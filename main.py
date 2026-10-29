import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import joblib
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

EAR_THRESHOLD = 0.22
EYE_CLOSED_SECONDS = 1.5

YAWN_THRESHOLD = 0.35
YAWN_SECONDS = 1.0


# --------------------------------------------------
# LOAD AI MODEL
# --------------------------------------------------

MODEL_PATH = Path(__file__).with_name("drowsiness_model.pkl")
model = joblib.load(MODEL_PATH)

print("AI model loaded successfully!")


# --------------------------------------------------
# MEDIAPIPE FACE MESH
# --------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# --------------------------------------------------
# DISTANCE FUNCTION
# --------------------------------------------------

def distance(point1, point2):
    return np.linalg.norm(
        np.array(point1) - np.array(point2)
    )


# --------------------------------------------------
# EYE ASPECT RATIO
# --------------------------------------------------

def eye_aspect_ratio(landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear


# --------------------------------------------------
# MOUTH OPEN RATIO
# --------------------------------------------------

def mouth_open_ratio(landmarks, width, height):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    top_point = (
        int(top.x * width),
        int(top.y * height)
    )

    bottom_point = (
        int(bottom.x * width),
        int(bottom.y * height)
    )

    left_point = (
        int(left.x * width),
        int(left.y * height)
    )

    right_point = (
        int(right.x * width),
        int(right.y * height)
    )

    vertical = distance(top_point, bottom_point)
    horizontal = distance(left_point, right_point)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


# --------------------------------------------------
# ALARM SETUP
# --------------------------------------------------

pygame.mixer.init()

sample_rate = 44100
duration = 0.5
frequency = 1000

sound_time = np.arange(
    int(sample_rate * duration)
) / sample_rate

wave = (
    12000 * np.sin(2 * np.pi * frequency * sound_time)
).astype(np.int16)

stereo_wave = np.column_stack((wave, wave))

alarm_sound = pygame.sndarray.make_sound(stereo_wave)
alarm_channel = pygame.mixer.Channel(0)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        pygame.quit()
        return

    eye_closed_start = None
    yawn_start = None

    drowsy = False
    yawning = False

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while True:

                success, frame = cap.read()

                if not success:
                    print("Cannot read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = face_mesh.process(rgb_frame)

                # Reset current frame prediction
                ai_prediction = "unknown"

                drowsy = False
                yawning = False

                if results.multi_face_landmarks:

                    face_landmarks = (
                        results.multi_face_landmarks[0].landmark
                    )

                    # ------------------------------------------
                    # CALCULATE EAR
                    # ------------------------------------------

                    left_ear = eye_aspect_ratio(
                        face_landmarks,
                        LEFT_EYE,
                        width,
                        height
                    )

                    right_ear = eye_aspect_ratio(
                        face_landmarks,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    ear = (left_ear + right_ear) / 2.0

                    # ------------------------------------------
                    # CALCULATE MOUTH RATIO
                    # ------------------------------------------

                    mouth_ratio = mouth_open_ratio(
                        face_landmarks,
                        width,
                        height
                    )

                    current_time = time.time()

                    # ------------------------------------------
                    # EYE CLOSURE DETECTION
                    # ------------------------------------------

                    if ear < EAR_THRESHOLD:

                        if eye_closed_start is None:
                            eye_closed_start = current_time

                        elif (
                            current_time - eye_closed_start
                            >= EYE_CLOSED_SECONDS
                        ):
                            drowsy = True

                    else:
                        eye_closed_start = None

                    # ------------------------------------------
                    # YAWNING DETECTION
                    # ------------------------------------------

                    if mouth_ratio > YAWN_THRESHOLD:

                        if yawn_start is None:
                            yawn_start = current_time

                        elif (
                            current_time - yawn_start
                            >= YAWN_SECONDS
                        ):
                            yawning = True

                    else:
                        yawn_start = None

                    # ------------------------------------------
                    # AI PREDICTION - DISPLAY ONLY
                    # ------------------------------------------

                    features = pd.DataFrame(
                        [[ear, mouth_ratio]],
                        columns=["ear", "mouth_ratio"]
                    )

                    ai_prediction = model.predict(features)[0]

                    # ------------------------------------------
                    # DISPLAY VALUES
                    # ------------------------------------------

                    cv2.putText(
                        frame,
                        f"EAR: {ear:.2f}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Mouth: {mouth_ratio:.2f}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"AI: {ai_prediction.upper()}",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

                    if drowsy:
                        cv2.putText(
                            frame,
                            "WARNING: EYES CLOSED!",
                            (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                    if yawning:
                        cv2.putText(
                            frame,
                            "WARNING: YAWNING!",
                            (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                else:
                    eye_closed_start = None
                    yawn_start = None

                    cv2.putText(
                        frame,
                        "Face not detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                # ------------------------------------------
                # IMPORTANT: ALARM LOGIC
                # AI PREDICTION DOES NOT TRIGGER SOUND
                # ------------------------------------------

                warning_active = drowsy or yawning

                if warning_active:
                    if not alarm_channel.get_busy():
                        alarm_channel.play(
                            alarm_sound,
                            loops=-1
                        )
                else:
                    if alarm_channel.get_busy():
                        alarm_channel.stop()

                cv2.imshow(
                    "Driver Drowsiness Detection",
                    frame
                )

                # Press Q to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        if alarm_channel.get_busy():
            alarm_channel.stop()

        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

        print("Driver Safety AI stopped.")


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()