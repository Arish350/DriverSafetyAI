
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from collections import deque

st.set_page_config(
    page_title="Driver Safety AI",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Driver Safety AI")
st.write("Real-Time Drowsiness and Yawning Detection")
st.warning(
    "Educational demo only. Do not rely on this app "
    "to keep a driver safe while driving."
)

MODEL_PATH = "drowsiness_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load AI model: {e}")
    st.stop()

class DriverProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.eye_closed_since = None
        self.yawn_since = None
        self.ear = 0.0
        self.mouth_ratio = 0.0
        self.status = "Face not detected"
        self.last_alarm_time = 0
        self.history = deque(maxlen=30)

    def eye_aspect_ratio(self, points):
        p = np.array(points, dtype=np.float32)
        vertical1 = np.linalg.norm(p[1] - p[5])
        vertical2 = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        return (vertical1 + vertical2) / (2 * horizontal + 1e-6)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        now = time.time()
        self.status = "Alert"

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0].landmark
            h, w = image.shape[:2]

            def point(i):
                return (int(face[i].x * w), int(face[i].y * h))

            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]

            left_ear = self.eye_aspect_ratio([point(i) for i in left_eye])
            right_ear = self.eye_aspect_ratio([point(i) for i in right_eye])
            self.ear = (left_ear + right_ear) / 2

            upper_lip = point(13)
            lower_lip = point(14)
            left_mouth = point(78)
            right_mouth = point(308)

            vertical = np.linalg.norm(
                np.array(upper_lip) - np.array(lower_lip)
            )
            horizontal = np.linalg.norm(
                np.array(left_mouth) - np.array(right_mouth)
            )
            self.mouth_ratio = vertical / (horizontal + 1e-6)

            if self.ear < 0.22:
                if self.eye_closed_since is None:
                    self.eye_closed_since = now
            else:
                self.eye_closed_since = None

            if self.mouth_ratio > 0.35:
                if self.yawn_since is None:
                    self.yawn_since = now
            else:
                self.yawn_since = None

            drowsy = (
                self.eye_closed_since is not None
                and now - self.eye_closed_since >= 1.5
            )
            yawning = (
                self.yawn_since is not None
                and now - self.yawn_since >= 1.0
            )

            features = np.array([[
                self.ear,
                self.mouth_ratio
            ]])

            try:
                prediction = str(model.predict(features)[0]).lower()
            except Exception:
                prediction = "unknown"

            if drowsy:
                self.status = "DROWSINESS WARNING!"
            elif yawning:
                self.status = "YAWNING DETECTED"
            elif "drowsy" in prediction:
                self.status = "AI: Drowsy"
            else:
                self.status = "Alert"

            color = (0, 0, 255) if (drowsy or yawning) else (0, 200, 0)
            cv2.putText(
                image, self.status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
            cv2.putText(
                image, f"EAR: {self.ear:.2f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            cv2.putText(
                image, f"Mouth ratio: {self.mouth_ratio:.2f}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            self.history.append(self.status)
        else:
            self.status = "Face not detected"
            self.eye_closed_since = None
            self.yawn_since = None

        return frame.from_ndarray(image, format="bgr24")


st.info(
    "Click START below and allow browser camera access. "
    "Use a well-lit area and look toward the camera."
)

webrtc_streamer(
    key="driver-safety",
    video_processor_factory=DriverProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.caption(
    "Camera access requires permission. Detection performance depends "
    "on lighting, camera quality, and model accuracy."
)
