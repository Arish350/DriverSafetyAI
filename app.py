
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import av

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    AudioProcessorBase,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Driver Safety AI",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Driver Safety AI")
st.write("Real-Time Drowsiness & Yawning Detection")

st.warning(
    "Educational demo only. Do not rely on this app "
    "for real-world driving safety."
)


alarm_event = threading.Event()
MODEL_PATH = "drowsiness_model.pkl"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load AI model: {e}")
    st.stop()


# --------------------------------------------------
# VIDEO PROCESSOR
# --------------------------------------------------

class DriverProcessor(VideoProcessorBase):

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.eye_closed_since = None
        self.yawn_since = None
        self.ear = 0.0
        self.mouth_ratio = 0.0
        self.status = "Face not detected"

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
        warning_active = False

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0].landmark
            h, w = image.shape[:2]

            def point(index):
                return (int(face[index].x * w),
                        int(face[index].y * h))

            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]

            self.ear = (
                self.eye_aspect_ratio([point(i) for i in left_eye])
                + self.eye_aspect_ratio([point(i) for i in right_eye])
            ) / 2

            upper_lip = np.array(point(13))
            lower_lip = np.array(point(14))
            left_mouth = np.array(point(78))
            right_mouth = np.array(point(308))

            vertical = np.linalg.norm(upper_lip - lower_lip)
            horizontal = np.linalg.norm(left_mouth - right_mouth)
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

            try:
                prediction = str(
                    model.predict(
                        np.array([[self.ear, self.mouth_ratio]])
                    )[0]
                )
            except Exception:
                prediction = "Unknown"

            warning_active = drowsy or yawning

            if drowsy:
                self.status = "DROWSINESS WARNING!"
            elif yawning:
                self.status = "YAWNING DETECTED"
            else:
                self.status = f"Alert | AI: {prediction}"

            color = (0, 0, 255) if warning_active else (0, 200, 0)

            cv2.putText(
                image, self.status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )
            cv2.putText(
                image, f"EAR: {self.ear:.2f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            cv2.putText(
                image, f"Mouth ratio: {self.mouth_ratio:.2f}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        else:
            self.status = "Face not detected"
            self.eye_closed_since = None
            self.yawn_since = None

        if warning_active:
            alarm_event.set()
        else:
            alarm_event.clear()

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# --------------------------------------------------
# AUDIO ALARM PROCESSOR
# --------------------------------------------------

class AlarmAudioProcessor(AudioProcessorBase):

    def __init__(self):
        self.sample_index = 0

    def recv(self, frame):
        sample_rate = frame.sample_rate or 48000
        samples = frame.samples
        channels = len(frame.layout.channels)

        positions = (
            np.arange(samples, dtype=np.float64)
            + self.sample_index
        ) / sample_rate

        if alarm_event.is_set():
            pulse = (positions % 0.8) < 0.3
            tone = (
                0.25 * 32767
                * np.sin(2 * np.pi * 1000 * positions)
                * pulse
            ).astype(np.int16)
        else:
            tone = np.zeros(samples, dtype=np.int16)

        # Match PyAV's expected array shape for the output format.
        if frame.format.is_planar:
            audio = np.tile(tone, (channels, 1))
            output_format = "s16p"
        else:
            interleaved = np.repeat(tone[:, None], channels, axis=1)
            audio = interleaved.reshape(1, samples * channels)
            output_format = "s16"

        self.sample_index += samples

        output = av.AudioFrame.from_ndarray(
            audio,
            format=output_format,
            layout=frame.layout.name,
        )
        output.sample_rate = sample_rate
        output.pts = frame.pts
        output.time_base = frame.time_base

        return output


# --------------------------------------------------
# START CAMERA + AUDIO
# --------------------------------------------------

st.info(
    "Press START and allow camera and microphone access. "
    "The microphone audio stream carries the generated alarm "
    "back to the browser."
)

ctx = webrtc_streamer(
    key="driver-safety",
    video_processor_factory=DriverProcessor,
    audio_processor_factory=AlarmAudioProcessor,
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
    sendback_audio=True,
    async_processing=True,
)

st.caption(
    "The alarm triggers when the eye-closure or yawning "
    "threshold is reached."
)
st.caption(
    "This prototype is not a certified driver-monitoring "
    "or vehicle safety system."
)
