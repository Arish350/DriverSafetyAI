
import streamlit as st
import streamlit.components.v1 as components
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
components.html("""
<button onclick="playAlarm()"
style="padding:15px 25px;font-size:18px;cursor:pointer;">
🔊 Test Alarm Sound
</button>

<script>
function playAlarm() {
    const audio = new AudioContext();
    const oscillator = audio.createOscillator();
    const gain = audio.createGain();

    oscillator.type = "sine";
    oscillator.frequency.value = 1000;
    gain.gain.value = 0.3;

    oscillator.connect(gain);
    gain.connect(audio.destination);

    oscillator.start();
    setTimeout(() => {
        oscillator.stop();
        audio.close();
    }, 1000);
}
</script>
""", height=90)
st.warning(
    "Educational demo only. Do not rely on this app "
    "for real-world driving safety."
)

# Shared signal between video detection and alarm audio
alarm_event = threading.Event()

MODEL_PATH = "drowsiness_model.pkl"


# --------------------------------------------------
# LOAD AI MODEL
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

        return (
            (vertical1 + vertical2)
            / (2 * horizontal + 1e-6)
        )

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
                return (
                    int(face[index].x * w),
                    int(face[index].y * h),
                )

            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]

            left_ear = self.eye_aspect_ratio(
                [point(i) for i in left_eye]
            )
            right_ear = self.eye_aspect_ratio(
                [point(i) for i in right_eye]
            )

            self.ear = (left_ear + right_ear) / 2

            upper_lip = np.array(point(13))
            lower_lip = np.array(point(14))
            left_mouth = np.array(point(78))
            right_mouth = np.array(point(308))

            vertical = np.linalg.norm(upper_lip - lower_lip)
            horizontal = np.linalg.norm(left_mouth - right_mouth)

            self.mouth_ratio = vertical / (horizontal + 1e-6)

            # Eye closure timer
            if self.ear < 0.22:
                if self.eye_closed_since is None:
                    self.eye_closed_since = now
            else:
                self.eye_closed_since = None

            # Yawning timer
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

            # AI prediction (informational only)
            try:
                features = np.array([[
                    self.ear,
                    self.mouth_ratio,
                ]])
                prediction = str(
                    model.predict(features)[0]
                )
            except Exception:
                prediction = "Unknown"

            # Actual alarm is based on threshold detections
            warning_active = drowsy or yawning

            if drowsy:
                self.status = "DROWSINESS WARNING!"
            elif yawning:
                self.status = "YAWNING DETECTED"
            else:
                self.status = f"Alert | AI: {prediction}"

            color = (
                (0, 0, 255)
                if warning_active
                else (0, 200, 0)
            )

            cv2.putText(
                image,
                self.status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

            cv2.putText(
                image,
                f"EAR: {self.ear:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                image,
                f"Mouth ratio: {self.mouth_ratio:.2f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        else:
            self.status = "Face not detected"
            self.eye_closed_since = None
            self.yawn_since = None

        # Share warning with audio processor
        if warning_active:
            alarm_event.set()
        else:
            alarm_event.clear()

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# --------------------------------------------------
# AUDIO ALARM PROCESSOR
# --------------------------------------------------

class AlarmAudioProcessor(AudioProcessorBase):

    def __init__(self):
        self.sample_index = 0

    def recv(self, frame):
        sample_rate = frame.sample_rate or 48000
        samples = frame.samples
        layout_name = frame.layout.name
        channels = len(frame.layout.channels)

        # Silence unless a warning is active
        audio = np.zeros(
            (channels, samples),
            dtype=np.int16,
        )

        if alarm_event.is_set():
            positions = (
                np.arange(samples) + self.sample_index
            ) / sample_rate

            # 1000 Hz beep, pulsing on/off
            pulse_on = (positions % 0.8) < 0.3

            tone = (
                np.sin(2 * np.pi * 1000 * positions)
                * 0.25
                * 32767
                * pulse_on
            ).astype(np.int16)

            for channel in range(channels):
                audio[channel, :] = tone

        self.sample_index += samples

        output = av.AudioFrame.from_ndarray(
            audio,
            format="s16",
            layout=layout_name,
        )

        output.sample_rate = sample_rate
        output.pts = frame.pts
        output.time_base = frame.time_base

        return output


# --------------------------------------------------
# START CAMERA AND AUDIO
# --------------------------------------------------

st.info(
    "Click START and allow camera and microphone permissions. "
    "The microphone stream is used to carry the generated alarm "
    "audio back to your browser."
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
    "Alarm sound is generated when eye-closure or yawning "
    "thresholds trigger. It may take a moment to start."
)

st.caption(
    "Detection can be inaccurate. This is not a certified "
    "driver-monitoring or vehicle safety system."
)
