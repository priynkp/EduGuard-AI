"""EduGuard AI - Live webcam + microphone + keyboard/tab-switch demo.

Run from the same folder as:
  video_telemetry.csv
  system_events.csv

Install:
  pip install opencv-python mediapipe numpy pandas scikit-learn sounddevice pynput

MediaPipe note:
  Uses the current MediaPipe Tasks Face Landmarker API and is intended for
  the user's existing Python 3.13 environment. The first run downloads
  face_landmarker.task into this same folder.

Controls:
  Q = quit
  R = reset tab-switch counter
  T = simulate a tab-switch event (useful for testing)

Important:
  The trained model has only five features: gaze, audio, tab switches,
  and their rolling averages. It does NOT detect hands, phones, notes,
  or specific gestures. A normal hand-to-face gesture alone therefore
  should not be treated as proof of cheating.

Note: the webcam signal is a gaze/head-angle proxy. The original training data
contains an already-computed `eye_gaze_angle`; this script estimates a comparable
angle from the live face landmarks rather than claiming clinical-grade gaze tracking.
The dashboard's AI risk score combines the trained Random Forest probability with
sustained live behavioral evidence so gaze/audio can contribute even when the
historical model underweights them. It is a decision-support demo, not proof of cheating.
"""

import os
import time
import threading
from collections import deque

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Optional/live dependencies
# -----------------------------
try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("Missing mediapipe. Install with: pip install mediapipe") from e

try:
    import sounddevice as sd
except ImportError as e:
    raise SystemExit("Missing sounddevice. Install with: pip install sounddevice") from e

try:
    from pynput import keyboard
except ImportError as e:
    raise SystemExit("Missing pynput. Install with: pip install pynput") from e


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_CSV = os.path.join(BASE_DIR, "video_telemetry.csv")
SYSTEM_CSV = os.path.join(BASE_DIR, "system_events.csv")
THRESHOLD = 0.90
WINDOW = 10

# Live behavioral-evidence thresholds.
# These are deliberately conservative: a brief glance or short sound should
# not trigger a high-risk result. The final risk combines the ML model with
# sustained live evidence.
GAZE_SUSPICIOUS_DEG = 18.0
GAZE_HIGH_DEG = 28.0
AUDIO_SUSPICIOUS_DB = 55.0
AUDIO_HIGH_DB = 68.0
GAZE_SUSTAINED_SAMPLES = 8
AUDIO_SUSTAINED_SAMPLES = 8

# -----------------------------
# Train the same model used in the notebook
# -----------------------------
def train_model():
    video_df = pd.read_csv(VIDEO_CSV)
    system_df = pd.read_csv(SYSTEM_CSV)

    video_df["timestamp"] = pd.to_datetime(video_df["timestamp"])
    system_df["timestamp"] = pd.to_datetime(system_df["timestamp"])
    video_df = video_df.sort_values("timestamp")
    system_df = system_df.sort_values("timestamp")

    merged_df = pd.merge_asof(
        video_df,
        system_df,
        on="timestamp",
        direction="nearest",
    )

    merged_df["eye_gaze_angle"] = merged_df["eye_gaze_angle"].ffill().bfill()
    merged_df["audio_db"] = merged_df["audio_db"].interpolate().bfill()
    merged_df["tab_switches"] = merged_df["tab_switches"].fillna(0)
    merged_df["is_cheating"] = merged_df["is_cheating"].fillna(0)

    # Keep this identical to the completed notebook: rolling MEANS.
    merged_df["eye_gaze_rolling_10s"] = (
        merged_df["eye_gaze_angle"].rolling(WINDOW, min_periods=1).mean()
    )
    merged_df["audio_rolling_10s"] = (
        merged_df["audio_db"].rolling(WINDOW, min_periods=1).mean()
    )

    features = [
        "eye_gaze_angle",
        "audio_db",
        "tab_switches",
        "eye_gaze_rolling_10s",
        "audio_rolling_10s",
    ]

    X = merged_df[features]
    y = merged_df["is_cheating"].astype(int)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Shared live state
# -----------------------------
state_lock = threading.Lock()
tab_switches = 0
latest_audio_db = 35.0


def increment_tab_switch():
    global tab_switches
    with state_lock:
        tab_switches += 1


def reset_tab_switches():
    global tab_switches
    with state_lock:
        tab_switches = 0


def keyboard_on_press(key):
    """Global keyboard listener for real tab-switch combinations.

    Manual T/R controls are handled by OpenCV's window below. Keeping them
    out of this global listener prevents one T press from being counted twice.
    """
    pass


pressed = set()


def keyboard_on_press_combo(key):
    # Real Ctrl+Tab/PageUp/PageDown events are counted globally.
    # T/R are handled only by cv2.waitKey in the main window.
    pressed.add(key)
    try:
        if (keyboard.Key.ctrl_l in pressed or keyboard.Key.ctrl_r in pressed):
            if key in (keyboard.Key.tab, keyboard.Key.page_down, keyboard.Key.page_up):
                increment_tab_switch()
    except Exception:
        pass
    keyboard_on_press(key)


def keyboard_on_release(key):
    pressed.discard(key)


# -----------------------------
# Microphone
# -----------------------------
def audio_callback(indata, frames, callback_time, status):
    global latest_audio_db
    if status:
        pass
    samples = np.asarray(indata, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return
    rms = float(np.sqrt(np.mean(samples * samples)))
    # Approximate dB SPL-like scale for compatibility with the project's
    # 17-88-ish telemetry range. This is NOT calibrated physical SPL.
    # Shift the microphone scale into the same approximate range as the
    # training telemetry (roughly 17-88). This is still NOT physical dB SPL.
    db = 20.0 * np.log10(max(rms, 1e-7)) + 120.0
    db = float(np.clip(db, 17.0, 88.0))
    with state_lock:
        latest_audio_db = db


# -----------------------------
# Face/gaze proxy
# -----------------------------
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")


def ensure_face_model():
    """Download the MediaPipe Face Landmarker model once if needed."""
    if os.path.exists(FACE_MODEL_PATH) and os.path.getsize(FACE_MODEL_PATH) > 100_000:
        return

    print("Face Landmarker model not found. Downloading it once...")
    import urllib.request

    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    except Exception as e:
        raise SystemExit(
            "Could not download face_landmarker.task. "
            "Check your internet connection, then run the program again. "
            f"Download error: {e}"
        )

    if not os.path.exists(FACE_MODEL_PATH) or os.path.getsize(FACE_MODEL_PATH) <= 100_000:
        raise SystemExit("The Face Landmarker model download was incomplete.")


def make_gaze_estimator():
    """Create MediaPipe's current Face Landmarker Tasks API.

    This replaces the legacy `mp.solutions.face_mesh` API, which is not
    exposed by the MediaPipe package in the user's Python 3.13 environment.
    """
    ensure_face_model()

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def estimate_gaze_angle(face_landmarks, width, height):
    """Estimate horizontal gaze/head angle in degrees.

    Uses iris position relative to the eye corners. It is deliberately a
    simple proxy because the training CSV already contains an engineered
    `eye_gaze_angle`, not raw images.

    `face_landmarks` is the landmark list returned by the MediaPipe Tasks
    FaceLandmarker API.
    """
    lm = face_landmarks

    # MediaPipe refined face landmarks: left/right iris centers and eye corners.
    # Indices: left iris 468-472, right iris 473-477.
    left_iris = np.mean([[lm[i].x, lm[i].y] for i in range(468, 473)], axis=0)
    right_iris = np.mean([[lm[i].x, lm[i].y] for i in range(473, 478)], axis=0)

    left_inner = np.array([lm[133].x, lm[133].y])
    left_outer = np.array([lm[33].x, lm[33].y])
    right_inner = np.array([lm[362].x, lm[362].y])
    right_outer = np.array([lm[263].x, lm[263].y])

    def eye_ratio(iris, a, b):
        denom = max(np.linalg.norm(b - a), 1e-6)
        return float(np.dot(iris - a, b - a) / (denom * denom))

    l_ratio = eye_ratio(left_iris, left_outer, left_inner)
    r_ratio = eye_ratio(right_iris, right_inner, right_outer)
    ratio = np.mean([l_ratio, r_ratio])

    angle = abs((ratio - 0.5) * 80.0)
    return float(np.clip(angle, 0.0, 45.0))


# -----------------------------
# Behavioral evidence scoring
# -----------------------------
def calculate_behavior_score(gaze_history, audio_history, tabs):
    """Return a 0..1 live evidence score plus human-readable evidence.

    This is a demo decision-support layer on top of the Random Forest.
    Sustained signals are weighted more heavily than one-frame spikes.
    """
    evidence = []

    gaze_vals = list(gaze_history)
    audio_vals = list(audio_history)

    gaze_avg = float(np.mean(gaze_vals)) if gaze_vals else 0.0
    audio_avg = float(np.mean(audio_vals)) if audio_vals else 0.0

    # Sustained gaze deviation.
    gaze_sustained = (
        len(gaze_vals) >= GAZE_SUSTAINED_SAMPLES
        and sum(g >= GAZE_SUSPICIOUS_DEG for g in gaze_vals[-GAZE_SUSTAINED_SAMPLES:])
        >= GAZE_SUSTAINED_SAMPLES * 0.75
    )
    gaze_high = gaze_avg >= GAZE_HIGH_DEG or (
        len(gaze_vals) >= GAZE_SUSTAINED_SAMPLES
        and sum(g >= GAZE_HIGH_DEG for g in gaze_vals[-GAZE_SUSTAINED_SAMPLES:])
        >= GAZE_SUSTAINED_SAMPLES * 0.75
    )

    gaze_score = 0.0
    if gaze_high:
        gaze_score = 1.0
        evidence.append("HIGH: prolonged gaze deviation")
    elif gaze_sustained:
        gaze_score = 0.65
        evidence.append("WARN: prolonged gaze deviation")
    elif gaze_avg >= GAZE_SUSPICIOUS_DEG:
        gaze_score = 0.35
        evidence.append("INFO: gaze deviation")

    # Sustained abnormal audio.
    audio_sustained = (
        len(audio_vals) >= AUDIO_SUSTAINED_SAMPLES
        and sum(a >= AUDIO_SUSPICIOUS_DB for a in audio_vals[-AUDIO_SUSTAINED_SAMPLES:])
        >= AUDIO_SUSTAINED_SAMPLES * 0.75
    )
    audio_high = audio_avg >= AUDIO_HIGH_DB or (
        len(audio_vals) >= AUDIO_SUSTAINED_SAMPLES
        and sum(a >= AUDIO_HIGH_DB for a in audio_vals[-AUDIO_SUSTAINED_SAMPLES:])
        >= AUDIO_SUSTAINED_SAMPLES * 0.75
    )

    audio_score = 0.0
    if audio_high:
        audio_score = 1.0
        evidence.append("HIGH: sustained abnormal audio")
    elif audio_sustained:
        audio_score = 0.55
        evidence.append("WARN: sustained abnormal audio")
    elif audio_avg >= AUDIO_SUSPICIOUS_DB:
        audio_score = 0.25
        evidence.append("INFO: elevated audio")

    # A tab switch is a strong explicit system event in this project.
    tab_score = min(1.0, tabs / 2.0) if tabs > 0 else 0.0
    if tabs > 0:
        evidence.append(f"WARN: {tabs} tab switch{'es' if tabs != 1 else ''}")

    # Combine independent evidence. Multiple simultaneous signals receive
    # a modest synergy boost, while one weak signal remains low risk.
    behavior_score = (
        0.40 * gaze_score
        + 0.20 * audio_score
        + 0.40 * tab_score
    )

    active = sum(x > 0 for x in (gaze_score, audio_score, tab_score))
    if active >= 2:
        behavior_score += 0.10
    if active >= 3:
        behavior_score += 0.10

    return float(np.clip(behavior_score, 0.0, 1.0)), evidence


def risk_label(score):
    if score >= 0.90:
        return "HIGH RISK"
    if score >= 0.65:
        return "SUSPICIOUS"
    if score >= 0.35:
        return "WATCH"
    return "NORMAL"


# -----------------------------
# Main live loop
# -----------------------------
def main():
    global latest_audio_db

    if not os.path.exists(VIDEO_CSV) or not os.path.exists(SYSTEM_CSV):
        raise SystemExit(
            "Put live_eduguard.py, video_telemetry.csv and system_events.csv in the same folder."
        )

    print("Training EduGuard Random Forest from your project data...")
    model = train_model()
    print("Model ready. 90% decision threshold active.")

    gaze_history = deque(maxlen=WINDOW)
    audio_history = deque(maxlen=WINDOW)

    face_mesh = make_gaze_estimator()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions or another app using the camera.")

    listener = keyboard.Listener(on_press=keyboard_on_press_combo, on_release=keyboard_on_release)
    listener.start()

    try:
        audio_stream = sd.InputStream(
            channels=1,
            samplerate=16000,
            blocksize=1024,
            callback=audio_callback,
        )
        audio_stream.start()
    except Exception as e:
        listener.stop()
        cap.release()
        raise SystemExit(f"Could not open microphone: {e}")

    last_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = face_mesh.detect(mp_image)

            face_found = bool(result.face_landmarks)
            if face_found:
                gaze = estimate_gaze_angle(result.face_landmarks[0], w, h)
            else:
                # No face: use a conservative neutral value rather than
                # automatically accusing the student.
                gaze = 0.0

            with state_lock:
                audio_db = latest_audio_db
                tabs = tab_switches

            gaze_history.append(gaze)
            audio_history.append(audio_db)

            gaze_roll = float(np.mean(gaze_history))
            audio_roll = float(np.mean(audio_history))

            # The historical system_events.csv uses 1 as the no-switch baseline.
            # Convert the live counter (0 = no switch) to the same feature scale.
            model_tabs = max(1, tabs + 1)
            features = pd.DataFrame(
                [[gaze, audio_db, model_tabs, gaze_roll, audio_roll]],
                columns=[
                    "eye_gaze_angle", "audio_db", "tab_switches",
                    "eye_gaze_rolling_10s", "audio_rolling_10s"
                ],
            )
            model_probability = float(model.predict_proba(features)[0, 1])

            # Add sustained live behavioral evidence. The Random Forest remains
            # the ML core; this layer prevents the demo from relying on tab
            # switching alone.
            behavior_score, evidence = calculate_behavior_score(
                gaze_history, audio_history, tabs
            )

            # Keep the trained model influential while allowing signals that
            # the historical model underweights to affect live risk.
            risk_score = float(np.clip(
                0.60 * model_probability + 0.40 * behavior_score,
                0.0, 1.0
            ))

            status_text = risk_label(risk_score)

            # -------------------------
            # Dashboard overlay
            # -------------------------
            # Keep the dashboard readable on a 1280x720/fullscreen window.
            sx = max(w / 1280.0, 0.75)
            fs_title = 1.15 * sx
            fs_body = 0.78 * sx
            fs_small = 0.62 * sx
            panel_h = int(305 * sx)
            panel = frame.copy()
            cv2.rectangle(panel, (0, 0), (w, min(h, panel_h)), (15, 15, 15), -1)
            frame = cv2.addWeighted(panel, 0.78, frame, 0.22, 0)

            cv2.putText(frame, "EDUGUARD AI PROCTOR", (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_title, (255, 255, 255), 2)
            cv2.putText(frame, f"Gaze angle:        {gaze:6.1f} deg", (24, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1)
            cv2.putText(frame, f"Audio level:       {audio_db:6.1f} dB*", (24, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1)
            cv2.putText(frame, f"Tab switches:      {tabs:6d}", (24, 152),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1)
            cv2.putText(frame, f"10s gaze average:  {gaze_roll:6.1f} deg", (24, 186),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1)
            cv2.putText(frame, f"10s audio average: {audio_roll:6.1f} dB*", (24, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1)
            cv2.putText(frame, f"AI risk score:     {risk_score * 100:5.1f}%", (24, 254),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1)
            cv2.putText(frame, f"ML model:          {model_probability * 100:5.1f}%", (24, 288),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1)

            # Status badge
            badge = "STATUS:  " + status_text
            cv2.putText(frame, badge, (max(w - 420, 700), 48),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 2)

            # Evidence panel.
            evidence_y = 92
            cv2.putText(frame, "LIVE EVIDENCE", (max(w - 420, 700), evidence_y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1)
            if evidence:
                for item in evidence[:3]:
                    evidence_y += 32
                    cv2.putText(
                        frame, item, (max(w - 420, 700), evidence_y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_small, (255, 255, 255), 1
                    )
            else:
                cv2.putText(
                    frame, "No suspicious evidence", (max(w - 420, 700), evidence_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_small, (255, 255, 255), 1
                )

            if not face_found:
                cv2.putText(frame, "NO FACE DETECTED", (max(w - 420, 700), evidence_y + 64),
                            cv2.FONT_HERSHEY_SIMPLEX, fs_small, (255, 255, 255), 1)

            cv2.putText(frame, "Q: quit   T: test tab   R: reset tabs   |   *audio = relative mic level", (24, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_small, (220, 220, 220), 1)

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(frame, f"FPS {fps:.1f}", (w - 120, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_small, (220, 220, 220), 1)

            cv2.namedWindow("EduGuard AI - Live Proctor", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("EduGuard AI - Live Proctor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("EduGuard AI - Live Proctor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                increment_tab_switch()
            elif key == ord("r"):
                reset_tab_switches()

    finally:
        audio_stream.stop()
        audio_stream.close()
        listener.stop()
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
