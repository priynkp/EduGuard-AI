# EduGuard AI Proctor

**EduGuard AI** is an AI-assisted online exam proctoring prototype that combines
machine-learning predictions with live webcam, microphone, and keyboard/tab-switch
signals.

> **Important:** This is a decision-support prototype, not proof that a student is
> cheating. Human review and an appeal process should be used for real assessments.

## Features

- Live webcam monitoring
- Face-landmark based gaze/head-angle proxy
- Live microphone level monitoring
- Ctrl+Tab / Ctrl+PageUp / Ctrl+PageDown tab-switch event detection
- Manual `T` test event for demonstrations
- Rolling 10-sample gaze and audio features
- Random Forest classifier trained from project telemetry
- Live AI risk score combining the trained model with sustained behavioral evidence
- Full-screen monitoring dashboard
- Evidence messages explaining why risk increased

## Project structure

```text
EduGuard/
├── AI_Student_code.ipynb
├── live_eduguard.py
├── video_telemetry.csv
├── system_events.csv
├── requirements_live_eduguard.txt
├── face_landmarker.task          # created automatically on first run
├── README.md
└── .gitignore
```

## Machine-learning pipeline

The training notebook follows this workflow:

1. Load video and system telemetry.
2. Align the streams using timestamp-based nearest matching.
3. Handle missing values.
4. Create rolling gaze/audio features.
5. Train a Random Forest classifier.
6. Evaluate precision/recall and select a conservative decision threshold.
7. Use the trained model as the ML component of the live prototype.

### Offline model result

On the project's held-out test split, the selected 0.90 model threshold achieved approximately:

- **Precision: 99.51%**
- **Recall: 61.78%**
- **False positives: 2 / 2000 test samples**

These are offline telemetry results and should **not** be interpreted as guaranteed live-world accuracy.

## Live risk logic

The live dashboard keeps the Random Forest as the ML core and adds a conservative behavioral-evidence layer.

Examples:

- Short glance away → low concern
- Sustained gaze deviation → increased concern
- Sustained elevated audio → increased concern
- Tab switch → strong evidence
- Multiple simultaneous signals → higher risk

This is intended to reduce the chance that a single noisy signal creates a high-risk decision.

## Installation

Use Python 3.13.x.

From the project folder:

```bash
py -m pip install -r requirements_live_eduguard.txt
```

## Run

```bash
py live_eduguard.py
```

On the first run, the program downloads the MediaPipe Face Landmarker model
(`face_landmarker.task`) into the project folder.

## Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `R` | Reset tab-switch counter |
| `T` | Simulate one tab-switch event for testing |

The live listener also recognizes:

- `Ctrl + Tab`
- `Ctrl + PageUp`
- `Ctrl + PageDown`

as tab-switch events.

## Testing the demo

### Normal behavior
Look at the screen normally and remain quiet.

Expected result: **NORMAL / low risk**.

### Gaze test
Look significantly away from the screen and maintain the position for several
seconds.

Expected result: gaze evidence should increase.

### Audio test
Create sustained audio near the microphone.

Expected result: audio evidence should increase if it crosses the configured threshold.

### Combined test
Look away + create sustained audio + press `T`.

Expected result: multiple evidence signals should combine into a higher risk score.

## Technical limitations

- The webcam feature is a gaze/head-angle **proxy**, not clinical eye tracking.
- The microphone value is a relative/calibrated-to-training-range level, **not physical dB SPL**.
- The model was trained on engineered telemetry rather than raw video/audio.
- The prototype does not identify phones, notes, hands, or specific gestures.
- A hand-to-face movement alone is therefore not treated as cheating.
- Keyboard detection is limited to the supported hotkey events and can vary by OS/application.
- Live performance can differ from offline test-set performance.

## Ethical considerations

EduGuard should be used as an assistive monitoring system rather than an automatic
disciplinary system. Real deployments should include:

- clear student consent and notice
- data minimization and secure storage
- human review of flagged sessions
- an appeal mechanism
- testing for false positives and demographic/device differences
- transparent retention and deletion policies

## Future improvements

- Browser-integrated tab/visibility events
- Phone/object detection
- Multi-person detection
- Better gaze calibration
- Speech/activity classification instead of raw audio level
- Session event logging and review dashboard
- Model calibration and cross-validation
- More representative, consented training data
