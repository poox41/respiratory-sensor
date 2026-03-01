# Desktop BreathHeart Demo (Python + Qt)

This is a Windows desktop version of the BreathHeartDemo Android app.

## Features (planned)
- BLE scan/connect (same UUIDs as Android)
- Real-time waveforms (raw/resp/hr)
- Heart rate / Resp rate estimation (ported from Android)
- Amplitude and time-window controls

## Run (dev)
1. Install Python 3.10+ and create a venv
2. Install deps: `pip install -r requirements.txt`
3. Run: `python -m src.main`

## Packaging (single EXE)
Use PyInstaller after deps are installed.
