# GestureBot

**Intuitive Robot Arm Control via Hand Tracking**
## Robosuite
Ensure you have Robosuite properly installed for your environment, see: https://robosuite.ai/docs/installation.html.
## Dependencies

```bash
uv sync
```

### MacOS

```bash
brew install zlib
export DYLD_LIBRARY_PATH="/usr/lib:/opt/homebrew/opt/zlib/lib"
```

## Run

To start the interactive environment:
```bash
python test.py
# on macOS
mjpython test.py
```

This should open two windows. One with the robosuite simulated robot and another with the camera feed and hand tracking overlay.

>On macOS only the simulator will open

## Goals

Control a robot arm using hand gestures:
* Left Hand sets the method of control for the Right Hand (Open → translation, Closed → rotation)
* Right Hand Wrist movement → arm translation
* Right Hand Finger/palm orientation → arm rotation
* Pinch gesture → open/close gripper

A user should be able to:
* turn a door handle
* stacking blocks
