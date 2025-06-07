# GestureBot

**Intuitive Robot Arm Control via Hand Tracking**

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

* Wrist movement → arm translation
* Finger/palm orientation → arm rotation
* Pinch gesture → open/close gripper

A user should be able to:
* turn a door handle
* stacking blocks
