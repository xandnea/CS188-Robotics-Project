# GestureBot

**Intuitive Robot Arm Control via Hand Tracking**

## Prerequisites

- python3.10

## Run

To start the interactive environment:

```bash
python test.py
```

This should open two windows. One with the robosuite simulated robot and another with the camera feed and hand tracking overlay.

## Goals

Control a robot arm using hand gestures:

* Wrist movement → arm translation
* Finger/palm orientation → arm rotation
* Pinch gesture → open/close gripper

A user should be able to:
* turn a door handle
* stacking blocks
