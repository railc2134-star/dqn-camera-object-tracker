# DQN Camera Object Tracker

A Deep Q-Network (DQN) implementation that learns to move a virtual crosshair toward a colored object detected by a webcam.

## Features

- Custom reinforcement learning environment
- Real-time webcam input
- HSV color based object detection
- Deep Q-Network implemented in PyTorch
- Experience replay buffer
- Target network updates
- Epsilon-greedy exploration

## Environment

The environment captures frames from a webcam and detects an object using an HSV color mask.

The agent controls a virtual crosshair and can move in four directions:

- Up
- Down
- Left
- Right

The goal is to place the crosshair near the detected object.

The agent receives a positive reward when it moves closer to the object and a negative reward when it moves farther away.

Episodes terminate when the crosshair reaches the object.

## Requirements

Install the required libraries:

```bash
pip install torch opencv-python numpy
```

## Training

Set

```python
train = True
```

and run

```bash
python tracker.py
```

## Evaluation

Set

```python
train = False
```

and run

```bash
python tracker.py
```

The trained model is loaded from

```text
nett.pth
```

The webcam feed will open and the learned policy will control the crosshair.

## Notes

The HSV color range can be adjusted by changing

```python
lower_hsv
upper_hsv
```

to track different colored objects.
