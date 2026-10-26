# Vision subsystem

`engine.py` uses MediaPipe Pose and Face Mesh for camera telemetry and loads
[`models/skin_classifier.tflite`](../../../models/skin_classifier.tflite)
through LiteRT when available.
Model inputs are resized to the TFLite input dimensions, converted from BGR to
CIELAB, and scaled to float32 `[0, 1]`.

If TFLite initialization or inference fails, the current code logs a warning
and falls back to a basic HSV color heuristic. That heuristic is experimental,
not CNN inference or a clinically validated classifier. The deployed camera
face crop is not yet verified as equivalent to the dataset's extracted
forehead/cheek patches.
