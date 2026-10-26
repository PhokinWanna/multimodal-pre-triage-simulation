# Model assets

`skin_classifier.tflite` is the TFLite model loaded by the current vision
engine. The engine expects a single float32 image tensor and reports the
model's positive output as the Flushing class. Model-03 research checkpoints
are stored separately with their experiment report and are not wired into the
kiosk.

The packaged TFLite model is not accompanied by a source training manifest
that proves parity with the latest CNN experiment. Do not treat the Model-03
results as validation of this deployed artifact. Verify preprocessing,
camera-region selection, and TFLite parity before relying on its predictions.
