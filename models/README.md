# Model assets

`skin_classifier.tflite` is the skin CNN loaded by the kiosk's vision engine
(`src/triage_kiosk/vision/skin.py`). It is the Model-03 focal-loss α = 0.75 best
checkpoint, converted to TFLite. `skin_classifier.json` records its source checkpoint
(SHA-256), the decision threshold chosen on the validation set, and the expected input:
64x64 forehead/cheek patches, BGR → CIELAB, divided by 255, with the patch
probabilities averaged.

`legacy/skin_classifier_original.tflite` is the original model, kept only for the
original-vs-fixed comparison in the M3 experiment (`SKIN_PIPELINE=legacy_whole_face`).

The evaluation is patch-level on a prefix-grouped split (not verified person-disjoint),
so treat the predictions as research output, not a clinical measurement.
