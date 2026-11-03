# Duplicate check: verdict (2026-11-03)

**Result: no M3 test or development photo is a copy of a CNN training image.**

Method:
1. `check_duplicates.py` compared all 51 M3 photos with all 16,774 CNN training images (`data/dataset-cnn/Normal`, `Flushing`) using a 64-bit difference hash, including rotated (±5–15°, 90/180/270°) and mirrored versions of each M3 photo. It was calibrated on a known rotated + noisy copy (S01_body vs its Roboflow augmented version: distance 9, threshold 10).
2. The 64-bit hash produced 1,861 candidate pairs, mostly plain/low-texture images (white backgrounds, skin close-ups).
3. Each candidate was re-scored with grey-level correlation (32×32, best of 20 rotations/mirrors), and the closest training image of every M3 photo was **inspected visually** (`Claude outputs/tmp/dup_pairs.jpg`): all were different pictures.

Side observation (relevant to the skin CNN shortcut risk): the Flushing training images are mostly cropped clinical close-ups watermarked "©Dermnet.com" (some with black privacy bars), while Normal images are whole, ordinary face photos. The two classes therefore differ in source, framing and image quality, not only in redness.

Note: 9 test body photos and all 14 development photos come from the Roboflow "Medical Emergency Detection" posture dataset (CC BY 4.0), which contains augmented copies (rotation, noise, greyscale) of the same pictures. Development photos were chosen by eye to avoid copies of test photos (one copy of S01_body was found and excluded). This dataset was not used to build the posture rules.
