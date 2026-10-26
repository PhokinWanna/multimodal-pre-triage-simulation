import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PATCH_SIZE = 64              # CNN input size (64x64)
MIN_SKIN_RATIO = 0.70        # Minimum share of skin pixels in a patch
MAX_YAW_THRESHOLD = 0.25     # Head-yaw threshold for skipping a cheek

# Face landmark IDs (MediaPipe Face Mesh)
NOSE_ID = 1                  # Nose tip
FOREHEAD_ID = 10             # Forehead center
CHEEK_L_EDGE_ID = 234        # Left cheek edge
CHEEK_R_EDGE_ID = 454        # Right cheek edge
CHEEK_L_ID = 117             # Left cheekbone
CHEEK_R_ID = 346             # Right cheekbone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset-cnn"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset-cnn" / "patches"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def is_valid_skin_color(patch_bgr):
    """
    Check that a patch shows real skin (rejects green/blue, very dark, and glare).
    """
    if patch_bgr is None:
        return False
        
    h_patch, w_patch = patch_bgr.shape[:2]
    if h_patch != PATCH_SIZE or w_patch != PATCH_SIZE:
        return False

    # Split color channels
    b_chan, g_chan, r_chan = cv2.split(patch_bgr)

    r_mean = float(np.mean(r_chan))
    g_mean = float(np.mean(g_chan))
    b_mean = float(np.mean(b_chan))

    # 1. Skin rule: red must dominate blue
    if b_mean >= r_mean:
        return False

    # 2. Reject patches that are too dark or too bright
    if r_mean < 40 or (r_mean > 245 and g_mean > 245 and b_mean > 245):
        return False

    # 3. Check skin tone in YCrCb space
    ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)
    y_chan, cr_chan, cb_chan = cv2.split(ycrcb)

    skin_mask = (cr_chan >= 133) & (cr_chan <= 173) & (cb_chan >= 77) & (cb_chan <= 127)
    skin_ratio = float(np.sum(skin_mask)) / float(PATCH_SIZE * PATCH_SIZE)

    return skin_ratio >= MIN_SKIN_RATIO


def extract_square_patch(image_bgr, center_x, center_y, patch_size=64):
    """
    Crop a square patch_size x patch_size patch without distorting the aspect ratio.
    """
    h, w = image_bgr.shape[:2]
    half = patch_size // 2

    x1 = int(center_x - half)
    y1 = int(center_y - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None

    return image_bgr[y1:y2, x1:x2].copy()


# ==========================================
# MAIN EXTRACTION PIPELINE
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing Normal/ and Flushing/ source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New output directory for extracted patches; it must not exist.",
    )
    return parser.parse_args()


def process_dataset(dataset_dir, output_dir):
    classes = ["Normal", "Flushing"]
    valid_extensions = ('.jpg', '.jpeg', '.png')
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {dataset_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Patch output directory already exists; refusing to overwrite it: {output_dir}"
        )
    for class_name in classes:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required class directory is missing: {class_dir}")
        if not any(
            path.is_file() and path.suffix.lower() in valid_extensions
            for path in class_dir.iterdir()
        ):
            raise ValueError(f"No supported source images found in {class_dir}")

    for class_name in classes:
        in_class_dir = dataset_dir / class_name
        out_class_dir = output_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            path.name
            for path in in_class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in valid_extensions
        )
        saved_count = 0

        print(f"[*] Processing Class: {class_name} ({len(image_files)} files)")

        for filename in image_files:
            file_path = in_class_dir / filename
            
            frame_bgr = cv2.imread(str(file_path))
            if frame_bgr is None:
                continue

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                continue

            # Landmarks of the first detected face
            landmarks = None
            for face in results.multi_face_landmarks:
                landmarks = face.landmark
                break

            if landmarks is None:
                continue

            # 1. Estimate head yaw
            nose_x = landmarks[NOSE_ID].x
            left_cheek_edge_x = landmarks[CHEEK_L_EDGE_ID].x
            right_cheek_edge_x = landmarks[CHEEK_R_EDGE_ID].x

            dist_left = abs(nose_x - left_cheek_edge_x)
            dist_right = abs(right_cheek_edge_x - nose_x)
            total_span = dist_left + dist_right

            skip_left = False
            skip_right = False

            if total_span > 0:
                if (dist_left / total_span) < MAX_YAW_THRESHOLD:
                    skip_left = True
                if (dist_right / total_span) < MAX_YAW_THRESHOLD:
                    skip_right = True

            # 2. Center point of each zone
            forehead_pt = landmarks[FOREHEAD_ID]
            left_cheek_pt = landmarks[CHEEK_L_ID]
            right_cheek_pt = landmarks[CHEEK_R_ID]

            zones_to_extract = []
            zones_to_extract.append(("Forehead", forehead_pt.x * w, forehead_pt.y * h))
            
            if not skip_left:
                zones_to_extract.append(("Zone_L", left_cheek_pt.x * w, left_cheek_pt.y * h))
            if not skip_right:
                zones_to_extract.append(("Zone_R", right_cheek_pt.x * w, right_cheek_pt.y * h))

            # 3. Crop and quality-filter each patch
            base_name, _ = os.path.splitext(filename)

            for zone_name, cx, cy in zones_to_extract:
                patch = extract_square_patch(frame_bgr, cx, cy, patch_size=PATCH_SIZE)

                if is_valid_skin_color(patch):
                    out_name = f"{base_name}_{zone_name}.jpg"
                    out_path = out_class_dir / out_name
                    if not cv2.imwrite(str(out_path), patch):
                        raise OSError(f"Could not write extracted patch: {out_path}")
                    saved_count += 1

        print(f"[+] Finished {class_name}: Saved {saved_count} clean patches.")

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args.source_dir, args.output_dir)