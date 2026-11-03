"""Make sure no M3 photo is (a copy of) a CNN training/test image.

    python scripts/experiments/m3/check_duplicates.py [--training-dir data/dataset-cnn]

Each photo gets a 64-bit "difference hash" (dHash): a fingerprint that survives resizing,
re-saving as JPEG and small colour changes. Every M3 photo is compared with every image in
the training folders (Normal/ and Flushing/). A Hamming distance <= --max-distance (default 10
of 64 bits) is reported as a possible duplicate; exact file copies are always reported.
Also checks the M3 photos against each other (the same photo used twice).
Output: results/experiments/m3/duplicate_check.csv. Exit code 1 if anything is found.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from m3_common import IMAGE_EXTS, IMAGES_DIR, PROJECT_ROOT, RESULTS_DIR, write_csv


def dhash(path_or_image, size: int = 8) -> int:
    from PIL import Image
    im = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    with im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(im.tobytes())
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


# Augmented copies in public datasets are often rotated a little, mirrored or turned 90 degrees.
ROTATIONS = (0, -15, -10, -5, 5, 10, 15, 90, 180, 270)


def variant_hashes(path: Path) -> list[int]:
    """Fingerprints of the photo under the usual dataset augmentations (rotation, mirror)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        base = ImageOps.exif_transpose(im).convert("L")
        hashes = []
        for mirror in (False, True):
            b = ImageOps.mirror(base) if mirror else base
            for angle in ROTATIONS:
                r = b.rotate(angle, expand=angle % 90 != 0, fillcolor=0) if angle else b
                if angle % 90:   # crop the black corners added by the rotation
                    w, h = b.size
                    r = ImageOps.fit(r, (int(w * 0.8), int(h * 0.8)))
                hashes.append(dhash(r.copy()))
    return hashes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_entry(p: Path):
    try:
        return str(p), dhash(p), sha256(p)
    except Exception:
        return str(p), None, None


def training_fingerprints(paths: list[Path], cache_file: Path, time_budget: float) -> tuple[dict, bool]:
    """{path: (dhash, sha256)} for the training images, cached on disk so a long job can resume."""
    import json
    import time
    from concurrent.futures import ProcessPoolExecutor
    cache = json.loads(cache_file.read_text()) if cache_file.is_file() else {}
    todo = [p for p in paths if str(p) not in cache]
    start = time.time()
    if todo:
        print(f"Fingerprinting {len(todo)} training images ({len(cache)} already cached) ...")
        with ProcessPoolExecutor() as pool:
            for i, (k, h, sh) in enumerate(pool.map(_train_entry, todo, chunksize=64), 1):
                cache[k] = (h, sh)
                if i % 1000 == 0:
                    cache_file.write_text(json.dumps(cache))
                    print(f"  {len(cache)}/{len(paths)}")
                if time_budget and time.time() - start > time_budget:
                    pool.shutdown(cancel_futures=True)
                    break
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
    done = all(str(p) in cache for p in paths)
    return cache, done


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--training-dir", type=Path, default=PROJECT_ROOT / "data" / "dataset-cnn")
    ap.add_argument("--max-distance", type=int, default=10)
    ap.add_argument("--time-budget", type=float, default=0,
                    help="stop fingerprinting after N seconds and resume on the next run (0 = no limit)")
    ap.add_argument("--extra-dir", type=Path, action="append", default=[],
                    help="another image folder to compare against (e.g. the posture dataset)")
    args = ap.parse_args()

    m3 = images_in(IMAGES_DIR / "test") + images_in(IMAGES_DIR / "dev")
    if not m3:
        print(f"No photos found in {IMAGES_DIR}/test or /dev")
        return 1
    train = [p for sub in ("Normal", "Flushing") for p in images_in(args.training_dir / sub)]
    for extra in args.extra_dir:
        train += images_in(extra)
    if not train:
        print(f"WARNING: no training images found under {args.training_dir}/Normal or /Flushing; "
              "only checking the M3 photos against each other.")

    print(f"Fingerprinting {len(m3)} M3 photos and {len(train)} training images ...")
    m3_h = {p: (variant_hashes(p), sha256(p)) for p in m3}
    cache, done = training_fingerprints(train, RESULTS_DIR / "training_fingerprints.json", args.time_budget)
    if not done:
        print("Not finished yet: run the same command again to continue.")
        return 2
    rows = []
    for p in train:
        th, ts = cache[str(p)]
        if th is None:
            continue
        for q, (qhs, qs) in m3_h.items():
            dist = min(bin(th ^ qh).count("1") for qh in qhs)
            if qs == ts or dist <= args.max_distance:
                rows.append({"m3_photo": str(q.relative_to(IMAGES_DIR)), "matches": str(p.relative_to(PROJECT_ROOT)),
                             "hamming_distance": dist, "exact_copy": qs == ts})

    items = list(m3_h.items())
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            (pa, (ha, sa)), (pb, (hb, sb)) = items[a], items[b]
            dist = min(bin(ha[0] ^ h).count("1") for h in hb)
            if sa == sb or dist <= args.max_distance:
                rows.append({"m3_photo": str(pa.relative_to(IMAGES_DIR)), "matches": "M3:" + str(pb.relative_to(IMAGES_DIR)),
                             "hamming_distance": dist, "exact_copy": sa == sb})

    out = RESULTS_DIR / "duplicate_check.csv"
    write_csv(out, sorted(rows, key=lambda r: r["hamming_distance"]),
              ["m3_photo", "matches", "hamming_distance", "exact_copy"])
    if rows:
        print(f"\n{len(rows)} possible duplicate(s) -> look at each pair in {out}")
        for r in rows[:20]:
            print(f"  {r['m3_photo']}  ~  {r['matches']}  (distance {r['hamming_distance']}{', EXACT COPY' if r['exact_copy'] else ''})")
        print("Replace any photo that really is the same picture, then run again.")
        return 1
    print(f"\nNo duplicates found (threshold {args.max_distance}/64). Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
