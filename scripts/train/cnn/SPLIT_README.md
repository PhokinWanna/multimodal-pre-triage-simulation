# Numeric-prefix-grouped CNN dataset splitter

The splitter creates the `train/val/test` directory contract expected by
[`train_skin_cnn-02.py`](./train_skin_cnn-02.py):

```text
split_dataset/
├── train/
│   ├── Normal/
│   └── Flushing/
├── val/
│   ├── Normal/
│   └── Flushing/
├── test/
│   ├── Normal/
│   └── Flushing/
└── split_summary.json
```

It groups all image crops globally by the leading numeric filename prefix
(for example `00007_Flushing_Zone_L.jpg`), including across classes, then
stratifies these prefix groups by which target classes they contain. The
prefix is a grouping heuristic, not a verified person identifier. The source
images combine multiple datasets, and different prefixes may refer to the
same person. This split therefore does not establish person-disjoint
partitions or rule out identity leakage.

## Run

From the repository root, place authorized source images locally in
`data/dataset-cnn/{Normal,Flushing}/`, activate the Python 3.11 environment
created from `requirements-cnn.txt`, then run:

```bash
python scripts/data/dataset_extraction-CNN.py
python scripts/train/cnn/split_dataset.py
python scripts/train/cnn/train_skin_cnn-02.py
```

The extractor writes patches to `data/dataset-cnn/patches/`. The splitter reads those
patches and writes `data/dataset-cnn/splits/`. The trainer reads that split and writes
new run artifacts under `results/cnn/model02/training_runs/`.

To specify another extracted dataset or a fresh output destination:

```bash
python scripts/train/cnn/split_dataset.py \
  --source-dir /path/to/patches \
  --output-dir /path/to/new-split \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

Ratios must be positive and sum to 1.0. Every prefix-group class-signature
must have enough groups for stratification into all three splits. If not, the
script stops with a clear error before creating output.

## Safety and verification

- Input must have nonempty `Normal/` and `Flushing/` directories.
- Supported image files are JPG, JPEG, and PNG.
- Every image filename must begin with a numeric prefix and `_`.
- All crops/classes with the same prefix are assigned to one partition.
- Source and output paths may not be equal or nested.
- The output path must not already exist; the splitter never deletes or
  overwrites a prior split.
- Output is first prepared in a temporary sibling directory, then renamed into
  place after copying and metadata writing complete.
- `split_summary.json` records seed, requested ratios, class counts, prefix
  group counts, and prefix-disjointness without listing prefixes. This is not
  proof of person-level disjointness.

The splitter only copies image patches. It does not perform image resizing,
color conversion, augmentation, or training; those are defined by the trainer.
