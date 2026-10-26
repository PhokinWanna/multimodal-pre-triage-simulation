# Train-02 CNN trainer

`train_skin_cnn-02.py` trains the binary classifier with `Normal=0` and
`Flushing=1`. It expects this dataset contract:

```text
split_dataset/
├── train/
│   ├── Normal/
│   └── Flushing/
├── val/
│   ├── Normal/
│   └── Flushing/
└── test/
    ├── Normal/
    └── Flushing/
```

Supported images are JPEG and PNG. Each filename must begin with a numeric
prefix and an underscore, such as `00007_Flushing_Zone_L.jpg`. The trainer
checks that all three splits exist, both classes are nonempty, images decode,
and no numeric filename prefix crosses split boundaries. Prefix grouping is
only a filename-based heuristic: it does not prove that the same person's
images stay together, and different prefixes may belong to the same person.
Use the repository's corrected splitter documented in
[`SPLIT_README.md`](./SPLIT_README.md) to create a prefix-disjoint split.

## Run

From the repository root:

```bash
python scripts/train/cnn/train_skin_cnn-02.py \
  --data-dir data/dataset-cnn/splits
```

Or provide an explicit, new output directory:

```bash
python scripts/train/cnn/train_skin_cnn-02.py \
  --data-dir /path/to/split_dataset \
  --output-dir /path/to/new-run-directory \
  --seed 42
```

An existing output directory is never overwritten. By default, results go to
a UTC-timestamped new folder under
`results/cnn/model02/training_runs/`.

The trainer preserves Model-02's architecture and default loss settings. It
uses on-demand batches with OpenCV resize to 64×64, BGR-to-CIELAB conversion,
and float32 `/255` scaling. Early stopping monitors validation AUC. Thresholds
are chosen from validation data at several precision floors and applied
unchanged to the held-out test scores. The test split is not used for gradient
updates or threshold selection.

Use the [numeric-prefix-grouped splitter](./SPLIT_README.md) to create
the required split; the trainer checks that filename-prefix groups do not
cross partitions. This does not prove person-level separation.

Each run folder contains the model checkpoint, TFLite export, model summary,
training history, test/validation scores, JSON and text metrics, a training and
evaluation plot, and run configuration/version/split metadata. No original
model, dataset, or prior result file is overwritten.

The default settings are 35 epochs maximum, batch size 32, seed 42, legacy Adam
with learning rate `1e-4`, and binary focal loss with alpha `0.25`, gamma `2`.
Install the project's CNN dependency group from the repository root with
`python -m pip install -r requirements-cnn.txt`. The script's `--help` output
lists the available run options.
