# Train-03 CNN Fine-Tuning Experiment

This experiment investigates the low Flushing recall reported for Model-02 without
modifying Model-02 or its source dataset.

## Important split-integrity limitation

The existing splitter formed groups separately inside each class. Image names
preserve a numeric filename prefix (for example,
`00007_Flushing_Zone_L.jpg`), so the same prefix could be assigned to different
old splits through the other class. The inspected old split contains 8,097
numeric-prefix groups, and 1,033 occur in more than one old split.

`train_and_evaluate.py` rebuilds a deterministic 70/15/15 split across the
combined image pool, keeping each leading numeric filename prefix in one
partition across both classes. The prefix is a grouping heuristic, not a
verified person identifier. The dataset combines multiple sources, and the
owner reports that different prefixes may still refer to the same person.
Person-level separation and absence of identity leakage therefore remain
unverified. The script reads the existing split directory only; it does not
move, rename, or copy the source images.

## What is compared

The script uses the Model-02 image preprocessing and CNN architecture, then
trains two controlled focal-loss variants with the same seed and grouped split:

- `focal_alpha_025`: the original positive-class alpha setting.
- `focal_alpha_075`: gives Flushing (label 1) the higher focal-loss weight.

For each run it saves the best validation-AUC checkpoint, epoch history, split
counts, validation/test score metrics, and per-image labels/scores without
source filenames or person IDs. Thresholds are selected from validation data
at several precision floors, then applied unchanged to the held-out test data.
Test data is not used to select a threshold.

## Initial results

The dataset provenance, license, access, and preparation limitations are
documented in [`docs/data/README.md`](../../../../docs/data/README.md).

The old Model-02 report was reproduced on its original validation split:
ROC-AUC 0.9738, precision 0.9821, and recall 0.2644 at threshold 0.50. Because
the old split has cross-split numeric-prefix overlap, and neither split has
verified person-level separation, those values are exploratory rather than an
independent-person evaluation.

On the rebuilt numeric-prefix-grouped split, the positive-class focal alpha
change improved both recall and AUC for this experiment. These remain
patch-level results and are not evidence of person-independent performance:

| Positive-class alpha | Test ROC-AUC | Test average precision | Test precision at 0.50 | Test recall at 0.50 | Test specificity at 0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.25 (original weighting) | 0.9842 | 0.9360 | 0.9442 | 0.6614 | 0.9891 |
| 0.75 (Flushing upweighted) | 0.9926 | 0.9726 | 0.9537 | 0.8570 | 0.9884 |

For alpha 0.75, the validation-selected threshold 0.47999 (chosen to maximize
recall while keeping validation precision at least 0.95) produced test
precision 0.9495, recall 0.8729, and specificity 0.9870. A lower
validation-selected threshold of 0.36592 (precision floor 0.90) produced test
precision 0.8979 and recall 0.9352. This illustrates the remaining operating
point trade-off; it is not a clinical threshold recommendation.

Detailed confusion matrices, all threshold results, and score files are in
`results/focal_alpha_025/` and `results/focal_alpha_075/`. `results/REPORT.md`
summarizes the experiment and its limitations. `results/FINAL_SUMMARY.md` is the
complete Model-02/Model-03 comparison, method, split description, interpretation,
and visualization guide. The `results/figures/` folder contains the generated
plots; the older ROC/PR figure was excluded because it predates the split
identity clarification.

Regenerate the figures (the Model-02 checkpoint and original validation images
are read-only inputs):

```bash
python generate_visualizations.py \
  --source-dir /path/to/your/split_dataset \
  --model02-path /path/to/your/Model-02-checkpoint.keras
```

## Run

Create a Python 3.11 environment and install the CNN dependency group from the
repository root:

```bash
python3.11 -m venv venv_train
source venv_train/bin/activate
python -m pip install -r requirements-cnn.txt
cd scripts/train/cnn/model03
python train_and_evaluate.py \
  --source-dir /path/to/your/split_dataset
```

Results are written to `results/` by default. They can be placed elsewhere with
`--output-dir`. The source directory must contain `train`, `val`, and `test`
subdirectories, each with `Normal` and `Flushing` class folders.

These are research experiments, not a clinical validation or a deployment
recommendation. The preferred precision/recall operating point still requires
agreement on the intended triage trade-off. Review the source data permissions
and identity-leakage limitations before sharing the dataset or presenting the
metrics as person-level evidence.
