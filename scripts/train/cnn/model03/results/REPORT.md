# Train-03 CNN experiment report

## Objective

Investigate Model-02's reported low Flushing recall while preserving strong
ranking performance, and retain reproducible experiment evidence.

## Findings before training

The current trainer uses `Normal=0`, `Flushing=1` and a focal-loss alpha of
`0.25`. In the implemented loss, alpha is applied to positive examples, so
Flushing examples receive weight `0.25` and Normal examples receive weight
`0.75`. This is opposite the intended emphasis if the goal is to penalize
missed Flushing cases more heavily.

The existing splitter also split groups separately within each class. Using
the source filename prefix preserved by the image extractor, 1,033 numeric
prefix groups were found in more than one of the old train/validation/test
splits. The prefix is not a verified person identifier, and different
prefixes may refer to the same person. Neither the old nor rebuilt split
establishes person-level separation or rules out identity leakage.

## Method

- Combined the 24,938 existing image patches from the old train/validation/test
  folders without changing or copying the source images.
- Created a deterministic 70/15/15 split by leading numeric filename prefix
  across both classes. The resulting split has 8,097 groups and no prefix
  overlap: train 17,495 images, validation 3,695, test 3,748. This grouping
  does not verify person identities or guarantee person-disjoint evaluation.
- Kept Model-02's architecture, augmentations, preprocessing, seed, batch size,
  learning rate, epoch limit, and early-stopping/checkpoint approach constant.
- Compared focal-loss positive-class alpha values of 0.25 and 0.75.
- Selected candidate thresholds from validation data at precision floors of
  0.95, 0.90, 0.80, and 0.70; applied those thresholds unchanged to test data.
- Reported exact scikit-learn ROC-AUC and average precision. The separately
  stored `best_validation_auc_keras_approx` is Keras's finite-threshold
  checkpoint-monitor value, not the exact ROC-AUC.

## Results

| Positive alpha | Test ROC-AUC | Test average precision | Test precision at 0.50 | Test recall at 0.50 | Test specificity at 0.50 | Test F1 at 0.50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.9842 | 0.9360 | 0.9442 | 0.6614 | 0.9891 | 0.7779 |
| 0.75 | 0.9926 | 0.9726 | 0.9537 | 0.8570 | 0.9884 | 0.9028 |

The alpha 0.75 model's validation-selected threshold of 0.479992 (maximum
validation recall among thresholds with precision at least 0.95) resulted in
test precision 0.9495, recall 0.8729, specificity 0.9870, and F1 0.9096.
Selecting the validation threshold under a 0.90 precision floor (0.365924)
resulted in test precision 0.8979, recall 0.9352, specificity 0.9703, and F1
0.9162.

At threshold 0.50, alpha 0.75 identified 701 of 818 Flushing test patches and
missed 117; it produced 34 false positives among 2,930 Normal test patches.

## Interpretation

Two issues contributed to the original recall concern:

1. AUC measures ranking across thresholds; it does not guarantee high recall at
   the default threshold of 0.50.
2. The original focal-loss alpha assigned less weight to the positive Flushing
   class. Increasing that weight to 0.75 materially improved test recall at
   0.50 while preserving (and in this run increasing) test ROC-AUC.

The old Model-02 validation report was reproduced at threshold 0.50 as
ROC-AUC 0.9738, precision 0.9821, recall 0.2644, with TN=2,955, FP=4, FN=612,
TP=220. This is not directly comparable to the new results because the old
split has cross-split group overlap and the new models were retrained on the
globally grouped split.

For further context, the earlier Model-01 baseline (same architecture, binary
cross-entropy loss, unadjusted 0.5/0.5 class weighting) reported on its
historical holdout test set (3,719 patches): accuracy 0.7989, ROC-AUC 0.9142,
precision 0.5209, recall 0.8832, specificity 0.7756, F1 0.6553, with TN=2,260,
FP=654, FN=94, TP=711. See `model01_reference_metrics.json`.

## Limitations and follow-up

- **Identity leakage audit is inconclusive.** The current checkout has no
  source images, extracted `cnn_dataset`, prior `split_dataset`, or source-to-
  person mapping to inspect. Saved score files contain sample indices, labels,
  and probabilities only. The rebuilt split guarantees that a numeric filename
  prefix stays in one partition, but the owner reports that different prefixes
  may refer to the same person. No person-level overlap rate can be calculated
  from the available evidence, so identity leakage is not ruled out. No face
  identity matching or new training was performed.
- The owner reports that different prefixes can correspond to the same person.
  The raw source dataset and original metadata are unavailable here, so no
  reliable identity linkage can be reconstructed from the current checkout.
- A single seed and one rebuilt split were used. Repeat with additional seeds
  or an external cohort before claiming robust generalization.
- These image-patch metrics do not validate the full triage workflow or its
  deployed inference path. The app preprocessing mismatch has not been changed
  or evaluated in this experiment.
- A precision/recall operating point is a product and clinical-risk decision.
  The report offers candidate trade-offs but does not select a final
  deployment threshold.

## Evidence files

- `experiment.json`: split seed, counts, preprocessing, and grouping caveat.
- `FINAL_SUMMARY.md`: detailed Model-01/Model-02/Model-03 parameter comparison,
  methodology, data counts, results, interpretation, and figure guide.
- `MODEL03_TECHNICAL_APPENDIX.md` and `MODEL03_LAYER_STRUCTURE.csv`: exact
  architecture, formulas, activations, learned-parameter counts, loss,
  optimizer, and training/evaluation procedure.
- `model01_reference_metrics.json` and `model02_reference_metrics.json`:
  historical baseline metrics used in the comparison figures.
- `figures/`: four-model metric comparison and confusion matrices, training curves,
  threshold trade-off, and data split distribution. The older ROC/PR figure is
  excluded because it predates the filename-prefix identity clarification.
- `split_counts.csv`: image and group counts for each split.
- `focal_alpha_025/` and `focal_alpha_075/`: model checkpoints, training
  histories, metrics, and validation/test score CSVs. Score CSVs omit source
  filenames and group IDs.
- `comparison.json`: machine-readable comparison of both controlled runs.

The historical Model-01 values come from its holdout test set and the
historical Model-02 values from its original validation set; Model-03 values use the rebuilt numeric-prefix-
grouped test set. They are presented side by side for context, not as a
perfectly matched benchmark or verified person-level evaluation. The two
Model-03 alpha runs are the controlled comparison on the same prefix-grouped
split.
