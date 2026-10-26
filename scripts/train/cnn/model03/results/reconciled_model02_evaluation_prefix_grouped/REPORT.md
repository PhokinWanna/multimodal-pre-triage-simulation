# Reconciled CNN evaluation charts

Charts and the CSV table in this folder are generated from the supplied metric
JSON files and Model-03 test score CSVs. No values are hard-coded into the
plotting code. The script checks
that each confusion matrix sums to its reported sample count and that
accuracy/precision/recall/specificity/F1 agree with the confusion counts
(allowing for rounded summary values).

## Results

| Model | Evaluation split | N | ROC-AUC | PR-AUC (trapezoidal) | Precision | Recall | Specificity | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model-02 original validation | Original Model-02 validation split | 3,791 | 0.9738 | 0.9203 | 0.9821 | 0.2644 | 0.9986 | 0.4167 |
| Model-03 alpha 0.25 | Filename-prefix-grouped test | 3,748 | 0.9842 | 0.9359 | 0.9442 | 0.6614 | 0.9891 | 0.7779 |
| Model-03 alpha 0.75 | Filename-prefix-grouped test | 3,748 | 0.9926 | 0.9725 | 0.9537 | 0.8570 | 0.9884 | 0.9028 |

The Model-02 row is the historical validation report. The Model-03 rows are
from a rebuilt filename-prefix-grouped test split. These are different splits
and must not be presented as a matched benchmark. Filename prefixes are not
verified person identities; person-level separation and absence of identity
leakage are not established. The controlled comparison is Model-03 alpha 0.25
versus Model-03 alpha 0.75, which share the same prefix-grouped split.

## Legacy chart audit

The former `generate_report_charts.py` contained a hard-coded 3,719-sample
matrix and hard-coded class metrics, with unsupported patient-level and
clinical labels. Those values do not match the verified Model-02 report
(3,791 validation samples; TN=2,955, FP=4, FN=612, TP=220) and are not
reproduced here. Treat charts produced by that old script as stale/unverified.
The existing Model-02 `training_report.png` matches the reported confusion
counts and AUC, but it is still a validation result at threshold 0.50 and
filename-prefix groups crossed boundaries in the old split. Neither the old
nor rebuilt split establishes independent person-level validation.

No new Model-02 training, threshold tuning, or test evaluation was performed
for this chart regeneration. Model-03 metrics are the previously recorded
experiment results.

The Model-02 text report's `PR-AUC` (0.9203) is the trapezoidal area under its
precision–recall curve. Model-03 also stores `average_precision`, a distinct
scikit-learn metric. For the cross-model chart, PR-AUC was recomputed from each
Model-03 test score CSV with the same trapezoidal integration convention as
Model-02. Average precision is retained in Model-03 run JSON but is not mixed
into this cross-model PR-AUC comparison.
