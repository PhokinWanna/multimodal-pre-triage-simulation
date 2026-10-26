# CNN results chart generator

`generate_report_charts.py` creates a Model-02/Model-03 metric comparison and
confusion-matrix figure from recorded metrics and Model-03 test score CSVs. It
contains no metric values of its own and checks sample/confusion-matrix totals
and derived classification metrics before creating output.
Use the repository's `requirements-cnn.txt` environment to install its
Matplotlib, NumPy, and scikit-learn dependencies.

Example using the currently recorded reports:

```bash
python scripts/train/cnn/generate_report_charts.py \
  --model02-metrics scripts/train/cnn/model03/results/model02_reference_metrics.json \
  --model03-alpha025-metrics scripts/train/cnn/model03/results/focal_alpha_025/metrics.json \
  --model03-alpha075-metrics scripts/train/cnn/model03/results/focal_alpha_075/metrics.json \
  --model03-alpha025-scores scripts/train/cnn/model03/results/focal_alpha_025/test_scores.csv \
  --model03-alpha075-scores scripts/train/cnn/model03/results/focal_alpha_075/test_scores.csv \
  --output-dir /path/to/new-chart-output
```

The output directory must not exist. The script will not overwrite prior
reports or figures. It writes:

- `metrics_comparison.png`: comparison of ROC-AUC, trapezoidal PR-AUC, accuracy,
  precision, recall, specificity, and F1.
- `confusion_matrices.png`: confusion counts in explicit `Normal=0`,
  `Flushing=1` orientation.
- `metrics_comparison.csv`: data behind the metric chart.
- `source_metrics.json`: normalized values used by the chart generator.
- `REPORT.md`: provenance, caveats, and audit of the former hard-coded chart.

Model-02's source metrics are historical validation results, while Model-03
values are held-out results from a filename-prefix-grouped split. The prefix
does not verify person identity or rule out identity leakage. The
historical Model-02 chart must be labeled accordingly; only Model-03 alpha
0.25 versus 0.75 is a controlled split-matched comparison. The old script's
hard-coded matrix and class metrics are stale/unverified and are intentionally
not used.

Model-02's reported PR-AUC is trapezoidal area; Model-03 records both
trapezoidal PR-AUC and scikit-learn average precision. The cross-model chart
uses the same trapezoidal definition for all three results and recomputes
Model-03 PR-AUC from its score CSVs.
