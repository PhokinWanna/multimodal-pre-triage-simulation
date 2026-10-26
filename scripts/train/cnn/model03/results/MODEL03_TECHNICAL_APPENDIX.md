# Model-03 technical appendix: layers, parameters, equations, and workflow

This appendix documents the exact architecture and training setup of the
Model-03 alpha 0.75 checkpoint. It was verified by loading the saved Keras
checkpoints. The Model-03 alpha 0.25 run and Model-02 checkpoint have the same
layer types, output shapes, and parameter count; only focal-loss alpha differs
between the two Model-03 runs. Model-02 and Model-03 learned parameter values
are different because their checkpoints were trained separately.

See [`MODEL03_LAYER_STRUCTURE.csv`](./MODEL03_LAYER_STRUCTURE.csv) for the
machine-readable layer table.

## 1. What “weight” means here

There are two different meanings of “weight”:

1. **Learned model weights** are the values learned by the convolution and
   dense layers during training. They are stored in the checkpoint files:
   [`focal_alpha_075/best_model.keras`](./focal_alpha_075/best_model.keras)
   and [`focal_alpha_025/best_model.keras`](./focal_alpha_025/best_model.keras).
   The model has **13,217 total parameters**: 12,769 trainable parameters and
   448 non-trainable Batch Normalization moving-statistic parameters. The exact
   learned values are binary tensors in the checkpoints, rather than one scalar
   hyperparameter.
2. **Class weight in the loss** is the focal-loss alpha coefficient. For
   Model-03 alpha 0.75, positive Flushing examples receive alpha 0.75 and
   negative Normal examples receive weight 0.25. Model-03 alpha 0.25 reverses
   those weights. The alpha-0.75 setting is the controlled change that targets
   the false-negative/recall problem.

No class-weight dictionary or sample-weight argument is additionally supplied
to `model.fit()`.

## 2. Layer-by-layer architecture

Tensor layouts below are batch-first, channels-last (`batch, height, width,
channels`). The table describes inference tensor shapes; random augmentations
are active only during training.

| # | Layer | Output shape (excluding batch) | Trainable parameters | Operation |
| ---: | --- | --- | ---: | --- |
| 1 | Input | 64 × 64 × 3 | 0 | A normalized CIELAB patch |
| 2 | Random horizontal flip | 64 × 64 × 3 | 0 | Randomly mirror training patches left-to-right |
| 3 | Random rotation | 64 × 64 × 3 | 0 | Random rotation within factor 0.05 |
| 4 | Conv2D, 32 filters, 3 × 3, stride 2, same, ReLU | 32 × 32 × 32 | 896 | Spatial/color feature extraction with a standard convolution |
| 5 | Batch Normalization | 32 × 32 × 32 | 64 | Normalize each feature channel, then learn scale and offset |
| 6 | SeparableConv2D, 64 filters, 3 × 3, same, ReLU | 32 × 32 × 64 | 2,400 | Depthwise spatial filtering followed by pointwise channel mixing |
| 7 | Batch Normalization | 32 × 32 × 64 | 128 | Normalize each feature channel |
| 8 | Max Pooling, 2 × 2 | 16 × 16 × 64 | 0 | Downsample by taking local maxima |
| 9 | SeparableConv2D, 128 filters, 3 × 3, same, ReLU | 16 × 16 × 128 | 8,896 | Extract higher-level features with depthwise-separable convolution |
| 10 | Batch Normalization | 16 × 16 × 128 | 256 | Normalize each feature channel |
| 11 | Global Average Pooling | 128 | 0 | Average each feature map over its 16 × 16 spatial positions |
| 12 | Dropout, 0.30 | 128 | 0 | Randomly zero 30% of units during training |
| 13 | Dense, 1 unit, sigmoid | 1 probability | 129 | Map 128 features to the Flushing probability |
|  | **Total** |  | **12,769 trainable / 13,217 total** |  |

The parameter totals count the 448 Batch Normalization moving-mean and
moving-variance values as non-trainable parameters. Their trainable scale and
offset values are included in the trainable count.

### Activation functions and hidden layers

- The first convolution and both separable convolutions apply **ReLU**:
  `ReLU(z) = max(0, z)`.
- Batch Normalization follows the convolution operations. During training it
  uses batch statistics; during inference it uses accumulated moving statistics.
- The dense output applies **sigmoid**:
  `sigmoid(z) = 1 / (1 + exp(-z))`, producing one score in `[0, 1]` for
  Flushing. A score is converted into a binary prediction by comparing it to a
  threshold (0.50 for the default metric row).
- There are no fully connected hidden dense layers. The CNN's convolution
  feature maps and their nonlinearities are the learned hidden representation;
  global average pooling reduces that representation to 128 features, followed
  by dropout and the one-unit output layer.

## 3. Forward calculation, from pixels to probability

Let the source patch be a 64 × 64 BGR image.

### 3.1 Preprocessing

OpenCV first ensures the patch size is 64 × 64, converts BGR to CIELAB, then
scales each channel:

`X = float32(CIELAB(resize(BGR, 64, 64))) / 255`

The resulting tensor has shape `(64, 64, 3)`. The same resize, color conversion,
and normalization are used for training, validation, and test evaluation.

### 3.2 Training-time augmentation

Training applies a random horizontal reflection and random rotation. For
TensorFlow/Keras 2.15, `RandomRotation(factor=0.05)` samples from a fraction of
a full turn, approximately `[-0.05 × 2π, +0.05 × 2π]` or `[-18°, +18°]`. These
layers are inactive at inference. They have no learned parameters.

### 3.3 Convolution and separable convolution

For an ordinary convolution, each output channel combines local input-channel
patches with learned kernels and a bias:

`z[h,w,o] = b[o] + Σ(i,u,v) K[u,v,i,o] × x[h+u,w+v,i]`

`a[h,w,o] = ReLU(z[h,w,o])`

With the first layer's input channels 3, output filters 32, kernel 3 × 3, and
bias, the parameter count is:

`(3 × 3 × 3 × 32) + 32 = 896`

Each depthwise-separable convolution applies a spatial kernel independently to
each input channel (depthwise stage), then combines the resulting channels by
1 × 1 pointwise convolution:

`d[h,w,i] = Σ(u,v) D[u,v,i] × x[h+u,w+v,i]`

`z[h,w,o] = b[o] + Σ(i) P[i,o] × d[h,w,i]`

`a[h,w,o] = ReLU(z[h,w,o])`

With depth multiplier 1, the parameter count is
`(3 × 3 × input_channels) + (input_channels × output_channels) + output_channels`.
Therefore:

- 32 input channels to 64 output channels:
  `(3 × 3 × 32) + (32 × 64) + 64 = 2,400`.
- 64 input channels to 128 output channels:
  `(3 × 3 × 64) + (64 × 128) + 128 = 8,896`.

For `padding="same"`, output spatial size is rounded up after division by
stride. The initial 3 × 3 convolution has stride 2, so 64 × 64 becomes 32 × 32.
The 2 × 2 max-pooling layer with stride 2 then changes 32 × 32 to 16 × 16.

### 3.4 Batch Normalization

For each feature channel, training normalizes using the mini-batch mean and
variance, then applies learned scale `gamma` and offset `beta`:

`BN(x) = gamma × (x - mean_batch) / sqrt(variance_batch + epsilon) + beta`

During inference, `mean_batch` and `variance_batch` are replaced by the saved
moving mean and moving variance. There are two trainable values (`gamma`,
`beta`) and two non-trainable moving statistics per channel.

### 3.5 Pooling, dropout, and classifier

Max pooling takes the maximum value in each 2 × 2 window:

`pool[h,w,c] = max(x[2h:2h+2, 2w:2w+2, c])`

Global average pooling turns each 16 × 16 feature map into one scalar:

`g[c] = (1 / (16 × 16)) × Σ(h=1..16,w=1..16) x[h,w,c]`

Dropout with rate 0.30 randomly masks feature components during training and
scales surviving values according to Keras dropout semantics. It is an
inference-time identity operation.

The final dense logit and probability are:

`z = b + Σ(c=1..128) w[c] × g[c]`

`p(Flushing | X) = sigmoid(z) = 1 / (1 + exp(-z))`

The dense layer has `128 × 1` kernel weights and 1 bias, for `129` trainable
parameters. Model prediction is `Flushing` when `p >= threshold`, and `Normal`
otherwise.

## 4. Loss function and what alpha changes

The output uses a sigmoid probability `p`; the binary label is `y ∈ {0,1}`.
Focal loss first defines the probability assigned to the correct class:

`p_t = p` when `y = 1`; otherwise `p_t = 1 - p`.

The class coefficient is:

`alpha_t = alpha` when `y = 1`; otherwise `alpha_t = 1 - alpha`.

For each example:

`FL(y,p) = -alpha_t × (1 - p_t)^gamma × log(p_t)`

Training uses `gamma = 2`. The batch loss is the mean of the per-example focal
losses. The modulation `(1 - p_t)^gamma` reduces the contribution of examples
already classified confidently and correctly, while retaining emphasis on
harder examples.

| Run | alpha | Flushing (`y=1`) coefficient | Normal (`y=0`) coefficient |
| --- | ---: | ---: | ---: |
| Model-02 / Model-03 alpha 0.25 | 0.25 | 0.25 | 0.75 |
| Model-03 alpha 0.75 | 0.75 | 0.75 | 0.25 |

The Model-03 alpha 0.75 experiment increases loss emphasis on positive
Flushing examples. It does not directly force recall to a particular value;
the network must learn from the images, and the decision threshold still
controls the final precision/recall trade-off.

## 5. Optimizer and parameter updates

Both runs use Keras legacy Adam with learning rate `η = 0.0001` and its defaults
`beta1 = 0.9`, `beta2 = 0.999`, and `epsilon = 1e-7`.

For parameter vector `theta` and gradient `g_t = ∇theta L_t`, Adam updates
first and second gradient moments:

`m_t = beta1 × m_(t-1) + (1 - beta1) × g_t`

`v_t = beta2 × v_(t-1) + (1 - beta2) × g_t^2`

`m_hat_t = m_t / (1 - beta1^t)`

`v_hat_t = v_t / (1 - beta2^t)`

`theta_t = theta_(t-1) - η × m_hat_t / (sqrt(v_hat_t) + epsilon)`

This is applied to trainable convolution kernels, biases, Batch Normalization
scale/offset, and the output dense weights/bias. Batch Normalization moving
statistics are updated from batches, not by Adam.

## 6. Training, checkpoint selection, and evaluation

1. Build the numeric-filename-prefix-grouped 70/15/15 split across both
   classes using seed 42. This does not verify person identity.
   The exact counts are in [`split_counts.csv`](./split_counts.csv), and the
   visualization is [`split_distribution.png`](./figures/split_distribution.png).
2. Create a fresh model for each alpha run, set the random seed, and train with
   batch size 32 for at most 35 epochs.
3. Monitor validation AUC. Early stopping uses patience 8 and restores the
   weights from the best monitored epoch. `ModelCheckpoint` saves the best
   validation-AUC checkpoint. There is no learning-rate scheduler.
4. Predict probabilities on validation and held-out test patches. Exact
   scikit-learn ROC-AUC and average precision are computed from scores, before
   applying any classification threshold.
5. The default confusion-matrix metrics use `p >= 0.50`. Additional thresholds
   are selected using validation precision floors (0.95, 0.90, 0.80, 0.70);
   each selected threshold is applied unchanged to test.
6. Accuracy, precision, recall, specificity, and F1 are calculated from the
   resulting confusion matrix:

   - `accuracy = (TP + TN) / (TP + TN + FP + FN)`
   - `precision = TP / (TP + FP)`
   - `recall = sensitivity = TP / (TP + FN)`
   - `specificity = TN / (TN + FP)`
   - `F1 = 2 × precision × recall / (precision + recall)`

   Precision/recall/F1 use zero when their denominator is zero.
7. Store model checkpoints, per-epoch histories, scores, JSON metrics, and
   plots under `results/`. The original Model-02 checkpoint and source images
   are read-only inputs.

The run history and evaluation artifacts are in
[`focal_alpha_025/`](./focal_alpha_025/) and
[`focal_alpha_075/`](./focal_alpha_075/). The detailed experiment narrative
and figures are in [`FINAL_SUMMARY.md`](./FINAL_SUMMARY.md).

## 7. Practical interpretation and limitations

In this one controlled split, alpha 0.75 improved test recall at threshold 0.50
from 0.6614 to 0.8570 and test ROC-AUC from 0.9842 to 0.9926 compared with the
retrained alpha-0.25 run. That is evidence that the class-weight change helped
in this experiment, not proof that alpha 0.75 is universally optimal.

The original Model-02 validation metrics are useful to understand the initial
problem, but numeric filename-prefix groups crossed boundaries in its old
split. The prefixes are not verified person identities. The rebuilt split
keeps prefixes together, but different prefixes may still belong to one person;
person-level separation and absence of identity leakage remain unverified.
These results must not be presented as a person-independent evaluation. The
controlled alpha 0.25 versus 0.75 comparison uses the same numeric-prefix-
grouped split.

One seed and one split are insufficient to establish robustness. Further work
should evaluate multiple seeds or an external cohort, agree on the triage
precision/recall trade-off, and separately verify that deployed preprocessing
matches training preprocessing. These results do not establish clinical
validity.
