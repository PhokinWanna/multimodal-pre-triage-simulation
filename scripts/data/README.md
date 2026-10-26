# Data preparation scripts

`dataset_extraction-CNN.py` reads local `data/dataset-cnn/{Normal,Flushing}/`
images and writes 64×64 face-zone patches to `data/dataset-cnn/patches/`. The default
paths are rooted at the repository and can be overridden with `--source-dir`
and `--output-dir`.

Source images and generated patches are sensitive and excluded by Git. Confirm
dataset permissions before use; see [`../../docs/data/README.md`](../../docs/data/README.md).
