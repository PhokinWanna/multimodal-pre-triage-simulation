# CNN image dataset: provenance, access, and preparation

This repository does not include the source image archive. The face images
remain in a locally managed dataset and are not intended to be committed or
redistributed from this repository. The source material is a mixture of five
third-party datasets; it is not a single, uniformly licensed or independently
verified dataset.

## Source datasets reported by the dataset owner

The following inventory reflects the owner's account of the downloads and
subsets. The local source files and source-to-image manifest were unavailable
for independent verification when this document was written (2026-10-24).
Public page notices below were checked on that date and may change.
Subset descriptions below must not be read as proof that every downloaded
image was used in the training experiment.

| Source | Owner-reported subset | License/provenance status |
| --- | --- | --- |
| [Flickr-Faces-HQ (FFHQ) on Kaggle](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) | 8,135 images selected from a 52,000-image listing | The Kaggle page declares CC BY-SA 3.0. The [upstream FFHQ project](https://github.com/NVlabs/ffhq-dataset) describes the dataset package as CC BY-NC-SA 4.0 and says individual Flickr images have varying licenses recorded in source metadata. The Kaggle and upstream notices differ; the applicable rights for the downloaded subset have not been resolved. |
| [Massive Skin Disease Balanced Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadabdulsami/massive-skin-disease-balanced-dataset) | The “Acne And Rosacea” folder | The Kaggle page declares MIT, but describes a compilation of publicly available sources. The underlying image sources and their separate rights have not been established. |
| [Skin Problem Detection Relabel Clean 3 on Roboflow Universe](https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-detection-relabel-clean3) | Owner reports using the full project download | The project's license and upstream image provenance were not verified. |
| [Skin Problem Detection Relabel Clean 2 on Roboflow Universe](https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-detection-relabel-clean-2) | Owner reports using the full project download | The project's license and upstream image provenance were not verified. |
| [FairFace on Kaggle](https://www.kaggle.com/datasets/alexstvn/fairface) | Owner reports 19,186 images from the train folder, from a stated 86,700 | The Kaggle page lists the license as unknown. The [upstream FairFace project](https://github.com/dchen236/FairFace) states CC BY 4.0 and requests citation. The rights and provenance for this Kaggle repack and selected subset still need confirmation. |

These page-level labels are leads, not a legal determination that the uploader
had rights to relicense every underlying photograph or that this project's
particular use, publication, or redistribution is permitted. Do not assume that
a dataset marked “MIT” or available for download clears the rights to every
image in it. In particular, FFHQ's Kaggle and upstream license notices differ,
and the Roboflow project licenses remain unverified.

### Use and redistribution status

The owner has not yet checked all source terms. Until the terms and source
attribution are verified, treat permission to use, publish, or redistribute
the combined images and derived patches as **unresolved**. Do not commit the
images, patches, private download links, or credentials to a public
repository. Confirm the relevant dataset and image licenses with each
publisher; obtain qualified legal or institutional guidance if use in an
internship portfolio, publication, or commercial setting is planned.

## Privacy and handling

The owner reports that the downloaded files contain pictures without separate
personal-information fields. They contain recognizable human faces,
nevertheless; absence of names or other metadata does not make face images
anonymous. The original subjects' consent, privacy permissions, and any
source-specific restrictions have not been verified. Treat the images and
face crops as sensitive, restrict access to authorized project work, and do
not publish examples or share the archive without confirming permission.

## Owner-reported counts and unresolved lineage

Counts below are reported by the owner and are not validated against a
source manifest or the absent local archive:

| Stage | Normal | Flushing | Total |
| --- | ---: | ---: | ---: |
| Combined raw archive pool | 27,331 | 12,440 | 39,771 |
| Source images selected for the described CNN experiment | 8,135 | 8,638 | 16,773 |
| Extracted 64x64 patches used in the recorded Model-03 experiment | 19,593 | 5,345 | 24,938 |

The Model-03 experiment reports 24,938 patches across train, validation, and
test. Its exact per-class totals are recorded in
[`../../scripts/train/cnn/model03/results/split_counts.csv`](../../scripts/train/cnn/model03/results/split_counts.csv).
The source-image counts and patch counts are different units because one
source image may yield multiple face-zone patches. The source-by-source
selection, class assignment, duplicate handling, and exact relationship
between the raw archive and the experiment inputs still need a manifest; do
not infer that relationship from the aggregate counts alone.

## Current preparation steps

The pipeline expects authorized source images organized in
`data/dataset-cnn/Normal/` and `data/dataset-cnn/Flushing/`. These local source
images, extracted patches, and split copies are ignored by Git. From the
repository root, install the CNN dependency group in a Python 3.11 environment:

```bash
python3.11 -m venv venv_train
source venv_train/bin/activate
python -m pip install -r requirements-cnn.txt
python scripts/data/dataset_extraction-CNN.py
python scripts/train/cnn/split_dataset.py
python scripts/train/cnn/train_skin_cnn-02.py
```

The extractor writes 64x64 patches under
`data/dataset-cnn/patches/{Normal,Flushing}/`; it uses
MediaPipe Face Mesh (one detected face per source image), crops the forehead
and eligible cheek zones, filters patches using a minimum 70% YCrCb skin-color
mask, and skips crops outside image bounds. Each side cheek is skipped when
its normalized share of nose-to-cheek-edge distance is below 0.25. It does
not perform illumination correction,
denoising, or image enhancement; some dark, bright, or grayscale images may
remain. The splitter copies patches into
`data/dataset-cnn/splits/{train,val,test}/{Normal,Flushing}/`.
The current split is 70/15/15 with seed 42, stratified by class presence of
numeric-prefix groups. The trainer later resizes to 64x64, converts BGR to
CIELAB, and scales values to float32 / 255.
See the [pipeline instructions](../../scripts/train/cnn/SPLIT_README.md)
and [Model-03 experiment guide](../../scripts/train/cnn/model03/README.md)
for details.

The split groups files by the leading numeric filename prefix. This is a
filename-based grouping heuristic, **not verified person identity**. The
owner reports that different prefixes can still refer to the same person.
Consequently, the existing split and reported metrics cannot establish
person-disjoint evaluation or rule out identity leakage. Do not describe this
split as patient-disjoint until identity linkage has been verified and the
split is rebuilt accordingly.

This workflow is not yet fully reproducible from the sources alone. Before
making a reproducibility claim, retain a private source manifest with
publisher/version, download date, license notice, selected file identifiers,
class mapping, selection/exclusion reasons, duplicate checks, and checksums.
Do not place face images, direct identifiers, or a private access link in that
manifest if it will be published.

## Access

The image archive is not stored in this repository. The owner reports that
they combined the local downloads into Normal and Flushing pools and that
access should be requested directly from them. No public Google Drive URL is
provided here. Access should only be granted after the requester confirms
they are authorized to use the relevant source data; a copy of this archive
does not transfer rights to the underlying images.

## References

- Kaggle and Roboflow project pages are linked in the source inventory above.
- [Upstream FFHQ repository and license description](https://github.com/NVlabs/ffhq-dataset).
- [Upstream FairFace repository and license/citation notice](https://github.com/dchen236/FairFace).
