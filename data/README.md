# Dataset status: DeepWeeds acquired and pretraining split frozen

DeepWeeds is the selected public image-classification dataset. It contains
17,509 in-situ RGB images across eight weed species and a negative class. The
official repository supplies five 60/20/20 train/validation/test folds and
states that the images and annotations are CC BY 4.0. The repository code is
Apache 2.0.

Authoritative source:
https://github.com/AlexOlsen/DeepWeeds

No image from the previous plant-disease project is part of this study. The
official archive was downloaded and verified on 2026-08-21:

- size: 491,516,047 bytes;
- SHA-256: `0961f63c01b997bfab1559ad09e99c0e8130617fd96a8b92fdc09940e01b0ce8`;
- extracted JPEGs: 17,509, all readable 256 x 256 RGB.

## Frozen preparation protocol

1. Download only from the source linked by the official repository.
2. Save the archive SHA-256 and acquisition date before extraction.
3. Audit official fold 0 as the initial reproducible split.
4. Build canonical manifests with `scripts/prepare_dataset.py`.
5. Run `scripts/validate_dataset.py` before training. It checks schema, image
   readability, class counts, exact cross-split duplicates, perceptual-hash
   candidates, and capture-session overlap.
6. Because the official random fold placed 3,418 capture-minute/instrument
   groups across partitions, use the frozen 90-second capture-session grouped
   split for confirmatory training. Retain fold 0 for sensitivity analysis.
7. The grouped split has 10,506 training, 3,502 validation, and 3,501 test
   images. It has no exact cross-split duplicate or capture-session overlap.
   Its one dHash candidate was visually and numerically reviewed as distinct.
8. Use training images only for augmentation, class weighting, calibration,
   and any model-selection decisions.

Full INT8 representative samples must come only from the frozen training
allocation. Details and selection rationale are in
[`dataset_selection.md`](dataset_selection.md).

Source note: `labels.csv` has `Filename,Label,Species`, but the official fold
files contain only `Filename,Label`. The preparation code derives the species
display name from the published integer label map and never guesses labels from
filenames or image contents.

Generated evidence is in [`../reports/dataset_report.md`](../reports/dataset_report.md),
with the original-fold comparison in
[`../reports/dataset_report_official_fold0.md`](../reports/dataset_report_official_fold0.md).
