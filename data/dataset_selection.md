# Dataset selection

Updated: 2026-08-21

## Selected dataset: DeepWeeds

The main dataset is **DeepWeeds**. Download integrity and split-leakage audits
are complete; the grouped pretraining split is frozen.

Reasons:

- 17,509 real in-situ RGB images across eight weed species and a negative
  class;
- native 256 x 256 images require only a modest crop/resize to 224 x 224;
- source images and annotations are explicitly licensed CC BY 4.0;
- the code repository is Apache 2.0;
- official 60/20/20 train/validation/test CSVs for five folds are available;
- 468 MB image archive is computationally practical;
- the dataset is large enough for stable representative calibration and
  per-class metrics on a 4 GiB training GPU;
- field backgrounds and class imbalance provide a less sanitized workload than
  PlantVillage while remaining a public non-human dataset.

This does not continue the previous plant-disease experiment. DeepWeeds uses a
different source, task, label space, capture conditions, split files, and model
training runs. Prior models or predictions will not be transferred.

## Alternatives considered

### EuroSAT RGB

EuroSAT provides 27,000 images, ten classes, and an MIT-licensed distribution.
It is an excellent fallback, but 64 x 64 satellite patches require substantial
upscaling and random splits can leak spatially autocorrelated locations. A
spatial split exists in TorchGeo, but remote-sensing preprocessing would add a
second methodological theme unrelated to Android quantization.

### CIFAR-100

CIFAR-100 is computationally convenient and widely benchmarked, but its 32 x 32
images are a poor match for 224 x 224 mobile backbones and its original hosting
does not state a clear dataset license. It is not selected.

### Beans

Beans contains smartphone field images but only 1,295 examples. The TensorFlow
Datasets catalog reports no known dataset license. It is too small and legally
less clear for the main study.

## Split policy

Official fold 0 was used as the initial fixed 60/20/20 protocol and retained as
a sensitivity baseline. Before any model result was observed:

1. verify every referenced file and label;
2. compute exact SHA-256 hashes and perceptual hashes;
3. test for exact and near duplicates across partitions;
4. inspect class and acquisition-date/instrument distributions;
5. the audit found 3,418 capture-minute/instrument groups crossing official
   fold-0 partitions and 39 cross-split dHash candidates;
6. a deterministic grouped split (seed 42) was therefore created, defining a
   session as consecutive captures from one instrument separated by no more
   than 90 seconds;
7. this yields 1,105 groups, maximum group size 224, and 10,506/3,502/3,501
   train/validation/test images with near-exact per-class 60/20/20 balance;
8. the grouped audit found zero filename, exact-hash, or session overlap. One
   remaining dHash candidate was reviewed as two distinct scenes (pixel
   correlation 0.369); and
9. the grouped manifests were marked `FROZEN_PRETRAINING_SPLIT`. The official
   fold remains available for sensitivity analysis rather than being hidden.

The negative class is approximately half of the corpus, so macro F1, per-class
recall, balanced accuracy, and confusion matrices are required in addition to
overall accuracy. Class weighting may use training counts only.

## Licensing and citation

Dataset and annotations: CC BY 4.0 according to the official DeepWeeds
repository. Repository code: Apache 2.0. The dataset paper and repository must
be cited, and redistributed artifacts must retain attribution and license
information.
