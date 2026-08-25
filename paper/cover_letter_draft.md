# Cover letter

25 August 2026

Editor-in-Chief  
Journal of Systems Architecture

Dear Editor,

Please consider my original research manuscript, **“Beyond Average Latency: Repeatability and Runtime-Dependent Quantization Rankings for Lightweight Vision Models on Android,”** for publication in the *Journal of Systems Architecture*.

The manuscript reports an auditable systems study of 12 deployable LiteRT flatbuffers across 72 planned model, runtime, and thread configurations on a lower-cost Android 15 phone. Because trial 3 was incomplete, it was excluded in full and the bounded analysis uses only the two complete balanced trials. The Android evidence is therefore explicitly exploratory. Rather than presenting another application-level comparison, the paper examines capture-session leakage control, a test split held closed until device selection was frozen, immutable per-inference records, randomized complete trials, p95/p99 latency, rank stability, memory and thermal context, and preserved delegate/configuration failures.

The locked 3,501-image evaluation produced 53.93%-84.43% accuracy. The best mean trial median was 146.77 ms, and the best mean trial p95 was 162.34 ms; they came from different thread settings. File-size and median-latency order differed in all 18 contexts, runtime changed quantization rank in 7 of 9 comparisons, and median versus p95 criteria changed the Pareto frontier. The benchmark retained 3 failed configurations as evidence rather than replacing them with fallback measurements.

The paper fits the journal's scope because it treats quantization as a systems problem. Deployable model artifacts are evaluated together with runtime kernels, delegate behavior, threading, and measurement design on real hardware.

This manuscript is original, is not under consideration elsewhere, and has not been published previously. The sole author has approved the submission. Funding, conflict-of-interest, authorship, and AI-assistance declarations are included in the manuscript. The accompanying artifact contains code, manifests, model and raw-data hashes, frozen selection evidence, per-sample predictions, tables, figures, the release APK, and the record of an excluded incomplete ANR run. The v1.0.0 reproducibility artifact is publicly available at https://doi.org/10.5281/zenodo.22082237. I reviewed the final content and take responsibility for the work.

Sincerely,

Akhtar Saleem  
Independent Researcher, Peshawar, Pakistan  
akhtarsaleem974@gmail.com  
ORCID: https://orcid.org/0009-0005-9440-224X
