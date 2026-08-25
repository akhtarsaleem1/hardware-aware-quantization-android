# Cover letter

25 August 2026

Editor-in-Chief  
Journal of Systems Architecture

Dear Editor,

Please consider my original research manuscript, **“Beyond Average Latency: Repeatability and Runtime-Dependent Quantization Rankings for Lightweight Vision Models on Android,”** for publication in the *Journal of Systems Architecture*.

The manuscript presents an auditable, system-level study of 12 deployable LiteRT flatbuffers across 72 planned model/runtime/thread configurations on a lower-cost Android 15 phone. A post hoc bounded analysis retains two complete balanced trials after the incomplete third trial was excluded wholesale, so the Android evidence is explicitly exploratory rather than confirmatory. The contribution is deliberately narrower than another application comparison: it focuses on capture-session leakage control, a test split held closed until device selection was frozen, immutable per-inference observations, complete randomized trials, p95/p99 latency, rank stability, memory and thermal context, and explicit preservation of delegate/configuration failures.

The locked 3,501-image evaluation produced 53.93%-84.43% accuracy. The fastest mean trial median and p95 were 146.77 ms and 162.34 ms. File-size and median-latency order differed in 18/18 contexts, runtime changed quantization rank in 7/9 comparisons, and median versus p95 criteria changed the Pareto frontier. 3 failed configurations were retained as evidence rather than replaced with fallback measurements.

The work fits the journal's published scope in embedded/mobile systems and software/system architecture by examining how deployable model artifacts interact with runtime kernels, delegates, threading, and measurement design. I make no unverified claim about Clarivate indexing; that administrative check remains separate from the scientific submission package.

This manuscript is original, is not under consideration elsewhere, and has not been published previously. The sole author has approved the submission. Funding and conflict-of-interest declarations are included in the manuscript. The artifact package contains code, manifests, model and raw-data hashes, frozen selection evidence, per-sample predictions, tables, figures, the release APK, and an explicit record of an excluded incomplete ANR run. The v1.0.0 reproducibility artifact is publicly available at https://doi.org/10.5281/zenodo.22082237.

Sincerely,

Akhtar Saleem  
Independent Researcher, Peshawar, Pakistan  
akhtarsaleem974@gmail.com  
ORCID: https://orcid.org/0009-0005-9440-224X
