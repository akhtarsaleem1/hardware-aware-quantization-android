#!/usr/bin/env python3
"""Fail-closed audit for the final scientific and publishing package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


SCIENTIFIC_PLACEHOLDERS = (
    "[FINAL",
    "[AUTO-FILL",
    "PENDING_COMPLETED",
    "PENDING_DATASET",
    "PENDING_VERIFIED",
    "SCAFFOLD_ONLY",
    "BLOCKED_UNTIL",
    "NOT_RUN",
)
HUMAN_METADATA_PLACEHOLDERS = (
    "[AUTHOR NAME]",
    "[AFFILIATION]",
    "[CORRESPONDING AUTHOR EMAIL]",
    "[AUTHOR TO COMPLETE",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_json(path: Path, status: str, problems: list[str]) -> dict:
    if not path.is_file():
        problems.append(f"missing required JSON: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        problems.append(f"unreadable JSON {path}: {error}")
        return {}
    if value.get("status") != status:
        problems.append(f"{path} status is {value.get('status')!r}, expected {status!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/publication_readiness.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    problems: list[str] = []
    human_metadata: list[str] = []

    publishing_files = [
        root / "paper/manuscript_main.md",
        root / "paper/highlights.md",
        root / "paper/cover_letter_draft.md",
        root / "paper/journal_target_version.md",
        root / "paper/data_availability.md",
        root / "paper/code_availability.md",
        root / "paper/reproducibility_statement.md",
        root / "research_integrity/report.md",
        root / "literature/novelty_audit.md",
    ]
    for path in publishing_files:
        if not path.is_file():
            problems.append(f"missing publishing file: {path}")
            continue
        content = path.read_text(encoding="utf-8")
        for marker in SCIENTIFIC_PLACEHOLDERS:
            if marker in content:
                problems.append(f"scientific placeholder {marker!r} remains in {path}")
        for marker in HUMAN_METADATA_PLACEHOLDERS:
            if marker in content:
                human_metadata.append(f"human metadata {marker!r} remains in {path}")

    bibliography = root / "paper/references.bib"
    if not bibliography.is_file():
        problems.append("missing paper/references.bib")
        reference_count = 0
    else:
        reference_count = sum(
            line.startswith("@")
            for line in bibliography.read_text(encoding="utf-8").splitlines()
        )
        if reference_count < 25:
            problems.append(f"bibliography contains only {reference_count} entries")
        bibliography_text = bibliography.read_text(encoding="utf-8")
        reference_keys = re.findall(r"^@\w+\{([^,]+),", bibliography_text, re.M)
        if len(reference_keys) != len(set(reference_keys)):
            problems.append("bibliography contains duplicate entry keys")
        manuscript_path = root / "paper/manuscript_main.md"
        if manuscript_path.is_file():
            cited = set(re.findall(
                r"(?<![A-Za-z0-9._%+-])@([A-Za-z0-9_:-]+)",
                manuscript_path.read_text(encoding="utf-8"),
            ))
            missing = sorted(cited - set(reference_keys))
            if missing:
                problems.append(f"manuscript citation keys missing from bibliography: {missing}")

    novelty_path = root / "literature/novelty_audit.md"
    if novelty_path.is_file() and (
        "FINAL_RECHECK_COMPLETE_REPLICATION_EXTENSION_FRAMING"
        not in novelty_path.read_text(encoding="utf-8")
    ):
        problems.append("final 2025-2026 novelty recheck is not complete")

    benchmark_audit = require_json(
        root / "reports/final_benchmark_audit.json", "PASS", problems
    )
    pretest = require_json(
        root / "results/pretest/selection_summary.json",
        "PRETEST_SELECTION_COMPLETE",
        problems,
    )
    selection = require_json(
        root / "reports/final_test_manifest.json", "FROZEN_FOR_FINAL_TEST", problems
    )
    test_report = require_json(
        root / "results/final/final_test_report.json",
        "FINAL_TEST_SINGLE_SESSION_COMPLETE",
        problems,
    )
    analysis = require_json(
        root / "results/final/analysis_summary.json", "FINAL_ANALYSIS_COMPLETE", problems
    )
    supplement = require_json(
        root / "results/final/publication_supplement/supplement_manifest.json",
        "PUBLICATION_SUPPLEMENT_COMPLETE",
        problems,
    )
    bundle = require_json(root / "reports/flutter_model_bundle.json", "PASS", problems)

    for relative in (
        "results/final/final_test_predictions.csv",
        "results/final/table_model_results.csv",
        "results/final/table_trial_summaries.csv",
        "results/final/table_configuration_results.csv",
        "results/final/table_configuration_errors.csv",
        "results/final/table_rank_stability.csv",
        "results/final/figures/figure_1_accuracy_size.png",
        "results/final/figures/figure_2_latency.png",
        "results/final/figures/figure_3_latency.png",
        "results/final/figures/figure_4_pareto.png",
        "results/final/figures/figure_5_rank_stability.png",
        "results/final/publication_supplement/tables/table_1_dataset_training_configuration.csv",
        "results/final/publication_supplement/tables/table_2_model_architecture_comparison.csv",
        "results/final/publication_supplement/tables/table_3_accuracy_metrics.csv",
        "results/final/publication_supplement/tables/table_4_quantization_model_sizes.csv",
        "results/final/publication_supplement/tables/table_5_android_latency_statistics.csv",
        "results/final/publication_supplement/tables/table_6_memory_measurements.csv",
        "results/final/publication_supplement/tables/table_7_runtime_delegate_comparison.csv",
        "results/final/publication_supplement/tables/table_8_pareto_efficient_configurations.csv",
        "results/final/publication_supplement/tables/table_9_hardware_aware_recommendations.csv",
        "research_integrity/result_provenance.csv",
        "paper/manuscript_main.docx",
        "paper/manuscript_main.pdf",
    ):
        if not (root / relative).is_file():
            problems.append(f"missing final artifact: {relative}")

    apk = root / "flutter_app/build/app/outputs/flutter-apk/app-release.apk"
    if not apk.is_file():
        problems.append("missing final release APK")
        apk_hash = None
    else:
        apk_hash = sha256(apk)
    apk_report_path = root / "reports/final_apk.json"
    apk_report = require_json(apk_report_path, "PASS", problems)
    if apk_hash is not None and apk_report.get("sha256") != apk_hash:
        problems.append("release APK hash does not match reports/final_apk.json")

    if pretest:
        if pretest.get("test_split_accessed") is not False:
            problems.append(
                "pretest report does not attest that the test split stayed locked"
            )
        raw_path_text = pretest.get("raw_benchmark")
        raw_hash = pretest.get("raw_benchmark_sha256")
        if not raw_path_text or not raw_hash:
            problems.append("pretest report is missing raw benchmark provenance")
        else:
            # Provenance may be generated on Windows and verified under WSL/Linux.
            # Treat either platform separator as a relative-path separator.
            raw_path = Path(str(raw_path_text).replace("\\", "/"))
            if not raw_path.is_absolute():
                raw_path = root / raw_path
            if not raw_path.is_file():
                problems.append(f"pretest raw benchmark is missing: {raw_path}")
            elif sha256(raw_path) != raw_hash:
                problems.append("pretest raw benchmark hash mismatch")
    if benchmark_audit and pretest:
        audit_path = root / "reports/final_benchmark_audit.json"
        if pretest.get("benchmark_audit_sha256") != sha256(audit_path):
            problems.append("pretest report does not hash-link the benchmark audit")
        if benchmark_audit.get("raw_csv_sha256") != pretest.get("raw_benchmark_sha256"):
            problems.append("benchmark audit raw hash differs from pretest report")
    if pretest and selection:
        pretest_path = root / "results/pretest/selection_summary.json"
        if selection.get("pretest_selection_report_sha256") != sha256(pretest_path):
            problems.append(
                "frozen manifest does not hash-link the pretest selection report"
            )
        if selection.get("raw_benchmark_sha256") != pretest.get(
            "raw_benchmark_sha256"
        ):
            problems.append("frozen manifest raw benchmark hash differs from pretest report")
    if selection and test_report:
        if test_report.get("selection_frozen_before_test") is not True:
            problems.append("final test report does not attest frozen selection")
        prediction_path = root / "results/final/final_test_predictions.csv"
        if prediction_path.is_file() and test_report.get(
            "prediction_file_sha256"
        ) != sha256(prediction_path):
            problems.append("prediction CSV hash mismatch")
    if analysis:
        if analysis.get("pareto_selection_uses_validation_not_test") is not True:
            problems.append("analysis does not attest validation-only Pareto selection")
        audit_path = root / "reports/final_benchmark_audit.json"
        if audit_path.is_file() and analysis.get("benchmark_audit_sha256") != sha256(
            audit_path
        ):
            problems.append("final analysis does not hash-link the benchmark audit")
    if bundle and bundle.get("model_count") != 12:
        problems.append("Flutter bundle does not contain 12 model configurations")
    provenance_path = root / "research_integrity/result_provenance.csv"
    if provenance_path.is_file():
        with provenance_path.open(newline="", encoding="utf-8") as handle:
            provenance_rows = list(csv.DictReader(handle))
        identifiers = [row.get("result_id", "") for row in provenance_rows]
        if len(provenance_rows) < 80:
            problems.append("result provenance ledger has fewer than 80 rows")
        if len(identifiers) != len(set(identifiers)) or any(not value for value in identifiers):
            problems.append("result provenance identifiers are empty or duplicated")

    if supplement:
        if supplement.get("table_count") != 9:
            problems.append("publication supplement does not contain nine tables")
        if int(supplement.get("figure_png_count", 0)) < 7 or int(
            supplement.get("figure_pdf_count", 0)
        ) < 7:
            problems.append("publication supplement has fewer than seven PNG/PDF figures")
        for artifact in supplement.get("artifacts", []):
            path = root / str(artifact.get("path", ""))
            if not path.is_file() or sha256(path) != artifact.get("sha256"):
                problems.append(f"publication supplement artifact hash mismatch: {path}")

    status = (
        "SCIENTIFIC_PACKAGE_READY_AUTHOR_METADATA_REQUIRED"
        if not problems and human_metadata
        else "READY_FOR_AUTHOR_SUBMISSION_REVIEW"
        if not problems
        else "NOT_READY"
    )
    report = {
        "status": status,
        "root": str(root),
        "scientific_blockers": problems,
        "human_metadata_actions": sorted(set(human_metadata)),
        "reference_count": reference_count,
        "apk_sha256": apk_hash,
        "clarivate_scie_directly_verified": False,
        "clarivate_note": "Human must verify both ISSNs in the live Master Journal List before any SCIE claim.",
    }
    output = root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
