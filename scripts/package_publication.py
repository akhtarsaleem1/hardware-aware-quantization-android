#!/usr/bin/env python3
"""Create hash-manifested journal and reproducibility archives."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files(root: Path, entries: list[str]) -> list[Path]:
    selected: set[Path] = set()
    for entry in entries:
        path = root / entry
        if path.is_file():
            selected.add(path)
        elif path.is_dir():
            selected.update(value for value in path.rglob("*") if value.is_file())
        else:
            raise FileNotFoundError(path)
    return sorted(selected, key=lambda value: value.relative_to(root).as_posix())


def allowed(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    if relative == "flutter_app/build/app/outputs/flutter-apk/app-release.apk":
        return True
    blocked_parts = (
        "/__pycache__/", "/.dart_tool/", "/build/", "/.gradle/",
        "/data/source/", "/models/checkpoints/", "/models/saved/",
    )
    blocked_suffixes = (".pyc", ".log.pid")
    return not any(value in "/" + relative for value in blocked_parts) and not relative.endswith(
        blocked_suffixes
    )


def archive(root: Path, output: Path, selected: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in selected:
            relative = path.relative_to(root).as_posix()
            handle.write(path, relative)
            records.append({
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    with zipfile.ZipFile(output, "r") as handle:
        if handle.testzip() is not None:
            raise ValueError(f"ZIP CRC failure: {output}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("publication"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness_path = root / "reports/publication_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if readiness.get("status") not in (
        "SCIENTIFIC_PACKAGE_READY_AUTHOR_METADATA_REQUIRED",
        "READY_FOR_AUTHOR_SUBMISSION_REVIEW",
    ):
        raise ValueError("publication readiness gate has not passed")

    journal_entries = [
        "paper/manuscript_main.md",
        "paper/manuscript_main.docx",
        "paper/manuscript_main.pdf",
        "paper/references.bib",
        "paper/cover_letter_draft.md",
        "paper/highlights.md",
        "paper/data_availability.md",
        "paper/code_availability.md",
        "paper/reproducibility_statement.md",
        "paper/journal_target_version.md",
        "results/final/figures",
        "results/final/publication_supplement",
        "research_integrity/report.md",
        "research_integrity/result_provenance.csv",
        "reports/publication_readiness.json",
    ]
    artifact_entries = [
        "README.md", "REPRODUCIBILITY.md", "LICENSE", "requirements.txt",
        "environment.yml", "config.yaml", "EXPERIMENT_PLAN.md",
        "environment", "literature", "data/manifests_grouped",
        "models/metadata", "benchmark", "reports", "results",
        "research_integrity", "paper", "scripts", "src", "tests",
        "flutter_app/lib", "flutter_app/test", "flutter_app/android",
        "flutter_app/assets", "flutter_app/pubspec.yaml",
        "flutter_app/pubspec.lock",
        "flutter_app/build/app/outputs/flutter-apk/app-release.apk",
        "prior_research",
    ]
    journal_files = [path for path in files(root, journal_entries) if allowed(path, root)]
    artifact_files = [path for path in files(root, artifact_entries) if allowed(path, root)]
    journal_zip = output_dir / "journal_submission_ready_for_review.zip"
    artifact_zip = output_dir / "reproducibility_artifact.zip"
    journal_records = archive(root, journal_zip, journal_files)
    artifact_records = archive(root, artifact_zip, artifact_files)
    manifest = {
        "status": "PUBLICATION_ARCHIVES_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "readiness_status": readiness["status"],
        "source_images_included": False,
        "redundant_training_checkpoints_included": False,
        "journal_archive": {
            "path": journal_zip.relative_to(root).as_posix(),
            "sha256": sha256(journal_zip),
            "bytes": journal_zip.stat().st_size,
            "file_count": len(journal_records),
            "files": journal_records,
        },
        "reproducibility_archive": {
            "path": artifact_zip.relative_to(root).as_posix(),
            "sha256": sha256(artifact_zip),
            "bytes": artifact_zip.stat().st_size,
            "file_count": len(artifact_records),
            "files": artifact_records,
        },
    }
    manifest_path = output_dir / "publication_archives_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "journal_archive": manifest["journal_archive"] | {"files": "omitted"},
        "reproducibility_archive": manifest["reproducibility_archive"] | {
            "files": "omitted"
        },
    }, indent=2))


if __name__ == "__main__":
    main()
