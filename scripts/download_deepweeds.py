"""Download and safely extract the official DeepWeeds image archive."""

from __future__ import annotations

import argparse
from http.cookiejar import CookieJar
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlencode
from urllib.request import build_opener, HTTPCookieProcessor, Request
from zipfile import BadZipFile, ZipFile


FILE_ID = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"
REPOSITORY_URL = "https://github.com/AlexOlsen/DeepWeeds"
DOWNLOAD_ENDPOINT = "https://drive.usercontent.google.com/download"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    query = urlencode({"id": FILE_ID, "export": "download", "confirm": "t"})
    request = Request(
        f"{DOWNLOAD_ENDPOINT}?{query}",
        headers={"User-Agent": "DeepWeeds research acquisition/1.0"},
    )
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    with opener.open(request, timeout=120) as response, temporary.open("wb") as output:
        if response.status != 200:
            raise RuntimeError(f"download returned HTTP {response.status}")
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if temporary.stat().st_size < 100_000_000:
        raise RuntimeError(
            f"download is unexpectedly small ({temporary.stat().st_size} bytes)"
        )
    with temporary.open("rb") as handle:
        if handle.read(4) != b"PK\x03\x04":
            raise RuntimeError("download is not a ZIP archive")
    temporary.replace(destination)


def safe_extract(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with ZipFile(archive) as bundle:
        members = bundle.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    return sum(1 for path in destination.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path, default=Path("data/source/DeepWeeds")
    )
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.source_root / "images.zip"
    image_directory = args.source_root / "images"
    if args.force_download or not archive.is_file():
        download(archive)
    try:
        image_count = safe_extract(archive, image_directory)
    except BadZipFile as exc:
        raise SystemExit(f"invalid archive: {exc}") from exc
    metadata = {
        "dataset": "DeepWeeds",
        "official_repository": REPOSITORY_URL,
        "google_drive_file_id": FILE_ID,
        "archive_path": str(archive.resolve()),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "extracted_jpeg_count": image_count,
        "expected_label_count": 17509,
        "status": "ACQUIRED" if image_count == 17509 else "COUNT_MISMATCH",
    }
    metadata_path = args.source_root / "acquisition.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if image_count != 17509:
        raise SystemExit("extracted image count does not match official label count")


if __name__ == "__main__":
    main()
