#!/usr/bin/env python3
"""Generate a traceable BibTeX file from the verified literature database."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _clean(value: str) -> str:
    return " ".join((value or "").split())


def _escape(value: str) -> str:
    return _clean(value).replace("&", r"\&")


def _entry(row: dict[str, str]) -> str:
    publication_type = row["publication_type"].strip()
    is_preprint = publication_type == "preprint"
    kind = "misc" if is_preprint else "inproceedings"
    fields: list[tuple[str, str]] = [
        ("title", "{{" + _escape(row["title"]) + "}}"),
        ("author", "{" + " and ".join(
            _escape(author) for author in row["authors"].split(";")
        ) + "}"),
        ("year", "{" + row["year"].strip() + "}"),
    ]
    if is_preprint:
        arxiv = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", row["url"])
        fields.append(("howpublished", "{arXiv preprint}"))
        if arxiv:
            fields.extend([
                ("eprint", "{" + arxiv.group(1) + "}"),
                ("archiveprefix", "{arXiv}"),
            ])
    else:
        fields.append(("booktitle", "{" + _escape(row["venue"]) + "}"))
    if row["doi_verified"].strip():
        fields.append(("doi", "{" + row["doi_verified"].strip() + "}"))
    fields.append(("url", "{" + row["url"].strip() + "}"))
    body = ",\n".join(f"  {name} = {value}" for name, value in fields)
    return f"@{kind}{{{row['paper_id']},\n{body}\n}}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path,
                        default=Path("literature/literature_database.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("paper/references.bib"))
    args = parser.parse_args()
    with args.database.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"paper_id", "title", "authors", "year", "venue",
                "doi_verified", "url", "publication_type"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit("literature database is empty or missing required fields")
    ids = [row["paper_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate paper_id values in literature database")
    entries = [_entry(row) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} verified-source entries to {args.output}")


if __name__ == "__main__":
    main()
