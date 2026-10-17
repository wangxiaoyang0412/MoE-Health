"""Radiology-report preprocessing: salient sections and de-identification cleanup.

Paper: "removing de-identification artifacts, and extracting salient sections
(e.g. Impression, Findings)". Reports without either section fall back to the
whole report so that no admission with a note loses its text modality.
"""

from __future__ import annotations

import re

__all__ = ["SALIENT_PREFIXES", "salient_text", "split_sections"]

HEADING = re.compile(r"^[ \t]*([A-Z][A-Z0-9 /&(),-]{1,60}):", re.MULTILINE)
DEID = re.compile(r"_{2,}")
WHITESPACE = re.compile(r"\s+")
SALIENT_PREFIXES = ("FINDING", "IMPRESSION")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a report into (HEADING, body) pairs using upper-case 'HEADING:' lines."""
    matches = list(HEADING.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(1).strip(), text[m.end() : end].strip()))
    return sections


def salient_text(text: str) -> tuple[str, bool]:
    """Return (cleaned text, whether salient sections were found)."""
    keep = [
        f"{name}: {body}"
        for name, body in split_sections(text)
        if name.startswith(SALIENT_PREFIXES) and body
    ]
    body = " ".join(keep) if keep else text
    body = DEID.sub(" ", body)
    return WHITESPACE.sub(" ", body).strip(), bool(keep)
