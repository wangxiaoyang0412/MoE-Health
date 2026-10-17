"""Section extraction on synthetic report text."""

from __future__ import annotations

from src.data_module.note_features import salient_text, split_sections

REPORT = """EXAMINATION:  CHEST (PORTABLE AP)

INDICATION:  ___ year old man with dyspnea.

COMPARISON:  None.

FINDINGS:

Mild pulmonary edema.  No pneumothorax.

IMPRESSION:

Mild edema, improved since ___.
"""


def test_split_sections_finds_headings() -> None:
    names = [n for n, _ in split_sections(REPORT)]
    assert names == ["EXAMINATION", "INDICATION", "COMPARISON", "FINDINGS", "IMPRESSION"]


def test_salient_text_keeps_findings_and_impression_only() -> None:
    text, found = salient_text(REPORT)
    assert found
    assert text == "FINDINGS: Mild pulmonary edema. No pneumothorax. IMPRESSION: Mild edema, improved since ."
    assert "INDICATION" not in text and "___" not in text


def test_salient_text_falls_back_to_full_report() -> None:
    text, found = salient_text("Portable chest film shows ___ line in place.")
    assert not found
    assert text == "Portable chest film shows line in place."
