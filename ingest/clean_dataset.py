"""
Clean and reconstruct the raw Epstein Files dataset.

The raw data is messy — fragmented lines from scanned PDFs, OCR artifacts,
broken sentences, and garbled unicode. This script:

1. Groups scattered lines back into full documents (by filename)
2. Cleans OCR junk, control characters, and unicode issues
3. Fixes fragmented sentences from bad line breaks
4. Assigns a unique ID (UUID) to each document for traceability
5. Filters out documents that are too short to be useful

Input:  data/raw.json
Output: data/cleaned.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
import uuid
from typing import Any, Dict, List

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, CLEANED_DATA_PATH, MIN_DOC_LENGTH  # noqa: E402


# ──────────────────────────────────────────────
# Regex Patterns
# ──────────────────────────────────────────────

# Matches lines that start with a filename (e.g., "some_file.txt,rest of text")
FILENAME_RE = re.compile(r'^([A-Za-z0-9_\-\.]+\.txt),?"?(.*)$')

# Matches control characters, null bytes, form feeds, etc. — common OCR garbage
CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

# Matches runs of repeated special characters that are clearly OCR noise
# e.g., "||||||||", "________", "########"
OCR_NOISE_RE = re.compile(r'([|_#@~`]{3,})')

# Matches 3+ consecutive newlines — we'll collapse these to 2
EXCESS_NEWLINES_RE = re.compile(r'\n{3,}')

# Matches 2+ consecutive spaces/tabs — we'll collapse to a single space
EXCESS_SPACES_RE = re.compile(r'[ \t]{2,}')


# ──────────────────────────────────────────────
# Unicode Normalization
# ──────────────────────────────────────────────

# Common mojibake (garbled unicode) replacements from badly encoded PDFs
UNICODE_FIXES = {
    "â€œ": '"',   # left double quote
    "â€\x9d": '"',  # right double quote
    "â€˜": "'",   # left single quote
    "â€™": "'",   # right single quote / apostrophe
    "â€”": "—",   # em dash
    "â€“": "–",   # en dash
    "â€¦": "…",   # ellipsis
    "Ã©": "é",    # accented e (common in names)
    "Ã¨": "è",
    "Ã¼": "ü",
    "Ã¶": "ö",
    "Ã¤": "ä",
    "\xa0": " ",  # non-breaking space → regular space
}


def fix_unicode(text: str) -> str:
    """
    Fix common unicode/encoding issues from OCR'd PDFs.
    First applies known mojibake replacements, then normalizes to NFC form.
    """
    for bad, good in UNICODE_FIXES.items():
        text = text.replace(bad, good)

    # NFC normalization — standardizes accented characters to a single form
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """
    Clean a single document's text:
    - Fix unicode/encoding issues
    - Strip OCR artifacts and control characters
    - Collapse excessive whitespace
    - Fix fragmented sentences from bad PDF line breaks
    """
    # Step 1: Fix garbled unicode before anything else
    text = fix_unicode(text)

    # Step 2: Remove control characters (null bytes, form feeds, etc.)
    text = CONTROL_CHARS_RE.sub("", text)

    # Step 3: Remove OCR noise patterns (runs of |||, ___, ###, etc.)
    text = OCR_NOISE_RE.sub(" ", text)

    # Step 4: Fix fragmented sentences from PDF line breaks
    # If a line ends without punctuation and the next line starts lowercase,
    # it's probably a continuation — join them with a space instead of a newline
    lines = text.split("\n")
    merged_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged_lines.append("")
            continue

        if (
            merged_lines
            and merged_lines[-1]
            and not merged_lines[-1].rstrip().endswith((".", "!", "?", ":", '"', ")"))
            and stripped[0].islower()
        ):
            merged_lines[-1] = merged_lines[-1].rstrip() + " " + stripped
        else:
            merged_lines.append(stripped)

    text = "\n".join(merged_lines)

    # Step 5: Collapse excessive whitespace
    text = EXCESS_NEWLINES_RE.sub("\n\n", text)
    text = EXCESS_SPACES_RE.sub(" ", text)

    return text.strip()


def group_lines_into_documents(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The raw dataset has one line per row. Lines belonging to the same file
    need to be grouped back into full documents.

    Each new document starts with a line matching the filename pattern.
    Everything else gets appended to the current document.
    """
    docs: List[Dict[str, Any]] = []
    current_file: str | None = None
    buffer: List[str] = []

    def flush() -> None:
        """Save the current buffer as a document (if it has enough content)."""
        nonlocal buffer, current_file

        if not current_file or not buffer:
            return

        full_text = " ".join([x for x in buffer if x and x.strip()])
        cleaned = clean_text(full_text)

        if len(cleaned) >= MIN_DOC_LENGTH:
            docs.append(
                {
                    "doc_id": str(uuid.uuid4()),
                    "file": current_file,
                    "text": cleaned,
                    "char_count": len(cleaned),
                }
            )

    for row in rows:
        line = str(row.get("text", "")).strip()

        # Skip empty lines and the CSV header if present
        if not line or line.lower() == "filename,text":
            continue

        match = FILENAME_RE.match(line)
        if match:
            # Save previous doc
            flush()

            current_file = match.group(1)
            first_fragment = match.group(2).strip()
            buffer = [first_fragment] if first_fragment else []
        else:
            # Regular line — append to current document
            buffer.append(line)

    # Don't forget the last document
    flush()
    return docs


def main() -> None:
    # Load raw data
    print(f"📂 Loading raw data from {RAW_DATA_PATH}")
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    print(f"✅ Loaded {len(rows)} raw records")

    # Group + clean
    print("🧹 Cleaning and reconstructing documents...")
    docs = group_lines_into_documents(rows)
    print(f"✅ Reconstructed {len(docs)} clean documents")

    # Ensure output directory exists
    out_dir = os.path.dirname(CLEANED_DATA_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save cleaned data
    with open(CLEANED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to {CLEANED_DATA_PATH}")

    # Quick stats
    if docs:
        total_chars = sum(d["char_count"] for d in docs)
        avg_chars = total_chars // len(docs)
        min_chars = min(d["char_count"] for d in docs)
        max_chars = max(d["char_count"] for d in docs)

        print(f"📊 Stats: {total_chars:,} total chars | {avg_chars:,} avg chars/doc")
        print(f"📊 Shortest doc: {min_chars:,} chars")
        print(f"📊 Longest doc: {max_chars:,} chars")
    else:
        print("⚠️ No docs were produced. Check RAW_DATA_PATH and MIN_DOC_LENGTH.")


if __name__ == "__main__":
    main()