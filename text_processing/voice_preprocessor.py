"""Lightweight real-time text cleaning for TTS.

Works on individual sentences / short text (not full markdown documents).
Uses only regex — no heavy dependencies.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Ordinal map for numbered-list conversion (1–10)
# ---------------------------------------------------------------------------
_ORDINALS = {
    1: "First",
    2: "Second",
    3: "Third",
    4: "Fourth",
    5: "Fifth",
    6: "Sixth",
    7: "Seventh",
    8: "Eighth",
    9: "Ninth",
    10: "Tenth",
}

# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------
_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_COMPLEX = re.compile(r"`([^`]*[^\w\s`][^`]*)`")
_INLINE_CODE_SIMPLE = re.compile(r"`(\w+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"\*(.+?)\*")

# Verbalization: symbols → spoken words for short inline code (order matters: longest first)
_CODE_SYMBOL_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"==="), " equals "),
    (re.compile(r"!=="), " not equals "),
    (re.compile(r"=="), " equals "),
    (re.compile(r"!="), " not equals "),
    (re.compile(r"=>"), " arrow "),
    (re.compile(r"<="), " less than or equal to "),
    (re.compile(r">="), " greater than or equal to "),
    (re.compile(r"\+"), " plus "),
    (re.compile(r"\*\*"), " power "),
    (re.compile(r"\*"), " times "),
    (re.compile(r"%"), " modulo "),
    (re.compile(r"(?<=\s)-(?=\s)|^-$|(?<=\s)-$|^-(?=\s)"), " minus "),
    (re.compile(r"<"), " less than "),
    (re.compile(r">"), " greater than "),
    (re.compile(r"\(\)"), ""),
    (re.compile(r"\."), " dot "),
    (re.compile(r"[\[\]{}()]"), ""),
    (re.compile(r"""[\"'`]"""), ""),
    (re.compile(r"(?<!\w)/(?!\w)"), " divided by "),
]
_VERBALIZE_MAX_LEN = 40
_NUMBERED_LIST = re.compile(r"(?m)^(\d+)\.\s")
_BULLET = re.compile(r"(?m)^[-*]\s")
_URL = re.compile(r"https?://\S+")
_ABBREVIATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\be\.g\.\B"), "for example"),
    (re.compile(r"\be\.g\."), "for example"),
    (re.compile(r"\bi\.e\.\B"), "that is"),
    (re.compile(r"\bi\.e\."), "that is"),
    (re.compile(r"\betc\."), "and so on"),
    (re.compile(r"\bvs\."), "versus"),
]
_MULTI_WHITESPACE = re.compile(r"[ \t]+")


def _numbered_replacement(match: re.Match[str]) -> str:
    num = int(match.group(1))
    if num in _ORDINALS:
        return f"{_ORDINALS[num]}, "
    return f"Step {num}, "


def _verbalize_code(code: str) -> str | None:
    """Try to convert short inline code into speakable text.

    Returns None if the code is too complex to verbalize.
    """
    if len(code) > _VERBALIZE_MAX_LEN:
        return None
    result = code
    for pattern, replacement in _CODE_SYMBOL_MAP:
        result = pattern.sub(replacement, result)
    result = re.sub(r"\s+", " ", result).strip()
    # Reject if it still contains non-word chars (except hyphens, commas, spaces)
    if re.search(r"[^ \w,\-]", result):
        return None
    return result or None


def _extract_identifiers(code: str) -> str | None:
    """Extract meaningful identifiers from complex code.

    Splits on separators (::, ->, ., /), strips arguments and operators,
    returns space-separated identifier names.  Returns None if nothing useful.
    """
    cleaned = re.sub(r"\(.*?\)", "", code)        # strip parenthesized args
    cleaned = re.sub(r"\[.*?\]", "", cleaned)      # strip bracket indexing
    cleaned = re.sub(r"[<>]", "", cleaned)         # strip angle brackets
    cleaned = re.sub(r"""[\"'`]""", "", cleaned)   # strip quotes
    cleaned = re.sub(r"\d+\.\.\d+", "", cleaned)   # strip range literals

    parts = re.split(r"::|->|[./]", cleaned)
    parts = [re.sub(r"[^a-zA-Z0-9_]", "", p).strip() for p in parts]
    parts = [p for p in parts if p]

    if not parts:
        return None

    # Convert snake_case/camelCase to readable words
    spoken_parts: list[str] = []
    for p in parts:
        p = p.replace("_", " ")
        p = re.sub(r"([a-z])([A-Z])", r"\1 \2", p)
        spoken_parts.append(p.lower())

    spoken = " ".join(spoken_parts)
    spoken = re.sub(r"\s+", " ", spoken).strip()

    if not spoken or len(spoken) < 2 or len(spoken.split()) > 8:
        return None

    return spoken


def _replace_inline_code(match: re.Match[str]) -> str:
    """Replace complex inline code — try to verbalize, extract, or omit."""
    content = match.group(1)
    # Tier 1: symbol-to-word verbalization for short snippets
    verbalized = _verbalize_code(content)
    if verbalized:
        return verbalized
    # Tier 2: extract meaningful identifiers
    extracted = _extract_identifiers(content)
    if extracted:
        return extracted
    # Tier 3: omit entirely — surrounding sentence provides context
    return ""


def clean_for_speech(text: str) -> str:
    """Clean *text* so it reads naturally when spoken by a TTS engine."""
    if not text or not text.strip():
        return text.strip()

    # 1. Fenced code blocks → placeholder
    text = _FENCED_CODE.sub("See the example in the chat.", text)

    # 2. Inline code — complex first (try to verbalize), then simple identifiers
    text = _INLINE_CODE_COMPLEX.sub(_replace_inline_code, text)
    text = _INLINE_CODE_SIMPLE.sub(r"\1", text)

    # 3. Bold / italic markers
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)

    # 4. Numbered-list prefixes → ordinals
    text = _NUMBERED_LIST.sub(_numbered_replacement, text)

    # 5. Bullet markers
    text = _BULLET.sub("", text)

    # 6. URLs
    text = _URL.sub("see the link in the chat", text)

    # 7. Abbreviations
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    # 8. Collapse multiple whitespace
    text = _MULTI_WHITESPACE.sub(" ", text)

    # 9. Strip leading / trailing whitespace
    text = text.strip()

    return text


_FENCE_MARKER = re.compile(r"^```")


class StreamingVoicePreprocessor:
    """Stateful preprocessor that tracks code-fence state across streaming chunks.

    Create one instance per conversation turn. Call ``clean()`` for each sentence
    produced by the sentence splitter.  Code blocks split across chunks are
    properly suppressed — the first fence triggers a "See the example in the
    chat." announcement, and all content until the closing fence is dropped.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._in_code_block = False
        self._announced = False
        self._enabled = enabled

    def clean(self, text: str) -> str:
        """Clean *text* for TTS, respecting streaming code-block state."""
        if not self._enabled:
            return text
        if not text or not text.strip():
            return text.strip()

        # Process line-by-line to catch fence markers
        out_lines: list[str] = []
        for line in text.split("\n"):
            if _FENCE_MARKER.match(line.strip()):
                if not self._in_code_block:
                    # Entering code block
                    self._in_code_block = True
                    if not self._announced:
                        out_lines.append("See the example in the chat.")
                        self._announced = True
                else:
                    # Exiting code block
                    self._in_code_block = False
                continue

            if self._in_code_block:
                # Suppress code block content
                continue

            out_lines.append(line)

        remaining = "\n".join(out_lines)
        if not remaining.strip():
            return ""

        # Apply the regular stateless cleanup to whatever is left
        return clean_for_speech(remaining)
