"""Finding the numbers documents print: sentinels, LaTeX macros and configured patterns."""

import re
from dataclasses import dataclass
from pathlib import Path

from numeric_claim.config import Binding, Config
from numeric_claim.errors import ConfigError
from numeric_claim.globs import scan_files
from numeric_claim.relations import RELATIONS

_SENTINEL = re.compile(
    r"<!--\s*claim:\s*(?P<name>[^\s>]+?)(?:\s+(?P<relation>[^\s>]+?))?\s*-->"
    r"(?P<literal>.*?)<!--\s*/claim\s*-->"
)
_MACRO = re.compile(
    r"\\claim\s*(?:\[(?P<relation>[^\]]*)\]\s*)?\{(?P<name>[^{}]*)\}\s*\{(?P<literal>[^{}]*)\}"
)
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
_CLOSING_FENCE = re.compile(r" {0,3}(`{3,}|~{3,}) *")
_BACKTICKS = re.compile(r"`+")


@dataclass(frozen=True)
class Occurrence:
    """One place where a document prints a number tied to a claim."""

    file: str
    line: int
    claim: str
    relation: str
    literal: str
    origin: str
    quantity: str = "value"


def _line(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def _relation(file: str, line: int, raw: str | None) -> str:
    relation = (raw or "").strip() or "eq"
    if relation not in RELATIONS:
        raise ConfigError(f"{file}:{line}: unknown relation {relation!r}")
    return relation


def _blank_spans(line: str) -> str:
    runs = list(_BACKTICKS.finditer(line))
    pieces: list[str] = []
    position = index = 0
    while index < len(runs):
        opening = runs[index]
        width = len(opening[0])
        partner = next((j for j in range(index + 1, len(runs)) if len(runs[j][0]) == width), None)
        if partner is None:
            index += 1
            continue
        end = runs[partner].end()
        pieces += [line[position : opening.start()], " " * (end - opening.start())]
        position, index = end, partner + 1
    return "".join(pieces) + line[position:]


def blank_code(text: str) -> str:
    """Markdown ``text`` with fenced code blocks and inline code spans turned into spaces."""
    lines = text.split("\n")
    fence = ""
    for number, line in enumerate(lines):
        if not fence:
            opening = _FENCE.match(line)
            if opening is None:
                lines[number] = _blank_spans(line)
                continue
            fence = opening[1]
        else:
            closing = _CLOSING_FENCE.fullmatch(line)
            if closing and closing[1][0] == fence[0] and len(closing[1]) >= len(fence):
                fence = ""
        lines[number] = " " * len(line)
    return "\n".join(lines)


def find_markers(file: str, text: str) -> list[Occurrence]:
    """Markdown sentinels and LaTeX macros in ``text``, outside code if ``file`` is Markdown."""
    searched = blank_code(text) if file.endswith(".md") else text
    found: list[Occurrence] = []
    for origin, regex in (("sentinel", _SENTINEL), ("latex", _MACRO)):
        for match in regex.finditer(searched):
            line = _line(searched, match.start())
            relation = _relation(file, line, match["relation"])
            name, literal = match["name"].strip(), match["literal"].strip()
            found.append(Occurrence(file, line, name, relation, literal, origin))
    return found


def find_matches(file: str, text: str, binding: Binding) -> list[Occurrence]:
    """Every match of a binding's pattern in ``text``, once per claim the binding lists."""
    found: list[Occurrence] = []
    for match in binding.pattern.finditer(text):
        start = match.start(1)
        line = _line(text, start if start >= 0 else match.start())
        literal = match.group(1) or ""
        found.extend(
            Occurrence(file, line, name, binding.relation, literal, "pattern", binding.quantity)
            for name in binding.claims
        )
    return found


def _read(root: Path, file: str) -> str:
    try:
        return (root / file).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{file}: not valid UTF-8 ({exc.reason})") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {file}: {exc.strerror or exc}") from exc


def collect(config: Config) -> list[Occurrence]:
    """Every occurrence in the project's documents, ordered by file and line."""
    texts: dict[str, str] = {}

    def text_of(file: str) -> str:
        if file not in texts:
            texts[file] = _read(config.root, file)
        return texts[file]

    found: list[Occurrence] = []
    for file in scan_files(config.root, config.scan, config.exclude):
        found.extend(find_markers(file, text_of(file)))
    for binding in config.bindings:
        for file in binding.files:
            found.extend(find_matches(file, text_of(file), binding))
    return sorted(found, key=lambda occurrence: (occurrence.file, occurrence.line))
