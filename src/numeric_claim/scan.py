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


def find_markers(file: str, text: str) -> list[Occurrence]:
    """Markdown sentinels and LaTeX macros in ``text``."""
    found: list[Occurrence] = []
    for origin, regex in (("sentinel", _SENTINEL), ("latex", _MACRO)):
        for match in regex.finditer(text):
            line = _line(text, match.start())
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
    for file in scan_files(config.root, config.scan):
        found.extend(find_markers(file, text_of(file)))
    for binding in config.bindings:
        for file in binding.files:
            found.extend(find_matches(file, text_of(file), binding))
    return sorted(found, key=lambda occurrence: (occurrence.file, occurrence.line))
