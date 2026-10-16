"""Finding files by glob, over paths relative to a root written with forward slashes."""

import fnmatch
import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

SKIPPED_DIRECTORIES = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"}
)


def _skipped(name: str) -> bool:
    return name in SKIPPED_DIRECTORIES or name.startswith(".")


def walk_files(root: Path, *, prune: bool) -> list[str]:
    """Sorted relative paths of the files under ``root``; ``prune`` skips ignored directories."""
    found: list[str] = []
    for directory, subdirectories, names in os.walk(root):
        if prune:
            subdirectories[:] = [name for name in subdirectories if not _skipped(name)]
        base = Path(directory).relative_to(root)
        found.extend((base / name).as_posix() for name in names)
    return sorted(found)


def _match(parts: tuple[str, ...], patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return not parts
    head, rest = patterns[0], patterns[1:]
    if head == "**":
        return any(_match(parts[start:], rest) for start in range(len(parts) + 1))
    return bool(parts) and fnmatch.fnmatchcase(parts[0], head) and _match(parts[1:], rest)


def matches(path: str, pattern: str) -> bool:
    """Whether a relative path matches a glob in which ``**`` stands for any directories."""
    return _match(tuple(path.split("/")), tuple(pattern.split("/")))


def scan_files(root: Path, patterns: Iterable[str], excluded: Iterable[str] = ()) -> list[str]:
    """Files under ``root`` matching a pattern and no excluded one, skipping ignored directories."""
    wanted = [PurePosixPath(pattern).as_posix() for pattern in patterns]
    unwanted = [PurePosixPath(pattern).as_posix() for pattern in excluded]
    return [
        path
        for path in walk_files(root, prune=True)
        if any(matches(path, pattern) for pattern in wanted)
        and not any(matches(path, pattern) for pattern in unwanted)
    ]


def bind_files(root: Path, pattern: str) -> list[str]:
    """Files under ``root`` matching a configured glob; no directory is skipped."""
    normalized = PurePosixPath(pattern).as_posix()
    if not any(char in normalized for char in "*?["):
        return [normalized] if (root / normalized).is_file() else []
    return [path for path in walk_files(root, prune=False) if matches(path, normalized)]
