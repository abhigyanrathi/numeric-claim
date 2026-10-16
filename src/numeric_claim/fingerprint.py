"""Fingerprints: everything a locked value depends on, hashed."""

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numeric_claim import __version__
from numeric_claim.claim import ClaimContext, ClaimSpec
from numeric_claim.errors import ClaimError


@dataclass(frozen=True)
class Fingerprint:
    """The digest of a claim's inputs and the environment it was computed in."""

    digest: str
    environment: dict[str, Any]


def environment(distributions: Iterable[str]) -> dict[str, Any]:
    """The interpreter, platform and installed versions a value was computed under."""
    versions: dict[str, str] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "distributions": versions,
        "machine": platform.machine(),
        "numeric_claim": __version__,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sys_platform": sys.platform,
    }


def _relative(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return path.as_posix()


def _files(top: Path) -> list[Path]:
    if top.is_file():
        return [top]
    found: list[Path] = []
    for directory, subdirectories, names in os.walk(top):
        subdirectories[:] = [
            name
            for name in subdirectories
            if name != "__pycache__" and not name.endswith(".egg-info")
        ]
        found.extend(Path(directory) / name for name in names if not name.endswith(".pyc"))
    return sorted(found)


def _syntax(data: bytes) -> bytes:
    """The syntax tree of Python source as bytes; the source itself if it does not parse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tree = ast.parse(data)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return data
    return ast.dump(tree, include_attributes=False).encode("utf-8")


def _digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".py":
        data = _syntax(data)
    return hashlib.sha256(data).hexdigest()


def _package_locations(name: str) -> list[Path]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError) as exc:
        raise ClaimError(f"cannot locate package {name!r}: {exc}") from exc
    if spec is None:
        raise ClaimError(f"package {name!r} not found")
    if spec.submodule_search_locations:
        return [Path(location) for location in spec.submodule_search_locations]
    if spec.has_location and spec.origin:
        return [Path(spec.origin)]
    raise ClaimError(f"package {name!r} has no source files")


def _sources(spec: ClaimSpec, module: Path, context: ClaimContext) -> dict[str, str]:
    sources = {f"module:{_relative(module, context.config_dir)}": _digest(module)}
    for name in spec.packages:
        locations = _package_locations(name)
        for index, location in enumerate(locations):
            label = f"package:{name}" if len(locations) == 1 else f"package:{name}#{index}"
            base = location if location.is_dir() else location.parent
            for file in _files(location):
                sources[f"{label}:{_relative(file, base)}"] = _digest(file)
    for entry in spec.paths:
        target = context.root / entry
        if not target.exists():
            raise ClaimError(f"declared path {entry!r} does not exist under {context.root}")
        for file in _files(target):
            sources[f"path:{_relative(file, context.root)}"] = _digest(file)
    return sources


def compute(
    spec: ClaimSpec, module: Path, context: ClaimContext, distributions: Iterable[str]
) -> Fingerprint:
    """Fingerprint a claim: its module, declared packages and paths, and the environment."""
    env = environment(distributions)
    try:
        sources = _sources(spec, module, context)
    except OSError as exc:
        raise ClaimError(f"cannot read sources for fingerprint: {exc}") from exc
    payload = {"environment": env, "sources": sources}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return Fingerprint(hashlib.sha256(canonical.encode("ascii")).hexdigest(), env)
