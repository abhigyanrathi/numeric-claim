"""Loading claims modules and collecting the claims defined in them."""

import hashlib
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numeric_claim.claim import ClaimSpec, spec_of
from numeric_claim.errors import ConfigError


@dataclass(frozen=True)
class LoadedClaim:
    """A claim function with its metadata and the module file that defines it."""

    spec: ClaimSpec
    function: Callable[..., Any]
    module: Path


def _load_module(path: Path) -> list[LoadedClaim]:
    if not path.is_file():
        raise ConfigError(f"claims module not found: {path}")
    name = "_numeric_claim_module_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ConfigError(f"cannot load claims module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except (Exception, SystemExit) as exc:
        del sys.modules[name]
        raise ConfigError(
            f"error while loading claims module {path}: {type(exc).__name__}: {exc}"
        ) from exc
    loaded: list[LoadedClaim] = []
    seen: set[int] = set()
    for candidate in vars(module).values():
        claim_spec = spec_of(candidate)
        if claim_spec is None or getattr(candidate, "__module__", None) != name:
            continue
        if id(candidate) not in seen:
            seen.add(id(candidate))
            loaded.append(LoadedClaim(claim_spec, candidate, path))
    return loaded


def load_claims(modules: Sequence[Path]) -> list[LoadedClaim]:
    """Every claim defined in ``modules``, sorted by name; a name defined twice is an error."""
    found: dict[str, LoadedClaim] = {}
    for path in modules:
        for loaded in _load_module(path):
            name = loaded.spec.name
            if name in found:
                raise ConfigError(
                    f"claim {name!r} is defined twice: in {found[name].module} and in {path}"
                )
            found[name] = loaded
    return [found[name] for name in sorted(found)]
