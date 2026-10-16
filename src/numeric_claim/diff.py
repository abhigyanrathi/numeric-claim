"""Comparing the lock file committed at a git revision with the current one."""

import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from numeric_claim import lockfile
from numeric_claim.budget import BAND
from numeric_claim.claim import Value
from numeric_claim.errors import ConfigError
from numeric_claim.lockfile import LockEntry
from numeric_claim.montecarlo import welch

_ABSENT = re.compile(r"does not exist in '|exists on disk, but not in '")


@dataclass(frozen=True)
class Change:
    """How one claim's lock entry changed between a revision and now.

    ``status`` is added, removed, unchanged, within-noise or moved. ``fingerprint_changed`` and
    ``environment``, the keys of the environment block that differ, are None for a claim found in
    one lock only.
    """

    name: str
    status: str
    old: Value | None
    new: Value | None
    fingerprint_changed: bool | None = None
    environment: tuple[str, ...] | None = None


def committed(path: Path, revision: str) -> dict[str, LockEntry]:
    """The lock file ``path`` as committed at ``revision``; empty if it did not exist there."""
    if revision.startswith("-"):
        raise ConfigError(f"invalid revision {revision!r}")
    target = f"{revision}:./{path.name}"
    try:
        completed = subprocess.run(
            ["git", "-C", str(path.parent), "show", target],
            capture_output=True,
            env={**os.environ, "LC_ALL": "C"},
            check=False,
        )
    except OSError as exc:
        raise ConfigError(f"cannot run git: {exc.strerror or exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip()
        if _ABSENT.search(message):
            return {}
        raise ConfigError(f"git show {target} failed: {message}")
    try:
        text = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"lock file {target} is not valid UTF-8 ({exc.reason})") from exc
    return lockfile.loads(text, target)


def _moved(old: LockEntry, new: LockEntry, alpha: float) -> bool:
    if old.kind != new.kind or old.kind == "exact":
        return True
    if old.summary is not None and new.summary is not None:
        return welch(old.summary, new.summary) <= alpha
    budgets = (entry.budget for entry in (old, new) if entry.budget is not None)
    sigma = max((budget.sigma for budget in budgets), default=0.0)
    tolerance = max(old.atol, old.rtol * abs(old.value))
    return abs(new.value - old.value) > max(tolerance, BAND * sigma)


def _differing(old: Mapping[str, object], new: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key)))


def compare(
    old: Mapping[str, LockEntry], new: Mapping[str, LockEntry], alpha: float
) -> list[Change]:
    """How each claim in either lock changed, in name order.

    A seeded claim moves by more than its old tolerance or 3 sigma of either budget; a random
    claim moves when Welch's test separates the two means at ``alpha``.
    """
    changes: list[Change] = []
    for name in sorted(old.keys() | new.keys()):
        before, after = old.get(name), new.get(name)
        if before is None or after is None:
            status = "added" if before is None else "removed"
            old_value = None if before is None else before.value
            new_value = None if after is None else after.value
            changes.append(Change(name, status, old_value, new_value))
            continue
        if before.kind == after.kind and before.value == after.value:
            status = "unchanged"
        else:
            status = "moved" if _moved(before, after, alpha) else "within-noise"
        changed = before.fingerprint != after.fingerprint
        environment = _differing(before.environment, after.environment)
        changes.append(Change(name, status, before.value, after.value, changed, environment))
    return changes
