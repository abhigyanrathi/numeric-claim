"""Claims for the cosfin checkout at commit 5f51ee5."""

import os
import re
import subprocess
import sys

from numeric_claim import ClaimContext, claim

_COLLECTED = re.compile(r"(\d+) tests? collected")


@claim("test_count", kind="exact", paths=["src", "tests"])
def test_count(ctx: ClaimContext) -> int:
    """How many tests pytest collects in the checkout."""
    source = str(ctx.root / "src")
    inherited = os.environ.get("PYTHONPATH")
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([source, inherited]) if inherited else source,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ctx.root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    lines = completed.stdout.splitlines()
    if completed.returncode == 0:
        for line in reversed(lines):
            match = _COLLECTED.match(line)
            if match:
                return int(match.group(1))
    tail = "\n".join([*lines, *completed.stderr.splitlines()][-20:])
    raise RuntimeError(
        f"pytest --collect-only did not report a test count (exit {completed.returncode}):\n{tail}"
    )
