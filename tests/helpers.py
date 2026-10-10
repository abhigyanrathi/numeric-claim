"""Helpers shared by the tests: building small projects and running the command line."""

import textwrap
from collections.abc import Mapping
from pathlib import Path

from numeric_claim.cli import main

DEFAULT_CONFIG = 'claims = ["claims.py"]\ndistributions = []\n'


def write(root: Path, relative: str, text: str) -> Path:
    """Write a UTF-8 file with LF line endings, creating parent directories."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def make_project(
    root: Path,
    claims: str,
    files: Mapping[str, str] | None = None,
    config: str = DEFAULT_CONFIG,
) -> None:
    """Create ``numeric-claim.toml``, ``claims.py`` and any other files under ``root``."""
    write(root, "numeric-claim.toml", config)
    write(root, "claims.py", claims)
    for name, text in (files or {}).items():
        write(root, name, text)


def run(root: Path, *args: str) -> int:
    """Run the command line against ``root/numeric-claim.toml`` and return the exit status."""
    return main([*args, "--config", str(root / "numeric-claim.toml")])


def counting_claims(kind: str = "exact", options: str = "", cast: str = "int") -> str:
    """A claims module whose ``answer`` claim logs each call and reads ``value.txt``."""
    return textwrap.dedent(
        f"""
        from pathlib import Path

        from numeric_claim import claim

        HERE = Path(__file__).parent


        @claim("answer", kind="{kind}"{options})
        def answer():
            with (HERE / "calls.log").open("a", encoding="utf-8") as handle:
                handle.write("call\\n")
            return {cast}((HERE / "value.txt").read_text())
        """
    )


def calls(root: Path) -> int:
    """How many times the counting claim has run under ``root``."""
    log = root / "calls.log"
    return len(log.read_text().splitlines()) if log.exists() else 0
