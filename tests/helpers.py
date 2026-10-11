"""Helpers shared by the tests: building small projects and running the command line."""

import json
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from numeric_claim.cli import main

DEFAULT_CONFIG = 'claims = ["claims.py"]\ndistributions = []\n'
LOGGING_HEADER = textwrap.dedent(
    """
    import json
    import math
    from pathlib import Path

    from numeric_claim import claim, closed_form

    HERE = Path(__file__).parent


    def log(file="calls.log", **arguments):
        with (HERE / file).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(arguments) + "\\n")
    """
)


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


def logging_claims(*definitions: str) -> str:
    """A claims module whose claims can ``log`` the arguments of each call to ``calls.log``."""
    return LOGGING_HEADER + "".join(
        "\n\n" + textwrap.dedent(definition).strip() + "\n" for definition in definitions
    )


def logged(root: Path, file: str = "calls.log") -> list[dict[str, Any]]:
    """The arguments of each call logged under ``root``, in order."""
    log = root / file
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def json_report(root: Path, *args: str) -> tuple[int, dict[str, Any]]:
    """Run the command line with ``--json``; return the exit status and the report."""
    path = root / "report.json"
    status = run(root, *args, "--json", str(path))
    return status, json.loads(path.read_text(encoding="utf-8"))


def lock_data(root: Path) -> dict[str, Any]:
    """The claims stored in ``root/claims.lock.json``."""
    text = (root / "claims.lock.json").read_text(encoding="utf-8")
    claims: dict[str, Any] = json.loads(text)["claims"]
    return claims
