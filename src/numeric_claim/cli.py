"""The numeric-claim command line."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from numeric_claim import lockfile
from numeric_claim.claim import ClaimContext
from numeric_claim.config import Config, load_config
from numeric_claim.diff import committed, compare
from numeric_claim.engine import check_claim, judge_all, lock_claim
from numeric_claim.errors import ConfigError
from numeric_claim.loader import LoadedClaim, load_claims
from numeric_claim.report import Report, check_report, diff_report, list_report, lock_report
from numeric_claim.scan import collect

COMMANDS = ("lock", "check", "list", "diff")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numeric-claim",
        description=(
            "Check that numbers published in documents agree with values computed "
            "from the project's own code."
        ),
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="lock: compute every claim and write the lock file; "
        "check: compare documents and the lock with the code; "
        "list: show claims and document occurrences without computing; "
        "diff: compare the lock file committed at a git revision with the current one",
    )
    parser.add_argument(
        "revision",
        nargs="?",
        metavar="REV",
        help="diff only: the git revision of the lock file to compare with (default: HEAD)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("numeric-claim.toml"),
        help="configuration file (default: ./numeric-claim.toml)",
    )
    parser.add_argument(
        "--root", type=Path, help="project root (default: the configuration file's directory)"
    )
    parser.add_argument("--lock", type=Path, help="lock file, overriding the configured one")
    parser.add_argument("--json", type=Path, dest="json_path", help="also write the report as JSON")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="check only: recompute every claim instead of reusing matching locked values",
    )
    return parser


def _emit(stream: TextIO, text: str) -> None:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    stream.write(text.encode(encoding, "backslashreplace").decode(encoding))


def _write_json(path: Path, report: Report) -> None:
    text = json.dumps(report.data, indent=2, sort_keys=True) + "\n"
    try:
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ConfigError(f"cannot write {path}: {exc.strerror or exc}") from exc


def _require_references(config: Config, claims: Sequence[LoadedClaim]) -> None:
    unreferenced = {loaded.spec.name for loaded in claims if loaded.spec.reference is None}
    for index, binding in enumerate(config.bindings, start=1):
        if binding.quantity != "error":
            continue
        for name in binding.claims:
            if name in unreferenced:
                raise ConfigError(
                    f"[[bind]] entry {index}: quantity 'error' needs a reference, "
                    f"and claim {name!r} declares none"
                )


def _diff(config: Config, revision: str) -> Report:
    if not config.lock.exists():
        raise ConfigError(f"lock file {config.lock} does not exist; run `numeric-claim lock`")
    current = lockfile.read(config.lock)
    changes = compare(committed(config.lock, revision), current, config.alpha)
    return diff_report(changes, revision)


def _run(args: argparse.Namespace) -> Report:
    if args.recompute and args.command != "check":
        raise ConfigError("--recompute applies to the check command only")
    if args.revision is not None and args.command != "diff":
        raise ConfigError("a revision applies to the diff command only")
    config = load_config(args.config, root=args.root, lock=args.lock)
    if args.command == "diff":
        return _diff(config, args.revision or "HEAD")
    claims = load_claims(config.claims)
    _require_references(config, claims)
    if args.command == "list":
        return list_report(claims, collect(config))
    context = ClaimContext(config.root, config.directory)
    if args.command == "lock":
        locked = [lock_claim(loaded, context, config.distributions) for loaded in claims]
        entries = {outcome.name: outcome.entry for outcome in locked if outcome.entry is not None}
        written = len(entries) == len(locked)
        if written:
            lockfile.write(config.lock, entries)
        return lock_report(locked, config.lock, written=written)
    occurrences = collect(config)
    stored = lockfile.read(config.lock)
    error_bound = {occurrence.claim for occurrence in occurrences if occurrence.quantity == "error"}
    outcomes = [
        check_claim(
            loaded,
            context,
            config.distributions,
            stored.get(loaded.spec.name),
            recompute=args.recompute,
            error_bound=loaded.spec.name in error_bound,
            alpha=config.alpha,
        )
        for loaded in claims
    ]
    results = judge_all(occurrences, claims, outcomes, config.alpha)
    return check_report(outcomes, results, config.alpha)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status: 0 ok, 1 failed, 2 configuration."""
    try:
        args = _parser().parse_intermixed_args(argv)
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else 2
    try:
        report = _run(args)
        _emit(sys.stdout, report.text)
        if args.json_path is not None:
            _write_json(args.json_path, report)
    except ConfigError as exc:
        _emit(sys.stderr, f"numeric-claim: error: {exc}\n")
        return 2
    return report.exit_code
