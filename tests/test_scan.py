from pathlib import Path

import pytest

from helpers import DEFAULT_CONFIG, make_project
from numeric_claim.config import load_config
from numeric_claim.errors import ConfigError
from numeric_claim.globs import matches
from numeric_claim.scan import Occurrence, collect

BIND = """
[[bind]]
files = {files}
pattern = '{pattern}'
claim = {claim}
{extra}
"""


def found(root: Path) -> list[Occurrence]:
    return collect(load_config(root / "numeric-claim.toml"))


def bound_config(
    files: str = '["README.md"]',
    pattern: str = r"(\d[\d,]*) tests\b",
    claim: str = '"count"',
    extra: str = "",
) -> str:
    return DEFAULT_CONFIG + BIND.format(files=files, pattern=pattern, claim=claim, extra=extra)


def test_markdown_sentinels_are_found_with_line_numbers_and_relations(tmp_path: Path) -> None:
    readme = (
        "# Title\n"
        "\n"
        "We run <!-- claim: test_count -->264<!-- /claim --> tests.\n"
        "\n"
        "Error <!-- claim: err lt -->4e-8<!-- /claim --> and "
        "<!--claim: other ge-->1,000<!--/claim-->.\n"
    )
    make_project(tmp_path, "", {"README.md": readme})
    assert found(tmp_path) == [
        Occurrence("README.md", 3, "test_count", "eq", "264", "sentinel"),
        Occurrence("README.md", 5, "err", "lt", "4e-8", "sentinel"),
        Occurrence("README.md", 5, "other", "ge", "1,000", "sentinel"),
    ]


def test_latex_macros_are_found_and_the_preamble_definition_is_not(tmp_path: Path) -> None:
    tex = (
        "\\newcommand{\\claim}[3][]{#3}\n"
        "Result \\claim{answer}{42} and \\claim[lt]{bound}{4e-8}, "
        "\\claim[]{plain}{7}.\n"
    )
    make_project(tmp_path, "", {"paper.tex": tex})
    assert found(tmp_path) == [
        Occurrence("paper.tex", 2, "answer", "eq", "42", "latex"),
        Occurrence("paper.tex", 2, "bound", "lt", "4e-8", "latex"),
        Occurrence("paper.tex", 2, "plain", "eq", "7", "latex"),
    ]


def test_patterns_find_every_match_once_per_listed_claim(tmp_path: Path) -> None:
    readme = "Intro\nWe run 1,024 tests daily.\n\nOnly 7 tests here, and 9 tests there.\n"
    config = bound_config(claim='["a", "b"]', extra='relation = "le"')
    make_project(tmp_path, "", {"README.md": readme}, config)
    expected = [
        Occurrence("README.md", line, claim, "le", literal, "pattern")
        for line, literal in [(2, "1,024"), (4, "7"), (4, "9")]
        for claim in ("a", "b")
    ]
    assert found(tmp_path) == expected


def test_a_pattern_defaults_to_eq_and_anchors_apply_per_line(tmp_path: Path) -> None:
    readme = "Tests: 12\nnot Tests: 13\nTests: 14\n"
    make_project(tmp_path, "", {"README.md": readme}, bound_config(pattern=r"^Tests: (\d+)$"))
    assert [(o.line, o.literal, o.relation) for o in found(tmp_path)] == [
        (1, "12", "eq"),
        (3, "14", "eq"),
    ]


def test_line_numbers_count_crlf_files_correctly(tmp_path: Path) -> None:
    make_project(tmp_path, "")
    (tmp_path / "doc.md").write_bytes(b"one\r\ntwo\r\n<!-- claim: x -->5<!-- /claim -->\r\n")
    assert [(o.file, o.line) for o in found(tmp_path)] == [("doc.md", 3)]


def test_a_sentinel_must_sit_on_one_line(tmp_path: Path) -> None:
    make_project(tmp_path, "", {"doc.md": "<!-- claim: x -->5\n<!-- /claim -->\n"})
    assert found(tmp_path) == []


def test_files_outside_scan_are_not_searched(tmp_path: Path) -> None:
    make_project(tmp_path, "", {"notes.txt": "<!-- claim: x -->5<!-- /claim -->\n"})
    assert found(tmp_path) == []
    config = 'claims = ["claims.py"]\ndistributions = []\nscan = ["*.txt"]\n'
    make_project(tmp_path, "", {"notes.txt": "<!-- claim: x -->5<!-- /claim -->\n"}, config)
    assert [o.file for o in found(tmp_path)] == ["notes.txt"]


def test_scan_skips_ignored_directories(tmp_path: Path) -> None:
    sentinel = "<!-- claim: x -->5<!-- /claim -->\n"
    skipped = [
        ".git/a.md",
        ".venv/a.md",
        "venv/a.md",
        "node_modules/a.md",
        "__pycache__/a.md",
        "build/a.md",
        "dist/a.md",
        ".hidden/a.md",
        "docs/.cache/a.md",
        "docs/build/a.md",
        "pkg/node_modules/deep/a.md",
    ]
    kept = ["top.md", "docs/ok.md", "docs/sub/ok.md", "buildings/ok.md"]
    make_project(tmp_path, "", dict.fromkeys([*skipped, *kept], sentinel))
    assert [o.file for o in found(tmp_path)] == sorted(kept)


def test_bound_files_are_not_subject_to_the_scan_skips(tmp_path: Path) -> None:
    config = bound_config(files='["build/report.md", "docs/**/*.md"]')
    files = {
        "build/report.md": "12 tests\n",
        "docs/a/b.md": "13 tests\n",
        "docs/c.md": "14 tests\n",
    }
    make_project(tmp_path, "", files, config)
    assert [(o.file, o.literal) for o in found(tmp_path)] == [
        ("build/report.md", "12"),
        ("docs/a/b.md", "13"),
        ("docs/c.md", "14"),
    ]


def test_an_unknown_relation_in_a_document_is_a_configuration_error(tmp_path: Path) -> None:
    make_project(tmp_path, "", {"doc.md": "x\n<!-- claim: a ne -->5<!-- /claim -->\n"})
    with pytest.raises(ConfigError, match=r"doc\.md:2: unknown relation 'ne'"):
        found(tmp_path)


def test_a_file_that_is_not_utf8_is_a_configuration_error_naming_it(tmp_path: Path) -> None:
    make_project(tmp_path, "")
    (tmp_path / "bad.md").write_bytes(b"caf\xe9 <!-- claim: x -->5<!-- /claim -->\n")
    with pytest.raises(ConfigError, match=r"bad\.md.*UTF-8"):
        found(tmp_path)


def test_occurrences_are_ordered_by_file_then_line(tmp_path: Path) -> None:
    sentinel = "<!-- claim: x -->5<!-- /claim -->\n"
    make_project(tmp_path, "", {"b.md": sentinel, "a.md": "\n\n" + sentinel, "c/d.md": sentinel})
    assert [(o.file, o.line) for o in found(tmp_path)] == [("a.md", 3), ("b.md", 1), ("c/d.md", 1)]


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("README.md", "**/*.md", True),
        ("a/b/c.md", "**/*.md", True),
        ("a/b/c.txt", "**/*.md", False),
        ("a/b.md", "*.md", False),
        ("b.md", "*.md", True),
        ("docs/x.md", "docs/*.md", True),
        ("docs/a/x.md", "docs/*.md", False),
        ("docs/a/x.md", "docs/**", True),
        ("x.md", "**", True),
        ("a/c.py", "a/**/c.py", True),
        ("a/b/d/c.py", "a/**/c.py", True),
        ("a/b/d/e.py", "a/**/c.py", False),
        ("File1.md", "File?.md", True),
        ("File10.md", "File?.md", False),
        ("README.MD", "*.md", False),
    ],
)
def test_glob_matching(path: str, pattern: str, expected: bool) -> None:
    assert matches(path, pattern) is expected
