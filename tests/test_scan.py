from pathlib import Path

import pytest

from helpers import DEFAULT_CONFIG, make_project, run
from numeric_claim.config import load_config
from numeric_claim.errors import ConfigError
from numeric_claim.globs import matches
from numeric_claim.scan import Occurrence, collect

REPOSITORY = Path(__file__).resolve().parents[1]
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


def marker(name: str) -> str:
    return f"<!-- claim: {name} -->5<!-- /claim -->"


def in_markdown(root: Path, *rows: str) -> list[tuple[int, str]]:
    make_project(root, "", {"doc.md": "".join(row + "\n" for row in rows)})
    return [(occurrence.line, occurrence.claim) for occurrence in found(root)]


def test_fences_of_backticks_and_of_tildes_hide_their_markers(tmp_path: Path) -> None:
    rows = (
        marker("a"),
        "```python",
        marker("b"),
        "\\claim{c}{5}",
        "```",
        "~~~",
        marker("d"),
        "```",
        marker("e"),
        "~~~",
        marker("f") + " \\claim{g}{5}",
    )
    assert in_markdown(tmp_path, *rows) == [(1, "a"), (11, "f"), (11, "g")]


def test_a_fence_closes_only_with_at_least_as_many_of_its_characters_and_nothing_else(
    tmp_path: Path,
) -> None:
    rows = (
        "````",
        "```",
        marker("a"),
        "```` more",
        marker("b"),
        "``````   ",
        marker("c"),
    )
    assert in_markdown(tmp_path, *rows) == [(7, "c")]


def test_a_fence_may_be_indented_by_up_to_three_spaces(tmp_path: Path) -> None:
    rows = (
        "   ```",
        marker("a"),
        "   ```",
        marker("b"),
        "    ```",
        marker("c"),
    )
    assert in_markdown(tmp_path, *rows) == [(4, "b"), (6, "c")]


def test_a_fence_that_never_closes_runs_to_the_end_of_the_file(tmp_path: Path) -> None:
    rows = (
        marker("a"),
        "~~~~",
        marker("b"),
        "~~~",
        "",
        marker("c"),
    )
    assert in_markdown(tmp_path, *rows) == [(1, "a")]


def test_inline_code_spans_of_one_and_of_two_backticks_hide_their_markers(tmp_path: Path) -> None:
    rows = (
        f"`{marker('a')}` and {marker('b')}",
        f"``{marker('c')} with ` inside`` and {marker('d')}",
        "`\\claim{e}{5}` and \\claim{f}{5}",
    )
    assert in_markdown(tmp_path, *rows) == [(1, "b"), (2, "d"), (3, "f")]


def test_a_backtick_run_without_a_partner_on_its_line_is_ordinary_text(tmp_path: Path) -> None:
    rows = (
        f"one ` tick {marker('a')}",
        f"`` two {marker('b')} ` one `",
        f"`{marker('c')}",
        f"{marker('d')}`",
    )
    assert in_markdown(tmp_path, *rows) == [(1, "a"), (2, "b"), (3, "c"), (4, "d")]


def test_a_code_span_closes_only_with_a_run_of_exactly_as_many_backticks(tmp_path: Path) -> None:
    rows = (
        f"`{marker('a')}``",
        f"``{marker('b')}``",
    )
    assert in_markdown(tmp_path, *rows) == [(1, "a")]


def test_line_numbers_and_markers_after_a_fence_are_unchanged(tmp_path: Path) -> None:
    text = f"intro\n```\ncode\n\nmore code\n```\n\nValue {marker('a')} here, `code`.\n"
    make_project(tmp_path, "", {"doc.md": text})
    assert found(tmp_path) == [Occurrence("doc.md", 8, "a", "eq", "5", "sentinel")]


def test_latex_files_are_scanned_as_before(tmp_path: Path) -> None:
    tex = "``quoted'' \\claim{a}{5} ``again''\n```\n\\claim{b}{5}\n```\n"
    make_project(tmp_path, "", {"paper.tex": tex})
    assert [(o.line, o.claim) for o in found(tmp_path)] == [(1, "a"), (3, "b")]


def test_bindings_still_read_code_in_markdown(tmp_path: Path) -> None:
    readme = "```\n12 tests\n```\nand `13 tests`\n"
    make_project(tmp_path, "", {"README.md": readme}, bound_config())
    assert [(o.line, o.literal) for o in found(tmp_path)] == [(2, "12"), (4, "13")]


def test_excluded_files_are_not_scanned(tmp_path: Path) -> None:
    names = ["top.md", "docs/a.md", "docs/sub/b.md", "notes/c.md", "notes/d.md", "paper.tex"]
    config = DEFAULT_CONFIG + 'exclude = ["docs/**", "*.tex", "notes/d.md"]\n'
    make_project(tmp_path, "", dict.fromkeys(names, marker("x") + "\n"), config)
    assert [o.file for o in found(tmp_path)] == ["notes/c.md", "top.md"]


def test_exclude_does_not_apply_to_bindings(tmp_path: Path) -> None:
    binding = BIND.format(files='["README.md"]', pattern=r"(\d+) tests\b", claim='"n"', extra="")
    config = DEFAULT_CONFIG + 'exclude = ["*.md"]\n' + binding
    make_project(tmp_path, "", {"README.md": f"12 tests\n{marker('x')}\n"}, config)
    assert [(o.line, o.origin) for o in found(tmp_path)] == [(1, "pattern")]


def test_nothing_is_excluded_by_default(tmp_path: Path) -> None:
    make_project(tmp_path, "")
    assert load_config(tmp_path / "numeric-claim.toml").exclude == ()


@pytest.mark.parametrize("value", ['"docs/**"', "[1]", '["docs/**", 2]', "true"])
def test_an_exclude_that_is_not_a_list_of_strings_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    make_project(tmp_path, "", config=DEFAULT_CONFIG + f"exclude = {value}\n")
    assert run(tmp_path, "check") == 2
    assert "'exclude' must be a list of strings" in capsys.readouterr().err


def test_this_repositorys_design_decisions_quote_markers_only_in_code(tmp_path: Path) -> None:
    text = (REPOSITORY / "docs" / "decisions.md").read_text(encoding="utf-8")
    assert "`<!-- claim: NAME -->VALUE<!-- /claim -->`" in text
    assert "`\\claim{NAME}{VALUE}`" in text
    make_project(tmp_path, "", {"docs/decisions.md": text})
    assert found(tmp_path) == []
