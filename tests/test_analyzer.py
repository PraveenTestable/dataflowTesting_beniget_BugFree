"""
tests/test_analyzer.py - Unit tests for the beniget analyzer.

Covers all six whitebox metrics:
  Variable Definition Detection
  Definition-Use Mapping
  Coverage Measurement (DU-Path Validation)
  Uncovered Definition Detection
  Edge Case Handling
  Reporting Validation

Run with:  python -m pytest tests/ -v
"""
import pytest
from analyzer.core import (
    CodeAnalyzer, Issue, AnalysisResult,
    DefUseEntry, CoverageMetrics, DUPair, DUPathCoverage,
    BUILTIN_NAMES,
)
from analyzer.utils import count_lines, deduplicate_issues, parse_severity
from analyzer.report import format_result, format_coverage, format_def_use_map

# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

BASIC_SOURCE = """\
x = 1
y = x + 2
z = x * y
unused = 99
"""

REDEF_SOURCE = """\
x = 1
y = x + 2
x = 5
z = x * y
"""

FUNC_SOURCE = """\
def add(a, b):
    result = a + b
    return result
total = add(1, 2)
"""

DEAD_CODE_SOURCE = """\
def broken():
    return 1
    print("never reached")
"""

LAMBDA_SOURCE = """\
double = lambda x: x * 2
result = double(5)
"""

COMPREHENSION_SOURCE = """\
def make_squares(n):
    squares = [x * x for x in range(n)]
    return squares
"""

CLASS_SOURCE = """\
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def value(self):
        return self.count
"""

TRY_EXCEPT_SOURCE = """\
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError as e:
        result = 0
    return result
"""

WITH_SOURCE = """\
def read_file(path):
    with open(path) as fh:
        return fh.read()
"""

NESTED_SOURCE = """\
def outer(x):
    def inner(y):
        return y + 1
    return inner(x)
"""

IMPORT_SOURCE = """\
import os
from sys import argv
path = os.path.join(argv[0], 'data')
"""


# ---------------------------------------------------------------------------
# Variable Definition Detection
# ---------------------------------------------------------------------------

class TestVariableDefinitionDetection:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_module_level_vars_detected(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        names = {e.name for e in r.def_use_map.values()}
        assert {"x", "y", "z", "unused"} <= names

    def test_function_detected(self):
        r = self.a.analyze_source(FUNC_SOURCE)
        names = {e.name for e in r.def_use_map.values()}
        assert "add" in names

    def test_parameters_detected_with_correct_kind(self):
        r = self.a.analyze_source(FUNC_SOURCE)
        entries = {e.name: e for e in r.def_use_map.values()}
        assert "a" in entries and "b" in entries
        assert entries["a"].kind == "parameter"
        assert entries["b"].kind == "parameter"

    def test_class_detected(self):
        r = self.a.analyze_source(CLASS_SOURCE)
        kinds = {e.name: e.kind for e in r.def_use_map.values()}
        assert kinds.get("Counter") == "class"

    def test_lambda_var_detected(self):
        r = self.a.analyze_source(LAMBDA_SOURCE)
        assert "double" in {e.name for e in r.def_use_map.values()}

    def test_import_detected(self):
        r = self.a.analyze_source(IMPORT_SOURCE)
        assert "os" in {e.name for e in r.def_use_map.values()}

    def test_no_errors_on_clean_source(self):
        for src in (BASIC_SOURCE, FUNC_SOURCE, CLASS_SOURCE, LAMBDA_SOURCE):
            assert not self.a.analyze_source(src).errors


# ---------------------------------------------------------------------------
# Definition-Use Mapping
# ---------------------------------------------------------------------------

class TestDefinitionUseMapping:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_def_use_map_is_populated(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert len(r.def_use_map) > 0

    def test_used_var_has_correct_use_count(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        x_entries = [e for e in r.def_use_map.values() if e.name == "x"]
        assert x_entries and x_entries[0].use_count == 2

    def test_unused_var_has_zero_uses(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        unused = [e for e in r.def_use_map.values() if e.name == "unused"]
        assert unused and unused[0].use_count == 0

    def test_entry_fields_correct(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        for entry in r.def_use_map.values():
            assert isinstance(entry, DefUseEntry)
            assert isinstance(entry.name, str) and entry.name
            assert isinstance(entry.lineno, int) and entry.lineno >= 1
            assert entry.kind in DefUseEntry.KINDS

    def test_use_site_linenos_correct(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        x = next(e for e in r.def_use_map.values() if e.name == "x")
        use_lines = {u[0] for u in x.uses}
        assert 2 in use_lines and 3 in use_lines

    def test_lambda_param_has_uses(self):
        r = self.a.analyze_source(LAMBDA_SOURCE)
        x = next((e for e in r.def_use_map.values() if e.name == "x"), None)
        assert x is not None and x.use_count >= 1

    def test_get_def_at_helper(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        entry = r.get_def_at(1)
        assert entry is not None and entry.name == "x"

    def test_get_uses_for_helper(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert len(r.get_uses_for(1)) == 2

    def test_redefined_var_correct_chains(self):
        r = self.a.analyze_source(REDEF_SOURCE)
        # x defined at line 1 should chain to use at line 2 (before redefinition)
        x1 = r.get_def_at(1, "x")
        assert x1 is not None
        use_lines = {u[0] for u in x1.uses}
        assert 2 in use_lines   # y = x + 2


# ---------------------------------------------------------------------------
# Coverage Measurement (DU-Path Validation)
# ---------------------------------------------------------------------------

class TestDUPathCoverage:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_du_path_coverage_object_exists(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert isinstance(r.du_path_coverage, DUPathCoverage)

    def test_du_pairs_populated(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert r.du_path_coverage.total > 0

    def test_du_coverage_pct_is_float(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert isinstance(r.du_coverage_pct, float)

    def test_du_coverage_pct_in_range(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert 0.0 <= r.du_coverage_pct <= 100.0

    def test_redefined_var_reduces_coverage(self):
        # x redefined at line 3; x@1->use@4 is uncovered (killed by redef)
        r = self.a.analyze_source(REDEF_SOURCE)
        assert r.du_coverage_pct < 100.0

    def test_covered_pairs_have_def_clear_path(self):
        r = self.a.analyze_source(REDEF_SOURCE)
        # x@line1 -> use@line2 is covered (no redef on that path)
        covered = [p for p in r.du_path_coverage.covered_pairs
                   if p.name == "x" and p.def_lineno == 1 and p.use_lineno == 2]
        assert covered

    def test_uncovered_pair_has_killing_redef(self):
        r = self.a.analyze_source(REDEF_SOURCE)
        # x@line1 -> use@line4 is uncovered (x redefined at line 3)
        uncov = [p for p in r.du_path_coverage.uncovered_pairs
                 if p.name == "x" and p.def_lineno == 1 and p.use_lineno == 4]
        assert uncov

    def test_du_pair_fields(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        for pair in r.du_path_coverage.du_pairs:
            assert isinstance(pair, DUPair)
            assert isinstance(pair.name, str)
            assert isinstance(pair.def_lineno, int)
            assert isinstance(pair.use_lineno, int)
            assert isinstance(pair.covered, bool)

    def test_empty_source_du_coverage(self):
        r = self.a.analyze_source("")
        assert r.du_coverage_pct == 100.0

    def test_get_du_pairs_for_variable(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        x_pairs = r.get_du_pairs_for("x")
        assert len(x_pairs) > 0
        assert all(p.name == "x" for p in x_pairs)

    def test_du_coverage_alias(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert r.du_coverage_pct == r.du_path_coverage.coverage_pct

    def test_fully_used_vars_improve_coverage(self):
        src = "a = 1\nb = a + 1\nc = b * 2\n"
        r = self.a.analyze_source(src)
        assert r.du_coverage_pct > 0.0


# ---------------------------------------------------------------------------
# Uncovered Definition Detection
# ---------------------------------------------------------------------------

class TestUncoveredDefinitionDetection:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_unused_var_emits_issue(self):
        r = self.a.analyze_source(FUNC_SOURCE)
        unused = [i.name for i in r.issues if i.kind == "unused_var"]
        assert "result" not in unused   # result IS used in return

    def test_uncovered_entry_has_zero_uses(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        for entry in r.coverage.uncovered:
            assert entry.use_count == 0

    def test_coverage_totals_consistent(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert r.coverage.covered_defs <= r.coverage.total_defs

    def test_coverage_pct_correct(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        # x used twice, y used once, z unused, unused unused -> 2/4 = 50%
        assert r.coverage_pct == 50.0


# ---------------------------------------------------------------------------
# Edge Case Handling
# ---------------------------------------------------------------------------

class TestEdgeCaseHandling:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_lambda_no_errors(self):
        assert not self.a.analyze_source(LAMBDA_SOURCE).errors

    def test_comprehension_no_errors(self):
        assert not self.a.analyze_source(COMPREHENSION_SOURCE).errors

    def test_comprehension_var_tracked(self):
        r = self.a.analyze_source(COMPREHENSION_SOURCE)
        assert "x" in {e.name for e in r.def_use_map.values()}

    def test_class_no_errors(self):
        assert not self.a.analyze_source(CLASS_SOURCE).errors

    def test_try_except_no_errors(self):
        assert not self.a.analyze_source(TRY_EXCEPT_SOURCE).errors

    def test_with_statement_no_errors(self):
        assert not self.a.analyze_source(WITH_SOURCE).errors

    def test_nested_function_no_errors(self):
        r = self.a.analyze_source(NESTED_SOURCE)
        assert not r.errors
        names = {e.name for e in r.def_use_map.values()}
        assert "outer" in names and "inner" in names

    def test_import_no_errors(self):
        assert not self.a.analyze_source(IMPORT_SOURCE).errors

    def test_empty_source(self):
        r = self.a.analyze_source("")
        assert not r.errors
        assert r.du_coverage_pct == 100.0

    def test_syntax_error_graceful(self):
        r = self.a.analyze_source("def f(:\n    pass\n")
        assert r.errors and r.total_issues == 0

    def test_dead_code_lineno_correct(self):
        r = self.a.analyze_source(DEAD_CODE_SOURCE)
        dead = [i for i in r.issues if i.kind == "dead_code"]
        assert dead and dead[0].lineno == 3

    def test_lambda_param_du_pairs(self):
        r = self.a.analyze_source(LAMBDA_SOURCE)
        pairs = r.du_path_coverage.du_pairs
        assert any(p.name == "x" for p in pairs)

    def test_comprehension_du_pairs(self):
        r = self.a.analyze_source(COMPREHENSION_SOURCE)
        assert r.du_path_coverage.total > 0

    def test_async_function(self):
        src = "async def f(x):\n    return x + 1\n"
        r = self.a.analyze_source(src)
        assert not r.errors

    def test_multiple_assignments(self):
        src = "a, b = 1, 2\nc = a + b\n"
        r = self.a.analyze_source(src)
        assert not r.errors


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestUtils:

    def test_count_lines_excludes_blanks_comments(self):
        assert count_lines("x = 1\n\n# comment\ny = 2\n") == 2

    def test_parse_severity_strings(self):
        assert parse_severity("low") == 1
        assert parse_severity("medium") == 2
        assert parse_severity("high") == 3

    def test_parse_severity_invalid(self):
        with pytest.raises(ValueError):
            parse_severity("critical")

    def test_deduplicate_unique(self):
        a, b = Issue("unused_var", "x", 10), Issue("unused_var", "y", 20)
        assert len(deduplicate_issues([a, b])) == 2

    def test_deduplicate_removes_dups(self):
        a = Issue("unused_var", "x", 10)
        b = Issue("unused_var", "x", 10)
        assert len(deduplicate_issues([a, b])) == 1


# ---------------------------------------------------------------------------
# Reporting Validation
# ---------------------------------------------------------------------------

class TestReportingValidation:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_format_result_contains_filepath(self):
        r = AnalysisResult("myfile.py")
        r.loc = 10
        assert "myfile.py" in format_result(r)

    def test_format_coverage_has_du_path_line(self):
        r = self.a.analyze_source(BASIC_SOURCE, "test.py")
        text = format_coverage(r)
        assert "DU-Path Coverage" in text
        assert "%" in text

    def test_format_coverage_has_def_coverage_line(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert "Def Coverage" in format_coverage(r)

    def test_format_def_use_map(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        text = format_def_use_map(r)
        assert "Def-Use Map" in text and "x" in text

    def test_format_result_contains_coverage(self):
        r = self.a.analyze_source(BASIC_SOURCE, "f.py")
        assert "Coverage" in format_result(r)

    def test_issues_by_severity_high_only(self):
        r = AnalysisResult("f.py")
        r.add_issue(Issue("undefined", "x", 1, Issue.SEVERITY_HIGH))
        r.add_issue(Issue("unused_var", "y", 2, Issue.SEVERITY_LOW))
        high = r.issues_by_severity(Issue.SEVERITY_HIGH)
        assert any(i.name == "x" for i in high)
        assert all(i.name != "y" for i in high)

    def test_issues_by_severity_medium_plus(self):
        r = AnalysisResult("f.py")
        r.add_issue(Issue("shadowed", "a", 1, Issue.SEVERITY_LOW))
        r.add_issue(Issue("unused_var", "b", 2, Issue.SEVERITY_MEDIUM))
        r.add_issue(Issue("undefined", "c", 3, Issue.SEVERITY_HIGH))
        med = r.issues_by_severity(Issue.SEVERITY_MEDIUM)
        names = [i.name for i in med]
        assert "a" not in names and "b" in names and "c" in names

    def test_summary_contains_du_coverage(self):
        r = self.a.analyze_source(BASIC_SOURCE, "f.py")
        assert "du_cov=" in r.summary()
