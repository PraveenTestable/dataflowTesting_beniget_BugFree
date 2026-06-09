"""
tests/test_analyzer.py - Unit tests for the beniget analyzer.

Run with:  python -m pytest tests/ -v
"""
import pytest
from analyzer.core import (
    CodeAnalyzer, Issue, AnalysisResult,
    DefUseEntry, CoverageMetrics, BUILTIN_NAMES,
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

UNDEFINED_SOURCE = """\
def use_undefined():
    return ghost_variable + 1
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
        assert "x" in names and "y" in names and "z" in names and "unused" in names

    def test_function_detected(self):
        r = self.a.analyze_source(FUNC_SOURCE)
        names = {e.name for e in r.def_use_map.values()}
        assert "add" in names

    def test_parameters_detected(self):
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
        names = {e.name for e in r.def_use_map.values()}
        assert "double" in names

    def test_comprehension_var_detected(self):
        r = self.a.analyze_source(COMPREHENSION_SOURCE)
        assert len(r.def_use_map) > 0

    def test_import_detected(self):
        r = self.a.analyze_source(IMPORT_SOURCE)
        names = {e.name for e in r.def_use_map.values()}
        assert "os" in names

    def test_no_errors(self):
        for src in (BASIC_SOURCE, FUNC_SOURCE, CLASS_SOURCE, LAMBDA_SOURCE):
            r = self.a.analyze_source(src)
            assert not r.errors, "unexpected errors: {}".format(r.errors)


# ---------------------------------------------------------------------------
# Definition-Use Mapping
# ---------------------------------------------------------------------------

class TestDefinitionUseMapping:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_def_use_map_populated(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert len(r.def_use_map) > 0

    def test_used_var_has_correct_use_count(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        # x is used in y = x+2 AND z = x*y  => 2 uses
        x_entries = [e for e in r.def_use_map.values() if e.name == "x"]
        assert x_entries, "x not in def_use_map"
        assert x_entries[0].use_count == 2

    def test_unused_var_has_zero_uses(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        unused = [e for e in r.def_use_map.values() if e.name == "unused"]
        assert unused and unused[0].use_count == 0

    def test_entry_fields_are_correct(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        for entry in r.def_use_map.values():
            assert isinstance(entry, DefUseEntry)
            assert isinstance(entry.name, str) and entry.name
            assert isinstance(entry.lineno, int) and entry.lineno >= 1
            assert entry.kind in DefUseEntry.KINDS

    def test_use_sites_have_correct_linenos(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        x = next(e for e in r.def_use_map.values() if e.name == "x")
        use_lines = [u[0] for u in x.uses]
        assert 2 in use_lines   # y = x + 2  (line 2, 1-indexed)
        assert 3 in use_lines   # z = x * y  (line 3)

    def test_lambda_param_in_map(self):
        r = self.a.analyze_source(LAMBDA_SOURCE)
        entries = {e.name: e for e in r.def_use_map.values()}
        # lambda x: x*2  -- 'x' parameter is used in the body
        assert "x" in entries
        assert entries["x"].use_count >= 1

    def test_get_def_at_helper(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        entry = r.get_def_at(1)   # x = 1 is on line 1
        assert entry is not None
        assert entry.name == "x"

    def test_get_uses_for_helper(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        uses = r.get_uses_for(1)   # uses of x defined on line 1
        assert len(uses) == 2

    def test_function_parameters_mapped(self):
        r = self.a.analyze_source(FUNC_SOURCE)
        entries = {e.name: e for e in r.def_use_map.values()}
        assert entries["a"].use_count >= 1
        assert entries["b"].use_count >= 1


# ---------------------------------------------------------------------------
# Coverage Measurement
# ---------------------------------------------------------------------------

class TestCoverageMeasurement:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_coverage_object_exists(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert isinstance(r.coverage, CoverageMetrics)

    def test_coverage_pct_is_float(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert isinstance(r.coverage_pct, float)

    def test_coverage_pct_in_valid_range(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert 0.0 <= r.coverage_pct <= 100.0

    def test_fully_used_code_has_high_coverage(self):
        # Every var is used
        src = "a = 1\nb = 2\nc = a + b\n"
        r = self.a.analyze_source(src)
        assert r.coverage_pct >= 60.0

    def test_unused_vars_lower_coverage(self):
        # 4 vars defined, only x and y used → 50%
        r = self.a.analyze_source(BASIC_SOURCE)
        assert r.coverage_pct == 50.0

    def test_covered_le_total(self):
        r = self.a.analyze_source(FUNC_SOURCE)
        assert r.coverage.covered_defs <= r.coverage.total_defs

    def test_uncovered_entries_have_zero_uses(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        for entry in r.coverage.uncovered:
            assert entry.use_count == 0
            assert not entry.is_covered

    def test_coverage_pct_alias(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        assert r.coverage_pct == r.coverage.coverage_pct

    def test_empty_source_coverage(self):
        r = self.a.analyze_source("")
        assert r.coverage_pct == 100.0   # no defs → 100%


# ---------------------------------------------------------------------------
# Edge Case Handling
# ---------------------------------------------------------------------------

class TestEdgeCaseHandling:

    def setup_method(self):
        self.a = CodeAnalyzer()

    def test_lambda_analyzed(self):
        r = self.a.analyze_source(LAMBDA_SOURCE)
        assert not r.errors
        assert r.coverage.total_defs > 0

    def test_comprehension_analyzed(self):
        r = self.a.analyze_source(COMPREHENSION_SOURCE)
        assert not r.errors

    def test_comprehension_var_tracked(self):
        r = self.a.analyze_source(COMPREHENSION_SOURCE)
        names = {e.name for e in r.def_use_map.values()}
        assert "x" in names   # comprehension variable

    def test_class_analyzed(self):
        r = self.a.analyze_source(CLASS_SOURCE)
        assert not r.errors

    def test_try_except_analyzed(self):
        r = self.a.analyze_source(TRY_EXCEPT_SOURCE)
        assert not r.errors

    def test_with_statement_analyzed(self):
        r = self.a.analyze_source(WITH_SOURCE)
        assert not r.errors

    def test_nested_function_analyzed(self):
        r = self.a.analyze_source(NESTED_SOURCE)
        assert not r.errors
        names = {e.name for e in r.def_use_map.values()}
        assert "outer" in names and "inner" in names

    def test_import_analyzed(self):
        r = self.a.analyze_source(IMPORT_SOURCE)
        assert not r.errors

    def test_empty_source(self):
        r = self.a.analyze_source("")
        assert not r.errors

    def test_syntax_error_handled_gracefully(self):
        r = self.a.analyze_source("def f(:\n    pass\n")
        assert r.errors
        assert r.total_issues == 0

    def test_dead_code_lineno_correct(self):
        r = self.a.analyze_source(DEAD_CODE_SOURCE)
        dead = [i for i in r.issues if i.kind == "dead_code"]
        assert dead and dead[0].lineno == 3

    def test_nested_scope_params_tracked(self):
        r = self.a.analyze_source(NESTED_SOURCE)
        entries = {e.name: e for e in r.def_use_map.values()}
        assert "x" in entries or "y" in entries


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestUtils:

    def test_count_lines_excludes_blanks_and_comments(self):
        assert count_lines("x = 1\n\n# comment\ny = 2\n") == 2

    def test_count_lines_counts_code(self):
        assert count_lines("a = 1\nb = 2\nc = 3\n") == 3

    def test_parse_severity_strings(self):
        assert parse_severity("low") == 1
        assert parse_severity("medium") == 2
        assert parse_severity("high") == 3

    def test_parse_severity_integers(self):
        assert parse_severity("1") == 1
        assert parse_severity("3") == 3

    def test_parse_severity_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_severity("critical")

    def test_deduplicate_keeps_unique(self):
        a = Issue("unused_var", "x", 10)
        b = Issue("unused_var", "y", 20)
        assert len(deduplicate_issues([a, b])) == 2

    def test_deduplicate_removes_exact_duplicates(self):
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

    def test_format_result_contains_coverage(self):
        r = self.a.analyze_source(BASIC_SOURCE, "test.py")
        output = format_result(r)
        assert "Coverage" in output

    def test_format_coverage_contains_percentage(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        text = format_coverage(r)
        assert "%" in text and "Coverage" in text

    def test_format_def_use_map_shows_definitions(self):
        r = self.a.analyze_source(BASIC_SOURCE)
        text = format_def_use_map(r)
        assert "Def-Use Map" in text
        assert "x" in text

    def test_issues_by_severity_high_only(self):
        r = AnalysisResult("f.py")
        r.add_issue(Issue("undefined", "x", 1, Issue.SEVERITY_HIGH))
        r.add_issue(Issue("unused_var", "y", 2, Issue.SEVERITY_LOW))
        high = r.issues_by_severity(Issue.SEVERITY_HIGH)
        assert any(i.name == "x" for i in high)
        assert all(i.name != "y" for i in high)

    def test_issues_by_severity_medium_and_above(self):
        r = AnalysisResult("f.py")
        r.add_issue(Issue("shadowed", "a", 1, Issue.SEVERITY_LOW))
        r.add_issue(Issue("unused_var", "b", 2, Issue.SEVERITY_MEDIUM))
        r.add_issue(Issue("undefined", "c", 3, Issue.SEVERITY_HIGH))
        med_plus = r.issues_by_severity(Issue.SEVERITY_MEDIUM)
        assert "a" not in [i.name for i in med_plus]
        assert "b" in [i.name for i in med_plus]
        assert "c" in [i.name for i in med_plus]
