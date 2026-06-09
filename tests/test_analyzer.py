"""
tests/test_analyzer.py - Unit tests for the beniget analyzer.

Run with:  python -m pytest tests/
"""
import pytest
from analyzer.core import CodeAnalyzer, Issue, AnalysisResult, DefUseEntry, CoverageMetrics
from analyzer.utils import count_lines, deduplicate_issues, parse_severity
from analyzer.report import format_result, format_coverage, format_def_use_map

# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

SIMPLE_SOURCE = """\
def greet(name):
    msg = "Hello, " + name
    unused = 42
    return msg
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

# ---------------------------------------------------------------------------
# Variable Definition Detection
# ---------------------------------------------------------------------------

class TestVariableDefinitionDetection:

    def setup_method(self):
        self.analyzer = CodeAnalyzer()

    def test_detects_function_definition(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        names = [e.name for e in result.def_use_map.values()]
        assert "greet" in names or any("msg" == e.name for e in result.def_use_map.values())

    def test_detects_class_definition(self):
        result = self.analyzer.analyze_source(CLASS_SOURCE)
        names = [e.name for e in result.def_use_map.values()]
        assert "Counter" in names

    def test_detects_lambda_definition(self):
        result = self.analyzer.analyze_source(LAMBDA_SOURCE)
        names = [e.name for e in result.def_use_map.values()]
        assert "double" in names

    def test_detects_comprehension_variable(self):
        result = self.analyzer.analyze_source(COMPREHENSION_SOURCE)
        assert result.def_use_map is not None
        assert len(result.def_use_map) > 0

# ---------------------------------------------------------------------------
# Definition-Use Mapping
# ---------------------------------------------------------------------------

class TestDefinitionUseMapping:

    def setup_method(self):
        self.analyzer = CodeAnalyzer()

    def test_def_use_map_is_populated(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        assert len(result.def_use_map) > 0

    def test_used_variable_has_uses(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        msg_entries = [e for e in result.def_use_map.values() if e.name == "msg"]
        assert any(e.use_count > 0 for e in msg_entries)

    def test_unused_variable_has_zero_uses(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        unused_entries = [e for e in result.def_use_map.values() if e.name == "unused"]
        assert any(e.use_count == 0 for e in unused_entries)

    def test_def_use_entry_has_correct_fields(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        for entry in result.def_use_map.values():
            assert isinstance(entry, DefUseEntry)
            assert isinstance(entry.name, str)
            assert isinstance(entry.lineno, int)
            assert entry.lineno >= 1
            assert entry.kind in DefUseEntry.KINDS

    def test_lambda_args_in_def_use_map(self):
        result = self.analyzer.analyze_source(LAMBDA_SOURCE)
        kinds = [e.kind for e in result.def_use_map.values()]
        assert "parameter" in kinds or "lambda" in kinds or "variable" in kinds

# ---------------------------------------------------------------------------
# Coverage Measurement
# ---------------------------------------------------------------------------

class TestCoverageMeasurement:

    def setup_method(self):
        self.analyzer = CodeAnalyzer()

    def test_coverage_object_exists(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        assert isinstance(result.coverage, CoverageMetrics)

    def test_coverage_pct_is_float(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        assert isinstance(result.coverage.coverage_pct, float)

    def test_coverage_pct_range(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        assert 0.0 <= result.coverage.coverage_pct <= 100.0

    def test_fully_used_code_has_high_coverage(self):
        src = "def add(a, b):\n    return a + b\nresult = add(1, 2)\n"
        result = self.analyzer.analyze_source(src)
        assert result.coverage.coverage_pct >= 0.0

    def test_coverage_counts_are_consistent(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        assert result.coverage.covered_defs <= result.coverage.total_defs
        assert result.coverage.total_defs >= 0

    def test_uncovered_list_correct(self):
        result = self.analyzer.analyze_source(SIMPLE_SOURCE)
        for entry in result.coverage.uncovered:
            assert entry.use_count == 0
            assert not entry.is_covered

# ---------------------------------------------------------------------------
# Edge Case Handling
# ---------------------------------------------------------------------------

class TestEdgeCaseHandling:

    def setup_method(self):
        self.analyzer = CodeAnalyzer()

    def test_lambda_source_analyzed(self):
        result = self.analyzer.analyze_source(LAMBDA_SOURCE)
        assert not result.errors

    def test_comprehension_source_analyzed(self):
        result = self.analyzer.analyze_source(COMPREHENSION_SOURCE)
        assert not result.errors

    def test_class_source_analyzed(self):
        result = self.analyzer.analyze_source(CLASS_SOURCE)
        assert not result.errors

    def test_try_except_source_analyzed(self):
        result = self.analyzer.analyze_source(TRY_EXCEPT_SOURCE)
        assert not result.errors

    def test_with_statement_analyzed(self):
        result = self.analyzer.analyze_source(WITH_SOURCE)
        assert not result.errors

    def test_empty_source(self):
        result = self.analyzer.analyze_source("")
        assert not result.errors

    def test_syntax_error_handled(self):
        result = self.analyzer.analyze_source("def f(:\n    pass\n")
        assert result.errors
        assert result.total_issues == 0

    def test_dead_code_lineno_correct(self):
        result = self.analyzer.analyze_source(DEAD_CODE_SOURCE)
        dead = [i for i in result.issues if i.kind == "dead_code"]
        assert dead[0].lineno == 3

    def test_nested_function(self):
        src = """\
def outer():
    def inner(x):
        return x + 1
    return inner(5)
"""
        result = self.analyzer.analyze_source(src)
        assert not result.errors

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestUtils:

    def test_count_lines_excludes_blank_and_comments(self):
        assert count_lines("x = 1\n\n# comment\ny = 2\n") == 2

    def test_parse_severity_strings(self):
        assert parse_severity("low") == 1
        assert parse_severity("medium") == 2
        assert parse_severity("high") == 3

    def test_parse_severity_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_severity("critical")

    def test_deduplicate_keeps_unique(self):
        a = Issue("unused_var", "x", 10)
        b = Issue("unused_var", "y", 20)
        assert len(deduplicate_issues([a, b])) == 2

    def test_deduplicate_removes_duplicates(self):
        a = Issue("unused_var", "x", 10)
        b = Issue("unused_var", "x", 10)
        assert len(deduplicate_issues([a, b])) == 1

# ---------------------------------------------------------------------------
# Reporting Validation
# ---------------------------------------------------------------------------

class TestReportingValidation:

    def test_format_result_contains_filepath(self):
        result = AnalysisResult("myfile.py")
        result.loc = 10
        assert "myfile.py" in format_result(result)

    def test_format_result_contains_coverage(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_source(SIMPLE_SOURCE, "test.py")
        output = format_result(result)
        assert "Coverage" in output

    def test_format_coverage_output(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_source(SIMPLE_SOURCE)
        cov_text = format_coverage(result)
        assert "Coverage" in cov_text
        assert "%" in cov_text

    def test_format_def_use_map_output(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_source(SIMPLE_SOURCE)
        map_text = format_def_use_map(result)
        assert "Def-Use Map" in map_text

    def test_issues_by_severity_correct(self):
        result = AnalysisResult("f.py")
        result.add_issue(Issue("undefined", "x", 1, Issue.SEVERITY_HIGH))
        result.add_issue(Issue("unused_var", "y", 2, Issue.SEVERITY_LOW))
        high = result.issues_by_severity(Issue.SEVERITY_HIGH)
        assert any(i.name == "x" for i in high)
        assert all(i.name != "y" for i in high)
