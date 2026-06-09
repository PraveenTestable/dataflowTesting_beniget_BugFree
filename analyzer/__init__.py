"""beniget-based static analyzer for Python 3.6+ code."""
from .core import (
    CodeAnalyzer, AnalysisResult, Issue,
    DefUseEntry, CoverageMetrics,
)
from .report import (
    format_result, format_summary_table, format_coverage,
    format_def_use_map, write_report,
)
from .utils import collect_python_files, count_lines, deduplicate_issues

__version__ = "0.2.0"
__all__ = [
    "CodeAnalyzer", "AnalysisResult", "Issue",
    "DefUseEntry", "CoverageMetrics",
    "format_result", "format_summary_table", "format_coverage",
    "format_def_use_map", "write_report",
    "collect_python_files", "count_lines", "deduplicate_issues",
]
