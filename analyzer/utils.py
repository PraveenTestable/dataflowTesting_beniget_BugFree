"""
utils.py - Helper utilities for the beniget analyzer.

Supports Python 3.6+.
"""
import os
import fnmatch
from typing import Iterable, List, Optional


def collect_python_files(root, exclude_patterns=None):
    # type: (str, Optional[List[str]]) -> List[str]
    """Recursively collect *.py files under *root*, skipping excluded patterns."""
    if exclude_patterns is None:
        exclude_patterns = []

    collected = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in ("venv", "env", "__pycache__")
        ]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            if _matches_any(filename, exclude_patterns):
                continue
            collected.append(os.path.join(dirpath, filename))

    return sorted(collected)


def _matches_any(name, patterns):
    # type: (str, Iterable[str]) -> bool
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def read_source(filepath, encoding="utf-8"):
    # type: (str, str) -> Optional[str]
    """Read *filepath* as text; returns None on IOError."""
    try:
        with open(filepath, "r", encoding=encoding) as fh:
            return fh.read()
    except IOError:
        return None


def count_lines(source):
    # type: (str) -> int
    """Return count of non-empty, non-comment lines in *source*."""
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def truncate_path(path, max_len=50):
    # type: (str, int) -> str
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


def parse_severity(value):
    # type: (str) -> int
    """Parse 'low'/'medium'/'high' or '1'/'2'/'3'. Raises ValueError on bad input."""
    mapping = {"low": 1, "medium": 2, "high": 3}
    v = value.strip().lower()
    if v in mapping:
        return mapping[v]
    try:
        n = int(v)
        if n < 1 or n > 3:
            raise ValueError
        return n
    except ValueError:
        raise ValueError(
            "Invalid severity {!r}. Expected low/medium/high or 1/2/3.".format(value)
        )


def deduplicate_issues(issues):
    # type: (list) -> list
    """Remove duplicate issues by (kind, lineno, name); preserve first-occurrence order."""
    seen = set()
    result = []
    for issue in issues:
        key = (issue.kind, issue.lineno, issue.name)
        if key not in seen:
            result.append(issue)
        seen.add(key)
    return result
