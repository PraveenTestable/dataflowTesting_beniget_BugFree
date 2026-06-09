"""
cli.py - Command-line interface for the beniget static analyzer.

Usage:
    python -m analyzer.cli [OPTIONS] PATH [PATH ...]

Supports Python 3.6+.
"""
import argparse
import sys
import os

from .core import CodeAnalyzer, Issue
from .report import format_result, format_summary_table, write_report
from .utils import collect_python_files, parse_severity, truncate_path


def build_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        prog="beniget-analyzer",
        description="Static analysis of Python code using beniget def-use chains.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Python file(s) or director(ies) to analyze.",
    )
    parser.add_argument(
        "--severity", "-s",
        default="low",
        metavar="LEVEL",
        help="Minimum severity to report: low | medium | high  (default: low)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Write report to FILE instead of stdout.",
    )
    parser.add_argument(
        "--exclude", "-e",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Exclude files matching PATTERN (fnmatch). May be repeated.",
    )
    parser.add_argument(
        "--ignore-private",
        action="store_true",
        default=False,
        help="Skip issues for names starting with an underscore.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show all issues in detail (default: summary table only).",
    )
    return parser


def resolve_targets(paths, exclude_patterns):
    # type: (list, list) -> list
    """Expand paths to a flat list of .py files."""
    targets = []
    for path in paths:
        if os.path.isfile(path):
            targets.append(os.path.abspath(path))
        elif os.path.isdir(path):
            targets.extend(collect_python_files(path, exclude_patterns))
        else:
            print("warning: path not found: {}".format(path), file=sys.stderr)
    return targets


def main(argv=None):
    # type: (object) -> int
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        min_severity = parse_severity(args.severity)
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    targets = resolve_targets(args.paths, args.exclude)
    if not targets:
        print("No Python files found.", file=sys.stderr)
        return 1

    analyzer = CodeAnalyzer(ignore_private=args.ignore_private)
    results = []

    for filepath in targets:
        if args.verbose:
            print("Analyzing {} ...".format(truncate_path(filepath)))
        result = analyzer.analyze_file(filepath)
        results.append(result)

    # -- output ---------------------------------------------------------------
    if args.output:
        write_report(results, args.output, min_severity, verbose=args.verbose)
        print("Report written to {}".format(args.output))
    else:
        print(format_summary_table(results, min_severity))
        if args.verbose:
            print()
            for result in results:
                print(format_result(result, min_severity, verbose=True))

    # exit 1 if any HIGH issues found
    any_high = any(
        i.severity == Issue.SEVERITY_HIGH
        for r in results
        for i in r.issues
    )
    return 1 if any_high else 0


if __name__ == "__main__":
    sys.exit(main())
