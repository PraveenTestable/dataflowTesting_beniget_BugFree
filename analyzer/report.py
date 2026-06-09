"""
report.py - Formats AnalysisResult objects into human-readable output.

Supports Python 3.6+.
"""
from typing import List
from .core import AnalysisResult, Issue


SEVERITY_LABELS = {
    Issue.SEVERITY_LOW: "LOW",
    Issue.SEVERITY_MEDIUM: "MEDIUM",
    Issue.SEVERITY_HIGH: "HIGH",
}

KIND_LABELS = {
    "unused_var": "Unused variable",
    "undefined":  "Undefined name",
    "dead_code":  "Dead code",
    "shadowed":   "Shadowed name",
}


def format_issue(issue, show_severity=True):
    # type: (Issue, bool) -> str
    sev = "[{}] ".format(SEVERITY_LABELS.get(issue.severity, "?")) if show_severity else ""
    kind = KIND_LABELS.get(issue.kind, issue.kind)
    return "  line {:4d}  {}{}  {}".format(issue.lineno, sev, kind, issue.message)


def format_coverage(result):
    # type: (AnalysisResult) -> str
    """Render both definition-level and DU-path coverage."""
    cov = result.coverage
    du = result.du_path_coverage
    lines = [
        "Def Coverage    : {:.1f}%  ({}/{} definitions used)".format(
            cov.coverage_pct, cov.covered_defs, cov.total_defs
        ),
        "DU-Path Coverage: {:.1f}%  ({}/{} DU-pairs covered)".format(
            du.coverage_pct, du.covered, du.total
        ),
    ]
    if cov.uncovered:
        lines.append("  Uncovered definitions:")
        for entry in sorted(cov.uncovered, key=lambda e: e.lineno):
            lines.append("    line {:4d}  [{}] {}".format(
                entry.lineno, entry.kind, entry.name
            ))
    if du.uncovered_pairs:
        lines.append("  Uncovered DU-pairs:")
        for pair in sorted(du.uncovered_pairs, key=lambda p: (p.name, p.def_lineno)):
            lines.append("    {!r}: def@{} -> use@{}".format(
                pair.name, pair.def_lineno, pair.use_lineno
            ))
    return "\n".join(lines)


def format_def_use_map(result, max_entries=20):
    # type: (AnalysisResult, int) -> str
    """Render a summary of the def-use map (first *max_entries* entries)."""
    entries = sorted(result.def_use_map.values(), key=lambda e: e.lineno)
    lines = ["Def-Use Map ({} definitions):".format(len(entries))]
    for entry in entries[:max_entries]:
        use_info = "{} use(s)".format(entry.use_count)
        if entry.uses:
            use_lines = ", ".join(str(u[0]) for u in entry.uses[:5])
            use_info += " at line(s) " + use_lines
        lines.append("  line {:4d}  [{}] {}  ->  {}".format(
            entry.lineno, entry.kind, entry.name, use_info
        ))
    if len(entries) > max_entries:
        lines.append("  ... and {} more".format(len(entries) - max_entries))
    return "\n".join(lines)


def format_result(result, min_severity=Issue.SEVERITY_LOW, verbose=False):
    # type: (AnalysisResult, int, bool) -> str
    lines = ["=" * 60,
             "File : {}".format(result.filepath),
             "LoC  : {}".format(result.loc)]

    if result.errors:
        lines.append("ERRORS:")
        for err in result.errors:
            lines.append("  ! " + err)
        lines.append("=" * 60)
        return "\n".join(lines)

    lines.append(format_coverage(result))

    if verbose:
        lines.append("")
        lines.append(format_def_use_map(result))

    filtered = result.issues_by_severity(min_severity)
    lines.append("\nIssues: {} (showing {})".format(result.total_issues, len(filtered)))

    if not filtered:
        lines.append("  (no issues at or above severity {})".format(
            SEVERITY_LABELS.get(min_severity, str(min_severity))
        ))
    else:
        by_kind = {}
        for issue in filtered:
            by_kind.setdefault(issue.kind, []).append(issue)
        for kind in sorted(by_kind.keys()):
            group = sorted(by_kind[kind], key=lambda i: i.lineno)
            lines.append("\n  -- {} ({}) --".format(KIND_LABELS.get(kind, kind), len(group)))
            for issue in group:
                lines.append(format_issue(issue))

    lines.append("=" * 60)
    return "\n".join(lines)


def format_summary_table(results, min_severity=Issue.SEVERITY_LOW):
    # type: (List[AnalysisResult], int) -> str
    header = "{:<40} {:>6} {:>8} {:>8} {:>11} {:>10} {:>9}".format(
        "File", "LoC", "cov%", "unused", "undefined", "dead_code", "shadowed"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for result in results:
        def count(kind):
            return len([i for i in result.issues_by_severity(min_severity)
                        if i.kind == kind])
        rows.append("{:<40} {:>6} {:>7.1f}% {:>8} {:>11} {:>10} {:>9}".format(
            result.filepath[-40:],
            result.loc,
            result.coverage.coverage_pct,
            count("unused_var"),
            count("undefined"),
            count("dead_code"),
            count("shadowed"),
        ))

    rows.append(sep)
    total_loc = sum(r.loc for r in results)
    total_issues = sum(r.total_issues for r in results)
    avg_cov = (sum(r.coverage.coverage_pct for r in results) / len(results)) if results else 0
    rows.append("Total: {} files, {} LoC, {:.1f}% avg coverage, {} issues".format(
        len(results), total_loc, avg_cov, total_issues
    ))
    return "\n".join(rows)


def write_report(results, output_path, min_severity=Issue.SEVERITY_LOW, verbose=False):
    # type: (List[AnalysisResult], str, int, bool) -> None
    parts = [format_summary_table(results, min_severity), ""]
    for result in results:
        parts.append(format_result(result, min_severity, verbose))
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(parts))
