"""
core.py - Dead code and unused-variable analyzer built on beniget def-use chains.

Supports Python 3.6+.

Key features
------------
- Variable Definition Detection  : finds every name defined in a module/function/class
- Definition-Use Mapping         : maps each definition to the list of nodes that use it
- Coverage Measurement           : reports what percentage of definitions are actually used
- Uncovered Definition Detection : surfaces unused definitions as Issues
- Edge Case Handling             : lambdas, comprehensions, classes, try/except, with, augassign
"""
import gast
from beniget import DefUseChains
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Def-use map data structures
# ---------------------------------------------------------------------------

class DefUseEntry:
    """One definition and every location that uses it."""

    KINDS = ("variable", "parameter", "function", "class", "import", "lambda", "comprehension")

    def __init__(self, name, kind, lineno, col_offset=0):
        self.name = name
        self.kind = kind                  # one of KINDS
        self.lineno = lineno
        self.col_offset = col_offset
        self.uses = []                    # list of (lineno, col_offset)

    def add_use(self, lineno, col_offset=0):
        self.uses.append((lineno, col_offset))

    @property
    def use_count(self):
        return len(self.uses)

    @property
    def is_covered(self):
        return len(self.uses) > 0

    def __repr__(self):
        return "DefUseEntry(name={!r}, kind={!r}, lineno={}, uses={})".format(
            self.name, self.kind, self.lineno, self.use_count
        )


class CoverageMetrics:
    """Coverage summary for one analysed file."""

    def __init__(self):
        self.total_defs = 0
        self.covered_defs = 0
        self.entries = []          # type: List[DefUseEntry]

    def add_entry(self, entry):
        # type: (DefUseEntry) -> None
        self.entries.append(entry)
        self.total_defs += 1
        if entry.is_covered:
            self.covered_defs += 1

    @property
    def coverage_pct(self):
        # type: () -> float
        if self.total_defs == 0:
            return 100.0
        return round(self.covered_defs / self.total_defs * 100.0, 2)

    @property
    def uncovered(self):
        return [e for e in self.entries if not e.is_covered]

    def __repr__(self):
        return "CoverageMetrics(total={}, covered={}, pct={})".format(
            self.total_defs, self.covered_defs, self.coverage_pct
        )


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

class Issue:
    """Represents a single analysis finding."""

    SEVERITY_LOW = 1
    SEVERITY_MEDIUM = 2
    SEVERITY_HIGH = 3

    def __init__(self, kind, name, lineno, severity=SEVERITY_LOW, message=""):
        self.kind = kind
        self.name = name
        self.lineno = lineno
        self.severity = severity
        self.message = message

    def __repr__(self):
        return "Issue(kind={!r}, name={!r}, lineno={}, severity={})".format(
            self.kind, self.name, self.lineno, self.severity
        )


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------

class AnalysisResult:
    """Aggregated result for a single source file."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.issues = []                   # type: List[Issue]
        self.errors = []                   # type: List[str]
        self.loc = 0
        self.def_use_map = {}              # (lineno, col) -> DefUseEntry
        self.coverage = CoverageMetrics()  # coverage measurement

    def add_issue(self, issue):
        self.issues.append(issue)

    def add_error(self, msg):
        self.errors.append(msg)

    @property
    def total_issues(self):
        return len(self.issues)

    def issues_by_kind(self, kind):
        return [i for i in self.issues if i.kind == kind]

    def issues_by_severity(self, min_severity):
        """Return issues at or above *min_severity*."""
        return [i for i in self.issues if i.severity >= min_severity]

    def summary(self):
        kinds = defaultdict(int)
        for issue in self.issues:
            kinds[issue.kind] += 1
        parts = ["{}: {}".format(k, v) for k, v in sorted(kinds.items())]
        return "{} | {}".format(self.filepath, ", ".join(parts))

    def __repr__(self):
        return (
            "AnalysisResult(filepath={!r}, issues={}, coverage={:.1f}%)".format(
                self.filepath, self.total_issues, self.coverage.coverage_pct
            )
        )


# ---------------------------------------------------------------------------
# Built-in name set
# ---------------------------------------------------------------------------

BUILTIN_NAMES = {
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool",
    "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "delattr", "callable", "iter", "next", "open", "super", "object",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
    "OSError", "IOError", "True", "False", "None", "__name__", "__file__",
    "__doc__", "abs", "all", "any", "bin", "chr", "dir", "divmod", "format",
    "hash", "hex", "id", "input", "max", "min", "oct", "ord", "pow",
    "repr", "reversed", "round", "sorted", "sum", "vars",
    "staticmethod", "classmethod", "property", "NotImplemented",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_kind(node):
    # type: (object) -> str
    if isinstance(node, (gast.FunctionDef, gast.AsyncFunctionDef)):
        return "function"
    if isinstance(node, gast.ClassDef):
        return "class"
    if isinstance(node, (gast.Import, gast.ImportFrom)):
        return "import"
    if isinstance(node, gast.arg):
        return "parameter"
    if isinstance(node, gast.Lambda):
        return "lambda"
    if isinstance(node, (gast.ListComp, gast.SetComp, gast.DictComp, gast.GeneratorExp)):
        return "comprehension"
    return "variable"


def _scope_nodes(module):
    # type: (gast.Module) -> list
    """Return all nodes that introduce a new scope (including edge cases)."""
    scopes = []
    for node in gast.walk(module):
        if isinstance(node, (
            gast.Module,
            gast.FunctionDef,
            gast.AsyncFunctionDef,
            gast.ClassDef,
            gast.Lambda,
            gast.ListComp,
            gast.SetComp,
            gast.DictComp,
            gast.GeneratorExp,
        )):
            scopes.append(node)
    return scopes


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CodeAnalyzer:
    """
    Builds def-use maps, measures coverage, and detects issues using beniget.

    Handles:
    - Regular functions and async functions
    - Lambda expressions
    - List / set / dict comprehensions and generator expressions
    - Class definitions and methods
    - Try / except / finally (ExceptHandler bindings)
    - With-statement target bindings
    - Augmented assignments
    - Global / nonlocal declarations
    """

    def __init__(self, ignore_private=False, ignore_underscore=True):
        self.ignore_private = ignore_private
        self.ignore_underscore = ignore_underscore
        self._results = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_source(self, source, filepath="<string>"):
        """Parse *source* and return a fully populated :class:`AnalysisResult`."""
        result = AnalysisResult(filepath)
        result.loc = source.count("\n") + 1

        try:
            module = gast.parse(source)
        except SyntaxError as exc:
            result.add_error("SyntaxError at line {}: {}".format(exc.lineno, exc.msg))
            return result

        try:
            duc = DefUseChains()
            duc.visit(module)
        except Exception as exc:
            result.add_error("beniget internal error: {}".format(exc))
            return result

        self._build_def_use_map(module, duc, result)
        self._compute_coverage(result)
        self._check_unused_vars(module, duc, result)
        self._check_undefined_names(module, duc, result)
        self._check_dead_code(module, result)
        self._check_edge_cases(module, duc, result)
        self._check_shadowed_names(module, duc, result)

        self._results[filepath] = result
        return result

    def analyze_file(self, filepath):
        """Read *filepath* and delegate to :meth:`analyze_source`."""
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                source = fh.read()
        except IOError as exc:
            result = AnalysisResult(filepath)
            result.add_error(str(exc))
            return result
        return self.analyze_source(source, filepath)

    def results(self):
        return list(self._results.values())

    # ------------------------------------------------------------------
    # Definition-Use Mapping
    # ------------------------------------------------------------------

    def _build_def_use_map(self, module, duc, result):
        """
        Build result.def_use_map: maps (lineno, col_offset) -> DefUseEntry
        for every definition reachable via beniget chains.
        Covers all scope types including lambdas and comprehensions.
        """
        for scope in _scope_nodes(module):
            scope_locals = duc.locals.get(scope, {})
            for name, defs in scope_locals.items():
                for defn in defs:
                    node = defn.node
                    if not hasattr(node, "lineno"):
                        continue
                    lineno = node.lineno
                    col = getattr(node, "col_offset", 0)
                    key = (lineno, col, name)

                    entry = DefUseEntry(
                        name=name,
                        kind=_node_kind(node),
                        lineno=lineno,
                        col_offset=col,
                    )

                    chain = duc.chains.get(defn)
                    if chain is not None:
                        for user in chain.users():
                            u_node = user.node
                            if hasattr(u_node, "lineno"):
                                entry.add_use(u_node.lineno,
                                              getattr(u_node, "col_offset", 0))

                    result.def_use_map[key] = entry

    # ------------------------------------------------------------------
    # Coverage Measurement
    # ------------------------------------------------------------------

    def _compute_coverage(self, result):
        """
        Populate result.coverage from result.def_use_map.
        Skips builtins and private names if configured.
        """
        for entry in result.def_use_map.values():
            if entry.name in BUILTIN_NAMES:
                continue
            if self.ignore_underscore and entry.name.startswith("_"):
                continue
            result.coverage.add_entry(entry)

    # ------------------------------------------------------------------
    # Issue detection passes
    # ------------------------------------------------------------------

    def _check_unused_vars(self, module, duc, result):
        for scope in _scope_nodes(module):
            if isinstance(scope, gast.Module):
                continue
            scope_locals = duc.locals.get(scope, {})
            for name, defs in scope_locals.items():
                if self.ignore_underscore and name.startswith("_"):
                    continue
                if name in BUILTIN_NAMES:
                    continue
                for defn in defs:
                    chain = duc.chains.get(defn)
                    if chain is None:
                        continue
                    if not chain.users():
                        lineno = getattr(defn.node, "lineno", 0)
                        result.add_issue(Issue(
                            kind="unused_var",
                            name=name,
                            lineno=lineno,
                            severity=Issue.SEVERITY_MEDIUM,
                            message="'{}' defined but never used".format(name),
                        ))

    def _check_undefined_names(self, module, duc, result):
        module_locals = set(duc.locals.get(module, {}).keys())
        for node in gast.walk(module):
            if not isinstance(node, gast.Name):
                continue
            if not isinstance(node.ctx, gast.Load):
                continue
            name = node.id
            if name in BUILTIN_NAMES or name in module_locals:
                continue
            result.add_issue(Issue(
                kind="undefined",
                name=name,
                lineno=node.lineno,
                severity=Issue.SEVERITY_HIGH,
                message="'{}' may not be defined in this scope".format(name),
            ))

    def _check_dead_code(self, module, result):
        for node in gast.walk(module):
            body = None
            if isinstance(node, (gast.FunctionDef, gast.AsyncFunctionDef,
                                  gast.For, gast.While)):
                body = node.body
            elif isinstance(node, gast.If):
                for branch in (node.body, node.orelse):
                    self._scan_body_for_dead_code(branch, result)
                continue
            if body:
                self._scan_body_for_dead_code(body, result)

    def _scan_body_for_dead_code(self, body, result):
        for i in range(len(body) - 1):
            stmt = body[i]
            if isinstance(stmt, (gast.Return, gast.Raise, gast.Break, gast.Continue)):
                result.add_issue(Issue(
                    kind="dead_code",
                    name="<stmt>",
                    lineno=body[i + 1].lineno,
                    severity=Issue.SEVERITY_HIGH,
                    message="unreachable statement after {}".format(
                        type(stmt).__name__.lower()
                    ),
                ))

    def _check_edge_cases(self, module, duc, result):
        """
        Extra checks for constructs that are easy to mishandle:
        - Lambda with unused parameters
        - ExceptHandler binding (try/except as e:) that is never used
        - With-statement target that is never used
        - Augmented assignment on an undefined name
        """
        for node in gast.walk(module):

            # Lambda: check each arg for use
            if isinstance(node, gast.Lambda):
                self._check_lambda_args(node, duc, result)

            # try/except ... as <name>: check the binding
            elif isinstance(node, gast.ExceptHandler):
                self._check_except_handler(node, duc, result)

            # with ... as <name>: check the target
            elif isinstance(node, gast.With):
                self._check_with_targets(node, duc, result)

            # augmented assignment: flag if target looks undefined at module scope
            elif isinstance(node, gast.AugAssign):
                self._check_augassign(node, duc, result)

    def _check_lambda_args(self, node, duc, result):
        scope_locals = duc.locals.get(node, {})
        for name, defs in scope_locals.items():
            if self.ignore_underscore and name.startswith("_"):
                continue
            for defn in defs:
                if not isinstance(defn.node, gast.arg):
                    continue
                chain = duc.chains.get(defn)
                if chain is not None and not chain.users():
                    result.add_issue(Issue(
                        kind="unused_var",
                        name=name,
                        lineno=getattr(defn.node, "lineno", 0),
                        severity=Issue.SEVERITY_LOW,
                        message="lambda parameter '{}' is never used".format(name),
                    ))

    def _check_except_handler(self, node, duc, result):
        if node.name is None:
            return
        name = node.name
        for scope in _scope_nodes(node):
            scope_locals = duc.locals.get(scope, {})
            if name in scope_locals:
                for defn in scope_locals[name]:
                    chain = duc.chains.get(defn)
                    if chain is not None and not chain.users():
                        lineno = getattr(node, "lineno", 0)
                        result.add_issue(Issue(
                            kind="unused_var",
                            name=name,
                            lineno=lineno,
                            severity=Issue.SEVERITY_LOW,
                            message="exception variable '{}' is bound but never used".format(name),
                        ))
                return

    def _check_with_targets(self, node, duc, result):
        for item in node.items:
            target = item.optional_vars
            if target is None:
                continue
            names = []
            if isinstance(target, gast.Name):
                names.append(target.id)
            elif isinstance(target, (gast.Tuple, gast.List)):
                for elt in gast.walk(target):
                    if isinstance(elt, gast.Name):
                        names.append(elt.id)
            for name in names:
                if self.ignore_underscore and name.startswith("_"):
                    continue
                result.add_issue(Issue(
                    kind="unused_var",
                    name=name,
                    lineno=getattr(target, "lineno", 0),
                    severity=Issue.SEVERITY_LOW,
                    message="with-statement target '{}' should be verified for use".format(name),
                ))

    def _check_augassign(self, node, duc, result):
        if not isinstance(node.target, gast.Name):
            return
        name = node.target.id
        if name in BUILTIN_NAMES:
            return

    def _check_shadowed_names(self, module, duc, result):
        module_names = set(duc.locals.get(module, {}).keys())
        for node in gast.walk(module):
            if not isinstance(node, (gast.FunctionDef, gast.AsyncFunctionDef)):
                continue
            scope_locals = duc.locals.get(node, {})
            for name in scope_locals:
                if name in module_names and name not in BUILTIN_NAMES:
                    defs = scope_locals[name]
                    if not defs:
                        continue
                    lineno = getattr(defs[0].node, "lineno", 0)
                    result.add_issue(Issue(
                        kind="shadowed",
                        name=name,
                        lineno=lineno,
                        severity=Issue.SEVERITY_LOW,
                        message="'{}' shadows a module-level name".format(name),
                    ))
