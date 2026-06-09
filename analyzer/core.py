"""
core.py - Static analyzer built on beniget def-use chains.

Compatible with beniget 0.4.x and 0.5.x, Python 3.6+.

Metrics implemented
-------------------
Variable Definition Detection  – every name defined anywhere in the file
Definition-Use Mapping         – each definition -> all its use sites
Coverage Measurement (DU-Path) – % of (def, use) pairs that have a
                                  def-clear path; computed as:
                                    covered = beniget-chained (def,use) pairs
                                    total   = every (def, use) pair for the
                                              same variable name
                                    pct     = covered / total * 100
Uncovered Definition Detection – definitions with zero users
Edge Case Handling             – lambdas, comprehensions, classes,
                                  try/except, with-targets, nested scopes,
                                  imports, *args/**kwargs
"""
import gast
from beniget import DefUseChains
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Compatibility: beniget 0.4.x vs 0.5.x
# ---------------------------------------------------------------------------

def _iter_scope_defs(duc, scope):
    # type: (DefUseChains, object) -> Iterator[Tuple[str, object]]
    """
    Yield (name, Def) for every definition in *scope*.

    beniget 0.4.x: duc.locals[scope] is {name: [Def]}
    beniget 0.5.x: duc.locals[scope] is ordered_set of Def  (defn.name() callable)
    """
    scope_data = duc.locals.get(scope)
    if scope_data is None:
        return

    if hasattr(scope_data, "items"):
        # 0.4.x dict
        for name, defs in scope_data.items():
            for defn in defs:
                yield name, defn
    else:
        # 0.5.x ordered_set
        for defn in scope_data:
            raw = defn.name
            name = raw() if callable(raw) else str(raw)
            yield name, defn


# ---------------------------------------------------------------------------
# DU-Pair  (Definition-Use pair)
# ---------------------------------------------------------------------------

class DUPair:
    """
    One (definition, use) pair for a named variable.

    covered = True  ⟺ beniget found a def-clear path from def_lineno to
                       use_lineno (the variable is not redefined on any path
                       between the two sites).
    """

    def __init__(self, name, def_lineno, def_col, use_lineno, use_col, covered):
        self.name = name
        self.def_lineno = def_lineno
        self.def_col = def_col
        self.use_lineno = use_lineno
        self.use_col = use_col
        self.covered = covered

    @property
    def key(self):
        return (self.name, self.def_lineno, self.use_lineno)

    def __repr__(self):
        status = "covered" if self.covered else "uncovered"
        return "DUPair({!r}: def@{}, use@{}, {})".format(
            self.name, self.def_lineno, self.use_lineno, status
        )


# ---------------------------------------------------------------------------
# DU-Path Coverage
# ---------------------------------------------------------------------------

class DUPathCoverage:
    """
    DU-path coverage for one source file.

    Algorithm
    ---------
    For each variable name *x*:
      D = all definition sites of *x*
      U = all use sites of *x*  (Name nodes in Load context)

      total_pairs  = { (d, u) | d ∈ D, u ∈ U }   — Cartesian product
      covered_pairs = those pairs where beniget chains d → u
                      (guarantees a def-clear path exists in the CFG)

    coverage_pct = |covered_pairs| / |total_pairs| * 100
    """

    def __init__(self):
        self.du_pairs = []       # type: List[DUPair]

    def add(self, pair):
        # type: (DUPair) -> None
        self.du_pairs.append(pair)

    @property
    def total(self):
        return len(self.du_pairs)

    @property
    def covered(self):
        return sum(1 for p in self.du_pairs if p.covered)

    @property
    def uncovered(self):
        return sum(1 for p in self.du_pairs if not p.covered)

    @property
    def covered_pairs(self):
        return [p for p in self.du_pairs if p.covered]

    @property
    def uncovered_pairs(self):
        return [p for p in self.du_pairs if not p.covered]

    @property
    def coverage_pct(self):
        # type: () -> float
        if self.total == 0:
            return 100.0
        return round(self.covered / self.total * 100.0, 2)

    def by_variable(self, name):
        # type: (str) -> List[DUPair]
        return [p for p in self.du_pairs if p.name == name]

    def __repr__(self):
        return "DUPathCoverage(total={}, covered={}, pct={:.1f}%)".format(
            self.total, self.covered, self.coverage_pct
        )


# ---------------------------------------------------------------------------
# DefUseEntry  (single definition + its use locations)
# ---------------------------------------------------------------------------

class DefUseEntry:
    """One definition and every location that references it."""

    KINDS = ("variable", "parameter", "function", "class",
             "import", "lambda", "comprehension", "other")

    def __init__(self, name, kind, lineno, col_offset=0):
        self.name = name
        self.kind = kind
        self.lineno = lineno
        self.col_offset = col_offset
        self.uses = []   # type: List[Tuple[int, int]]

    def add_use(self, lineno, col_offset=0):
        self.uses.append((lineno, col_offset))

    @property
    def use_count(self):
        return len(self.uses)

    @property
    def is_covered(self):
        return bool(self.uses)

    def __repr__(self):
        return "DefUseEntry(name={!r}, kind={!r}, lineno={}, uses={})".format(
            self.name, self.kind, self.lineno, self.use_count
        )


# ---------------------------------------------------------------------------
# CoverageMetrics  (definition-level coverage – "which defs have any use")
# ---------------------------------------------------------------------------

class CoverageMetrics:
    """
    Definition-level coverage: how many definitions have ≥1 user.

    Note: for DU-path coverage (pair-level granularity) see DUPathCoverage.
    """

    def __init__(self):
        self.total_defs = 0
        self.covered_defs = 0
        self.entries = []   # type: List[DefUseEntry]

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
        return "CoverageMetrics(total={}, covered={}, pct={:.1f}%)".format(
            self.total_defs, self.covered_defs, self.coverage_pct
        )


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

class Issue:
    """A single analysis finding."""

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
    """All findings for a single source file."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.issues = []                     # type: List[Issue]
        self.errors = []                     # type: List[str]
        self.loc = 0
        self.def_use_map = {}                # type: Dict[Tuple, DefUseEntry]
        self.coverage = CoverageMetrics()    # definition-level coverage
        self.du_path_coverage = DUPathCoverage()  # DU-pair coverage

    # ── convenience aliases ────────────────────────────────────────────────
    @property
    def coverage_pct(self):
        """Definition-level coverage %."""
        return self.coverage.coverage_pct

    @property
    def du_coverage_pct(self):
        """DU-path coverage %."""
        return self.du_path_coverage.coverage_pct

    # ── mutation helpers ───────────────────────────────────────────────────
    def add_issue(self, issue):
        self.issues.append(issue)

    def add_error(self, msg):
        self.errors.append(msg)

    # ── queries ────────────────────────────────────────────────────────────
    @property
    def total_issues(self):
        return len(self.issues)

    def issues_by_kind(self, kind):
        return [i for i in self.issues if i.kind == kind]

    def issues_by_severity(self, min_severity):
        """Return issues at or above *min_severity*."""
        return [i for i in self.issues if i.severity >= min_severity]

    def get_def_at(self, lineno, name=None):
        # type: (int, Optional[str]) -> Optional[DefUseEntry]
        for entry in self.def_use_map.values():
            if entry.lineno == lineno:
                if name is None or entry.name == name:
                    return entry
        return None

    def get_uses_for(self, lineno, name=None):
        # type: (int, Optional[str]) -> List[Tuple[int, int]]
        entry = self.get_def_at(lineno, name)
        return entry.uses if entry else []

    def get_du_pairs_for(self, name):
        # type: (str) -> List[DUPair]
        """Return all DU-pairs for variable *name*."""
        return self.du_path_coverage.by_variable(name)

    def summary(self):
        kinds = defaultdict(int)
        for issue in self.issues:
            kinds[issue.kind] += 1
        parts = ["{}: {}".format(k, v) for k, v in sorted(kinds.items())]
        return "{} | du_cov={:.1f}% | {}".format(
            self.filepath, self.du_coverage_pct, ", ".join(parts)
        )

    def __repr__(self):
        return (
            "AnalysisResult(filepath={!r}, issues={}, "
            "def_cov={:.1f}%, du_cov={:.1f}%)".format(
                self.filepath, self.total_issues,
                self.coverage_pct, self.du_coverage_pct,
            )
        )


# ---------------------------------------------------------------------------
# Built-in names
# ---------------------------------------------------------------------------

BUILTIN_NAMES = {
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool",
    "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "delattr", "callable", "iter", "next", "open", "super", "object",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
    "OSError", "IOError", "True", "False", "None", "__name__", "__file__",
    "__doc__", "__all__", "__slots__", "abs", "all", "any", "bin", "chr",
    "dir", "divmod", "format", "hash", "hex", "id", "input", "max", "min",
    "oct", "ord", "pow", "repr", "reversed", "round", "sorted", "sum",
    "vars", "staticmethod", "classmethod", "property", "NotImplemented",
    "bytearray", "bytes", "complex", "frozenset", "memoryview", "slice",
    "ArithmeticError", "AssertionError", "BlockingIOError", "BrokenPipeError",
    "BufferError", "ChildProcessError", "ConnectionError", "EOFError",
    "EnvironmentError", "FileExistsError", "FileNotFoundError",
    "FloatingPointError", "GeneratorExit", "ImportError", "InterruptedError",
    "IsADirectoryError", "LookupError", "MemoryError", "ModuleNotFoundError",
    "NameError", "NotADirectoryError", "OverflowError", "PermissionError",
    "ProcessLookupError", "RecursionError", "ReferenceError", "SyntaxError",
    "SystemError", "SystemExit", "TimeoutError", "UnicodeDecodeError",
    "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError",
    "UserWarning", "Warning", "ZeroDivisionError",
    "BaseException", "ExceptionGroup",
}


def _node_kind(node, scope=None):
    # type: (object, object) -> str
    if isinstance(node, (gast.FunctionDef, gast.AsyncFunctionDef)):
        return "function"
    if isinstance(node, gast.ClassDef):
        return "class"
    if isinstance(node, (gast.Import, gast.ImportFrom)):
        return "import"
    if isinstance(node, gast.Lambda):
        return "lambda"
    if isinstance(node, (gast.ListComp, gast.SetComp, gast.DictComp, gast.GeneratorExp)):
        return "comprehension"
    if isinstance(node, gast.Name) and scope is not None:
        if isinstance(scope, (gast.FunctionDef, gast.AsyncFunctionDef, gast.Lambda)):
            args = scope.args
            param_nodes = list(args.args)
            if hasattr(args, "posonlyargs"):
                param_nodes += list(args.posonlyargs)
            if args.vararg:
                param_nodes.append(args.vararg)
            if args.kwarg:
                param_nodes.append(args.kwarg)
            param_nodes += list(args.kwonlyargs)
            if node in param_nodes:
                return "parameter"
    return "variable"


def _is_synthetic(name):
    # type: (str) -> bool
    return name.startswith("<") and name.endswith(">")


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CodeAnalyzer:
    """
    Analyses Python source using beniget def-use chains.

    Key outputs in AnalysisResult
    ------------------------------
    def_use_map       : {(lineno,col,name) -> DefUseEntry}
    coverage          : CoverageMetrics  – definition-level
    du_path_coverage  : DUPathCoverage   – pair-level (DU-path validation)
    issues            : List[Issue]
    """

    def __init__(self, ignore_private=False, ignore_underscore=True):
        self.ignore_private = ignore_private
        self.ignore_underscore = ignore_underscore
        self._results = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_source(self, source, filepath="<string>"):
        # type: (str, str) -> AnalysisResult
        result = AnalysisResult(filepath)
        result.loc = source.count("\n") + 1

        try:
            module = gast.parse(source)
        except SyntaxError as exc:
            result.add_error("SyntaxError at line {}: {}".format(
                exc.lineno, exc.msg))
            return result

        try:
            duc = DefUseChains()
            duc.visit(module)
        except Exception as exc:
            result.add_error("beniget error: {}".format(exc))
            return result

        self._build_def_use_map(module, duc, result)
        self._compute_def_coverage(result)
        self._compute_du_path_coverage(module, duc, result)
        self._check_unused_vars(module, duc, result)
        self._check_undefined_names(module, duc, result)
        self._check_dead_code(module, result)
        self._check_shadowed_names(module, duc, result)

        self._results[filepath] = result
        return result

    def analyze_file(self, filepath):
        # type: (str) -> AnalysisResult
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
        Populate result.def_use_map: (lineno, col, name) -> DefUseEntry.
        Covers all scope types beniget knows about.
        """
        seen = set()
        for scope in duc.locals:
            for name, defn in _iter_scope_defs(duc, scope):
                if _is_synthetic(name):
                    continue
                node = defn.node
                if not hasattr(node, "lineno"):
                    continue
                lineno = node.lineno
                col = getattr(node, "col_offset", 0)
                key = (lineno, col, name)
                if key in seen:
                    continue
                seen.add(key)

                entry = DefUseEntry(
                    name=name,
                    kind=_node_kind(node, scope),
                    lineno=lineno,
                    col_offset=col,
                )
                for user in defn.users():
                    u = user.node
                    if hasattr(u, "lineno"):
                        entry.add_use(u.lineno, getattr(u, "col_offset", 0))

                result.def_use_map[key] = entry

    # ------------------------------------------------------------------
    # Definition-level coverage
    # ------------------------------------------------------------------

    def _compute_def_coverage(self, result):
        """
        Populate result.coverage: count of defs with ≥1 user vs total defs.
        """
        for entry in result.def_use_map.values():
            if entry.name in BUILTIN_NAMES or _is_synthetic(entry.name):
                continue
            if self.ignore_underscore and entry.name.startswith("_"):
                continue
            result.coverage.add_entry(entry)

    # ------------------------------------------------------------------
    # DU-Path Coverage  ← the Coverage Measurement metric
    # ------------------------------------------------------------------

    def _compute_du_path_coverage(self, module, duc, result):
        """
        Compute DU-path coverage (pair-level).

        Algorithm:
          For each variable name x (non-builtin, non-synthetic):
            D  = all definition nodes of x (across all scopes)
            U  = all Load-context Name nodes for x in the module
            covered_set = { (d.lineno, u.lineno) |
                            u ∈ d.users() }  ← beniget def-clear chains

          For every (d, u) ∈ D × U:
            pair.covered = (d.lineno, u.lineno) ∈ covered_set

          DU-path coverage = |covered| / |D × U| * 100
        """
        # Collect all definitions by variable name
        defs_by_name = defaultdict(list)   # name -> [(lineno, col, Def)]
        for scope in duc.locals:
            for name, defn in _iter_scope_defs(duc, scope):
                if _is_synthetic(name) or name in BUILTIN_NAMES:
                    continue
                if self.ignore_underscore and name.startswith("_"):
                    continue
                node = defn.node
                if not hasattr(node, "lineno"):
                    continue
                defs_by_name[name].append(
                    (node.lineno, getattr(node, "col_offset", 0), defn)
                )

        # Collect all use sites by variable name
        uses_by_name = defaultdict(list)   # name -> [(lineno, col)]
        for node in gast.walk(module):
            if isinstance(node, gast.Name) and isinstance(node.ctx, gast.Load):
                name = node.id
                if _is_synthetic(name) or name in BUILTIN_NAMES:
                    continue
                if self.ignore_underscore and name.startswith("_"):
                    continue
                uses_by_name[name].append(
                    (node.lineno, getattr(node, "col_offset", 0))
                )

        # Build covered set from beniget chains
        covered_keys = set()   # (name, def_lineno, use_lineno)
        for name, def_list in defs_by_name.items():
            for def_lineno, _, defn in def_list:
                for user in defn.users():
                    u_node = user.node
                    if hasattr(u_node, "lineno"):
                        covered_keys.add((name, def_lineno, u_node.lineno))

        # Build full DU-pair set: D × U for each name
        seen_pairs = set()
        for name in set(list(defs_by_name.keys()) + list(uses_by_name.keys())):
            for def_lineno, def_col, _ in defs_by_name.get(name, []):
                for use_lineno, use_col in uses_by_name.get(name, []):
                    pair_key = (name, def_lineno, use_lineno)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    covered = pair_key in covered_keys
                    result.du_path_coverage.add(DUPair(
                        name=name,
                        def_lineno=def_lineno,
                        def_col=def_col,
                        use_lineno=use_lineno,
                        use_col=use_col,
                        covered=covered,
                    ))

    # ------------------------------------------------------------------
    # Issue detection
    # ------------------------------------------------------------------

    def _should_skip(self, name):
        if _is_synthetic(name) or name in BUILTIN_NAMES:
            return True
        if self.ignore_underscore and name.startswith("_"):
            return True
        if self.ignore_private and name.startswith("__"):
            return True
        return False

    def _check_unused_vars(self, module, duc, result):
        for scope in duc.locals:
            if isinstance(scope, gast.Module):
                continue
            for name, defn in _iter_scope_defs(duc, scope):
                if self._should_skip(name):
                    continue
                if not defn.users():
                    lineno = getattr(defn.node, "lineno", 0)
                    result.add_issue(Issue(
                        kind="unused_var",
                        name=name,
                        lineno=lineno,
                        severity=Issue.SEVERITY_MEDIUM,
                        message="'{}' defined but never used".format(name),
                    ))

    def _check_undefined_names(self, module, duc, result):
        module_names = {name for name, _ in _iter_scope_defs(duc, module)}
        for node in gast.walk(module):
            if not isinstance(node, gast.Name):
                continue
            if not isinstance(node.ctx, gast.Load):
                continue
            name = node.id
            if name in BUILTIN_NAMES or name in module_names:
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
                    self._scan_for_dead_code(branch, result)
                continue
            if body:
                self._scan_for_dead_code(body, result)

    def _scan_for_dead_code(self, body, result):
        for i in range(len(body) - 1):
            stmt = body[i]
            if isinstance(stmt, (gast.Return, gast.Raise,
                                  gast.Break, gast.Continue)):
                result.add_issue(Issue(
                    kind="dead_code",
                    name="<stmt>",
                    lineno=body[i + 1].lineno,
                    severity=Issue.SEVERITY_HIGH,
                    message="unreachable code after {}".format(
                        type(stmt).__name__.lower()),
                ))

    def _check_shadowed_names(self, module, duc, result):
        module_names = {name for name, _ in _iter_scope_defs(duc, module)}
        for scope in duc.locals:
            if isinstance(scope, gast.Module):
                continue
            for name, defn in _iter_scope_defs(duc, scope):
                if name in module_names and not self._should_skip(name):
                    result.add_issue(Issue(
                        kind="shadowed",
                        name=name,
                        lineno=getattr(defn.node, "lineno", 0),
                        severity=Issue.SEVERITY_LOW,
                        message="'{}' shadows a module-level name".format(name),
                    ))
