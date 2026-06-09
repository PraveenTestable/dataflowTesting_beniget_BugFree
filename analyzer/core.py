"""
core.py - Static analyzer built on beniget def-use chains.

Compatible with beniget 0.4.x and 0.5.x, Python 3.6+.

beniget 0.5.x API (used throughout):
  duc.locals[scope]   -> ordered_set of Def objects  (NOT a {name:[Def]} dict)
  defn.name()         -> str identifier (callable method)
  defn.users()        -> list[Def] – every node that uses this definition
  defn.node           -> underlying gast AST node

beniget 0.4.x fallback:
  duc.locals[scope]   -> dict {name: [Def]}
"""
import gast
from beniget import DefUseChains
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Compatibility helper
# ---------------------------------------------------------------------------

def _iter_scope_defs(duc, scope):
    # type: (DefUseChains, object) -> Iterator[Tuple[str, object]]
    """
    Yield (name, Def) for every definition in *scope*.

    Handles both beniget 0.4.x (dict) and 0.5.x (ordered_set).
    """
    scope_data = duc.locals.get(scope)
    if scope_data is None:
        return

    if hasattr(scope_data, "items"):
        # beniget 0.4.x: {name: [Def, ...]}
        for name, defs in scope_data.items():
            for defn in defs:
                yield name, defn
    else:
        # beniget 0.5.x: ordered_set of Def objects
        for defn in scope_data:
            raw = defn.name
            name = raw() if callable(raw) else str(raw)
            yield name, defn


# ---------------------------------------------------------------------------
# Data containers
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


class CoverageMetrics:
    """Coverage summary: how many definitions are actually used."""

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


class AnalysisResult:
    """All findings for a single source file."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.issues = []                    # type: List[Issue]
        self.errors = []                    # type: List[str]
        self.loc = 0
        self.def_use_map = {}               # type: Dict[Tuple, DefUseEntry]
        self.coverage = CoverageMetrics()

    # convenience alias so callers can use result.coverage_pct directly
    @property
    def coverage_pct(self):
        return self.coverage.coverage_pct

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

    def get_def_at(self, lineno, name=None):
        # type: (int, Optional[str]) -> Optional[DefUseEntry]
        """Return the DefUseEntry defined on *lineno* (optionally matching *name*)."""
        for key, entry in self.def_use_map.items():
            if entry.lineno == lineno:
                if name is None or entry.name == name:
                    return entry
        return None

    def get_uses_for(self, lineno, name=None):
        # type: (int, Optional[str]) -> List[Tuple[int, int]]
        """Return all use-sites for the definition on *lineno*."""
        entry = self.get_def_at(lineno, name)
        return entry.uses if entry else []

    def summary(self):
        kinds = defaultdict(int)
        for issue in self.issues:
            kinds[issue.kind] += 1
        parts = ["{}: {}".format(k, v) for k, v in sorted(kinds.items())]
        return "{} | {}".format(self.filepath, ", ".join(parts))

    def __repr__(self):
        return (
            "AnalysisResult(filepath={!r}, issues={}, coverage={:.1f}%)".format(
                self.filepath, self.total_issues, self.coverage_pct
            )
        )


# ---------------------------------------------------------------------------
# Built-in names (never flagged as undefined / unused)
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
    "BaseException", "BaseExceptionGroup", "ExceptionGroup",
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
    # Detect function parameters: Name nodes inside function arg lists
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
    """Return True for beniget-internal synthetic names like '<ListComp>'."""
    return name.startswith("<") and name.endswith(">")


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CodeAnalyzer:
    """
    Analyses Python source using beniget def-use chains.

    Covers:
      Variable Definition Detection  – every name defined anywhere in the file
      Definition-Use Mapping         – each definition -> list of use sites
      Coverage Measurement           – % of definitions with ≥1 use
      Uncovered Definition Detection – definitions with zero uses
      Edge Case Handling             – lambdas, comprehensions, classes,
                                       try/except bindings, with-targets,
                                       augmented assignments, nested scopes,
                                       imports, *args/**kwargs, decorators
    """

    def __init__(self, ignore_private=False, ignore_underscore=True):
        self.ignore_private = ignore_private
        self.ignore_underscore = ignore_underscore
        self._results = {}   # type: Dict[str, AnalysisResult]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_source(self, source, filepath="<string>"):
        # type: (str, str) -> AnalysisResult
        """Parse *source* and return a fully populated :class:`AnalysisResult`."""
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

        # Core passes ─ order matters
        self._build_def_use_map(module, duc, result)
        self._compute_coverage(result)
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

        Iterates EVERY scope beniget knows about, which automatically covers:
        - Module level
        - Functions and async functions
        - Classes and their methods
        - Lambda expressions
        - List / set / dict comprehensions and generator expressions
        - Nested / inner functions
        - Exception-handler bindings  (appear in enclosing scope)
        - with-statement targets       (appear in enclosing scope)
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

                # defn.users() is beniget's direct API – works in both 0.4.x and 0.5.x
                for user in defn.users():
                    u = user.node
                    if hasattr(u, "lineno"):
                        entry.add_use(u.lineno, getattr(u, "col_offset", 0))

                result.def_use_map[key] = entry

    # ------------------------------------------------------------------
    # Coverage Measurement
    # ------------------------------------------------------------------

    def _compute_coverage(self, result):
        """
        Populate result.coverage from result.def_use_map.

        Every non-synthetic, non-builtin definition counts toward
        total_defs; those with ≥1 use count toward covered_defs.
        """
        for entry in result.def_use_map.values():
            if entry.name in BUILTIN_NAMES:
                continue
            if _is_synthetic(entry.name):
                continue
            if self.ignore_underscore and entry.name.startswith("_"):
                continue
            result.coverage.add_entry(entry)

    # ------------------------------------------------------------------
    # Issue detection
    # ------------------------------------------------------------------

    def _should_skip(self, name):
        # type: (str) -> bool
        if _is_synthetic(name):
            return True
        if name in BUILTIN_NAMES:
            return True
        if self.ignore_underscore and name.startswith("_"):
            return True
        if self.ignore_private and name.startswith("__"):
            return True
        return False

    def _check_unused_vars(self, module, duc, result):
        """
        Emit MEDIUM issues for every definition that has zero users.
        Covers all scope types including lambdas and comprehensions.
        """
        for scope in duc.locals:
            if isinstance(scope, gast.Module):
                continue   # module-level unused names are reported differently
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
        """
        Emit HIGH issues for Name nodes in Load context that have no
        definition reachable at module scope.
        """
        module_names = set()
        for name, _ in _iter_scope_defs(duc, module):
            module_names.add(name)

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
                if name in module_names and name not in BUILTIN_NAMES:
                    if not _is_synthetic(name):
                        result.add_issue(Issue(
                            kind="shadowed",
                            name=name,
                            lineno=getattr(defn.node, "lineno", 0),
                            severity=Issue.SEVERITY_LOW,
                            message="'{}' shadows a module-level name".format(name),
                        ))
