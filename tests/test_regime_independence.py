"""
Static guard: the regime-label pipeline must never depend on satellite fields
used as prediction inputs (CLAUDE.md §4.2, §8).

This is a source-level import check — it does NOT execute regime_labels, so it
runs even while that module is still a stub. It fails if regime_labels.py (or
anything in src/regime/) references a forbidden module.

Run:  .venv/Scripts/python.exe -m pytest tests/test_regime_independence.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

REGIME_DIR = Path(__file__).resolve().parents[1] / "src" / "regime"

# Modules / names that would reintroduce the circularity flaw.
FORBIDDEN_IMPORT_SUBSTRINGS = (
    "fetch_satellite",
    "fetch_argo",        # ground truth — also must not leak into labels
    "data_access.fetch_satellite",
)
# Bare identifiers that, if imported, signal a satellite field is being pulled in.
FORBIDDEN_NAMES = {"SST", "SSS", "SSH", "ADT", "ocean_color"}


def _iter_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                yield mod, f"{mod}.{alias.name}" if mod else alias.name


def test_regime_module_has_no_satellite_imports():
    py_files = sorted(REGIME_DIR.glob("*.py"))
    assert py_files, f"no python files found in {REGIME_DIR}"

    violations = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for mod, full in _iter_imports(tree):
            for bad in FORBIDDEN_IMPORT_SUBSTRINGS:
                if bad in mod or bad in full:
                    violations.append(f"{path.name}: imports '{full}'")
            tail = full.rsplit(".", 1)[-1]
            if tail in FORBIDDEN_NAMES:
                violations.append(f"{path.name}: imports name '{tail}'")

    assert not violations, (
        "regime pipeline must not import satellite/ground-truth fields:\n  "
        + "\n  ".join(violations)
    )
