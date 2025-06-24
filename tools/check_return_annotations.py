#!/usr/bin/env python3
"""
Simple script to check for missing return type annotations on functions that return values.

Reads `exclude` patterns from pyproject.toml. It respects an `exclude-from`
key in its own [tool.check-return-annotations] section to inherit settings from
other tools like mypy or coverage.

Features:
- Checks files and directories recursively.
- Ignores common non-project directories (.venv, __pycache__, etc.).
- Configurable via `pyproject.toml` [tool.check-return-annotations].
- Can inherit excludes from other tools via `exclude-from = "tool.section:key"`.
- Command-line flags can override all configurations for one-off runs.
"""

import argparse
import ast
import contextlib
import sys
import types
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

# safe load tomllib with fallback
if TYPE_CHECKING:

    class TomlLib(Protocol):
        def load(self, f: BinaryIO) -> Dict[str, Any]: ...


tomllib: "Optional[TomlLib]" = None
tl_main_import: Optional[types.ModuleType] = None
tl_fallback: Optional[types.ModuleType] = None
try:
    import tomllib as tl_main_import  # Python 3.11+
except ImportError:
    with contextlib.suppress(ImportError):
        import tomli as tl_fallback  # Fallback for Python < 3.11
finally:
    tomllib = tl_main_import or tl_fallback or None

# --- AST Visitor Logic ---


class ReturnVisitor(ast.NodeVisitor):
    """Visitor to find return statements with values in a function."""

    def __init__(self):
        self.returns_value = False
        self.function_depth = 0

    def visit_Return(self, node: ast.Return) -> None:
        # We only care about returns at the top level of a function.
        if self.function_depth == 1 and node.value is not None:
            # Check if it's explicitly returning None (ast.Constant for Python 3.8+)
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                return
            # Check for `return None` in older Python versions (ast.Name)
            if isinstance(node.value, ast.Name) and node.value.id == "None":
                return

            # If we reach here, it's a non-None return value.
            self.returns_value = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1


def function_returns_value(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function returns a value (not None)."""
    visitor = ReturnVisitor()
    visitor.visit(func_node)
    return visitor.returns_value


def has_return_annotation(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function has a return type annotation."""
    return func_node.returns is not None


def check_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Check a Python file for missing return annotations."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)
        return []

    errors = []

    class FunctionVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._check_function(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._check_function(node)
            self.generic_visit(node)

        def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            # Skip if function already has a return annotation
            if has_return_annotation(node):
                return

            # Skip if function doesn't return values
            if not function_returns_value(node):
                return

            # Skip dunder methods like __init__ which often return None implicitly
            if node.name.startswith("__") and node.name.endswith("__"):
                return

            # Report missing annotation
            errors.append(
                (
                    node.lineno,
                    node.name,
                    f"Function '{node.name}' returns a value but has no return type annotation",
                )
            )

    visitor = FunctionVisitor()
    visitor.visit(tree)
    return errors


# --- Configuration and File Collection Logic ---


def find_project_root(srcs: Tuple[Path, ...]) -> Path:
    """Find the root of the project by looking for pyproject.toml."""
    for src in srcs:
        path = src.resolve()
        if not path.is_dir():
            path = path.parent

        while path != path.parent:
            if (path / "pyproject.toml").is_file():
                return path
            path = path.parent
    return Path.cwd().resolve()


def load_pyproject_data(project_root: Path) -> Optional[Dict[str, Any]]:
    """Load the entire pyproject.toml file into a dictionary."""
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return None

    if tomllib is None:
        print(
            "Warning: 'tomli' is not installed, can't read pyproject.toml on Python < 3.11",
            file=sys.stderr,
        )
        return None

    with open(pyproject_path, "rb") as f:
        try:
            return tomllib.load(f)
        except Exception as e:
            print(f"Warning: Could not parse {pyproject_path}: {e}", file=sys.stderr)
            return None


def get_exclude_patterns(config: Optional[Dict], section: str, key: str) -> List[str]:
    """Safely extract a list of strings from a nested config dictionary."""
    if not config:
        return []

    keys = section.split(".")
    current_level = config
    for part in keys:
        if not isinstance(current_level, dict) or part not in current_level:
            return []
        current_level = current_level[part]

    if isinstance(current_level, dict):
        patterns = current_level.get(key, [])
        if isinstance(patterns, list):
            return [str(p) for p in patterns]
    return []


def collect_python_files(
    paths: List[Path], project_root: Path, exclude_patterns: List[str]
) -> List[Path]:
    """Collect all .py files from the given paths, applying exclude patterns."""
    collected_files: Set[Path] = set()
    built_in_exclude_dirs = {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "venv",
        "__pycache__",
        "build",
        "dist",
    }

    for path in paths:
        path = path.resolve()
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            continue

        all_py_files = path.rglob("*.py") if path.is_dir() else [path]

        for py_file in all_py_files:
            if py_file.suffix != ".py":
                continue

            try:
                relative_file_path = py_file.relative_to(project_root)
            except ValueError:
                relative_file_path = py_file

            if any(part in built_in_exclude_dirs for part in py_file.parts):
                continue

            if any(relative_file_path.match(pattern) for pattern in exclude_patterns):
                continue

            collected_files.add(py_file)

    return sorted(collected_files)


# --- Main Execution ---


def main():
    """Main entry point with argument parsing and advanced configuration handling."""
    parser = argparse.ArgumentParser(
        description="Check for missing return type annotations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "paths",
        metavar="path",
        type=Path,
        nargs="+",
        help="One or more files or directories to check.",
    )
    parser.add_argument(
        "--config-section",
        default=None,
        help="Override: Dotted path to the config section in pyproject.toml (e.g., 'tool.mypy').",
    )
    parser.add_argument(
        "--config-exclude-key",
        default=None,
        help="Override: Key for the exclude list within the config section (e.g., 'omit').",
    )
    args = parser.parse_args()

    # --- Configuration Logic with Precedence ---
    project_root = find_project_root(tuple(args.paths))
    pyproject_data = load_pyproject_data(project_root)

    # Defaults
    config_section = "tool.check-return-annotations"
    exclude_key = "exclude"

    # Precedence 3: Check for `exclude-from` in our tool's pyproject.toml section
    if pyproject_data:
        tool_config = get_exclude_patterns(
            pyproject_data, "tool", "check-return-annotations"
        )
        if isinstance(tool_config, dict) and "exclude-from" in tool_config:
            source_str = tool_config["exclude-from"]
            parts = source_str.rsplit(":", 1)
            config_section = parts[0]
            exclude_key = parts[1] if len(parts) > 1 else "exclude"
    # Precedence 1 & 2: Command-line arguments override everything
    if args.config_section is not None:
        config_section = args.config_section
    if args.config_exclude_key is not None:
        exclude_key = args.config_exclude_key

    # --- End Configuration Logic ---

    exclude_patterns = get_exclude_patterns(pyproject_data, config_section, exclude_key)

    files_to_check = collect_python_files(args.paths, project_root, exclude_patterns)

    if not files_to_check:
        print("No Python files found to check.")
        sys.exit(0)

    total_errors = 0
    for filepath in files_to_check:
        errors = check_file(filepath)
        for line_no, _func_name, message in errors:
            try:
                relative_path = filepath.relative_to(Path.cwd())
            except ValueError:
                relative_path = filepath
            print(f"{relative_path}:{line_no}: error: {message}")
            total_errors += 1

    if total_errors > 0:
        print(f"\nFound {total_errors} error(s)")
        sys.exit(1)
    else:
        print("No missing return annotations found")


if __name__ == "__main__":
    main()
