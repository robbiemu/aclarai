#!/usr/bin/env python3
"""
Script to discover test directories from pyproject.toml workspace configuration.
Usage: python -m tools.discover_tests <format>
"""

import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List


def load_pyproject() -> Dict[str, Any]:
    """Load and parse pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found in current directory")

    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


def get_workspace_members(config: Dict[str, Any]) -> List[str]:
    """Extract workspace members from pyproject.toml."""
    try:
        members = config["tool"]["uv"]["workspace"]["members"]
    except KeyError as e:
        raise KeyError("No workspace members found in pyproject.toml") from e

    if not isinstance(members, list):
        raise TypeError("Workspace members must be a list")

    if not all(isinstance(m, str) for m in members):
        raise TypeError("All workspace members must be strings")

    return members


def find_test_directories(workspace_members: List[str]) -> List[str]:
    """Find test directories that actually exist"""
    test_dirs = []

    # Check for root tests directory
    if Path("tests").is_dir():
        test_dirs.append("tests/")

    # Check for test directories in each workspace member
    for member in workspace_members:
        member_path = Path(member)
        if member_path.is_dir():
            tests_path = member_path / "tests"
            if tests_path.is_dir():
                test_dirs.append(f"{member}/tests/")

    return test_dirs


def get_excluded_dirs(config: Dict[str, Any]) -> List[str]:
    """Get excluded directories from mypy configuration."""
    try:
        excludes = config["tool"]["mypy"]["exclude"]
    except KeyError:
        return []

    if not isinstance(excludes, list):
        raise TypeError("Exclude must be a list of strings")

    if not all(isinstance(e, str) for e in excludes):
        raise TypeError("All exclude entries must be strings")

    return excludes


def filter_test_directories(
    test_dirs: List[str], excluded_dirs: List[str]
) -> List[str]:
    """Filter out excluded test directories"""
    # Normalize excluded directories for comparison
    normalized_excluded = [dir.rstrip("/") + "/" for dir in excluded_dirs]

    filtered_dirs = []
    for test_dir in test_dirs:
        if test_dir not in normalized_excluded:
            filtered_dirs.append(test_dir)

    return filtered_dirs


def get_coverage_source_for_test_dir(test_dir: str) -> str:
    """Determine the appropriate coverage source for a test directory"""
    coverage_sources = {
        "shared/tests/": "aclarai_shared",
        "services/aclarai-core/tests/": "aclarai_core",
        "services/aclarai-ui/tests/": "aclarai_ui",
        "services/scheduler/tests/": "aclarai_scheduler",
        "services/vault-watcher/tests/": "aclarai_vault_watcher",
    }
    return coverage_sources.get(test_dir, ".")


def main():
    """Main function to discover and output test directories"""
    if len(sys.argv) < 2:
        print("Usage: python -m tools.discover_tests <format>")
        print("Formats: github-matrix, pytest-args, list, coverage-mapping")
        sys.exit(1)

    output_format = sys.argv[1]

    try:
        config = load_pyproject()
        workspace_members = get_workspace_members(config)
        all_test_dirs = find_test_directories(workspace_members)
        excluded_dirs = get_excluded_dirs(config)
        test_dirs = filter_test_directories(all_test_dirs, excluded_dirs)

        if not test_dirs:
            print("No test directories found", file=sys.stderr)
            sys.exit(1)

        if output_format == "github-matrix":
            # Output for GitHub Actions matrix strategy
            import json

            matrix = {"test-dir": test_dirs}
            print(json.dumps(matrix))
        elif output_format == "pytest-args":
            # Output space-separated for pytest command
            print(" ".join(test_dirs))
        elif output_format == "list":
            # Output one per line
            for test_dir in test_dirs:
                print(test_dir)
        elif output_format == "coverage-mapping":
            # Output test-dir:coverage-source mapping for CI
            for test_dir in test_dirs:
                coverage_source = get_coverage_source_for_test_dir(test_dir)
                print(f"{test_dir}:{coverage_source}")
        else:
            print(f"Unknown format: {output_format}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
