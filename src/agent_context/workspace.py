"""Content fingerprints for one checkout, including working tree changes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .paths import CatalogError, file_hash, safe_path

INDEX_VERSION = 1
MAX_FILES = 20_000


def git(root: Path, *args: str, optional: bool = False) -> str:
    """Run bounded read-only Git commands using an explicit checkout."""
    executable = shutil.which("git")
    if executable is None:
        raise CatalogError("Git is required to identify the current checkout")
    try:
        result = subprocess.run(
            [executable, "-C", str(root), *args],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CatalogError(f"Cannot inspect Git checkout: {exc}") from exc
    if result.returncode and not optional:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise CatalogError(f"Git inspection failed: {error}")
    return (
        result.stdout.decode("utf-8", errors="strict").strip()
        if result.returncode == 0
        else ""
    )


def digest(value: Any) -> str:
    """Hash canonical serialized data; timestamps never establish freshness."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_files(catalog: Catalog) -> list[str]:
    """Expand registered directories over tracked and nonignored new files."""
    roots = {p for c in catalog.components for p in c.sources}
    files = {catalog.path}
    for component in catalog.components:
        files.update(component.documentation + component.tests)
        files.update(ref.path for ref in component.entrypoints)
    for relation in catalog.relations:
        files.update((relation.contract, *relation.inputs, *relation.tests))
    files.update(i["path"] for i in catalog.inventories)
    candidates = git(
        catalog.root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    ).split("\0")
    for candidate in candidates:
        if candidate and any(
            candidate == p or candidate.startswith(p + "/") for p in roots
        ):
            # Deleted tracked files are absent from the current snapshot.
            if (catalog.root / candidate).exists():
                files.add(candidate)
    for root in roots:
        if (catalog.root / root).is_file():
            files.add(root)
    if len(files) > MAX_FILES:
        raise CatalogError(
            f"Registered corpus exceeds {MAX_FILES} files; narrow component sources"
        )
    for file in files:
        safe_path(catalog.root, file)
    return sorted(files)


def dependency_state(catalog: Catalog) -> dict[str, dict[str, str]]:
    """Report both pinned gitlinks and actual checked-out dependency state."""
    result = {}
    for path in catalog.dependencies:
        record = git(catalog.root, "ls-files", "--stage", "--", path)
        fields = record.split()
        if len(fields) < 3 or fields[0] != "160000":
            raise CatalogError(f"Dependency is not a pinned Git submodule: {path}")
        child = safe_path(catalog.root, path, directory=True)
        if not (child / ".git").exists():
            raise CatalogError(f"Dependency is not initialized: {path}")
        actual = git(child, "rev-parse", "HEAD")
        dirty = git(child, "status", "--porcelain", "--untracked-files=normal")
        provider = child / "src/agent_context"
        runtime_matches = not provider.is_dir() or implementation_id(
            provider
        ) == implementation_id(Path(__file__).parent)
        result[path] = {
            "pinned": fields[1],
            "checkout": actual,
            "runtime": "matched" if runtime_matches else "different-context-package",
            "state": "verified"
            if actual == fields[1] and not dirty and runtime_matches
            else "unverified",
        }
    return result


def implementation_id(folder: Path) -> str:
    """Match installed context code to its pinned provider, independent of location."""
    return digest(
        {
            p.name: file_hash(p)
            for p in sorted(folder.iterdir())
            if p.suffix in {".py", ".html"}
        }
    )


def snapshot(catalog: Catalog) -> dict[str, Any]:
    """Read the current registered corpus; never infer validity from HEAD alone."""
    head_before = git(catalog.root, "rev-parse", "--verify", "HEAD", optional=True)
    files = {p: file_hash(safe_path(catalog.root, p)) for p in source_files(catalog)}
    if files[catalog.path] != catalog.source_hash:
        raise CatalogError("Catalog changed during context inspection; retry")
    dependencies = dependency_state(catalog)
    second_read = {
        p: file_hash(safe_path(catalog.root, p)) for p in source_files(catalog)
    }
    head_after = git(catalog.root, "rev-parse", "--verify", "HEAD", optional=True)
    if head_before != head_after or files != second_read:
        raise CatalogError("Checkout changed during context inspection; retry")
    implementation = implementation_id(Path(__file__).parent)
    return {
        "version": INDEX_VERSION,
        "root": str(catalog.root),
        "commit": head_after or None,
        "implementation": implementation,
        "digest": digest(
            {
                "version": INDEX_VERSION,
                "implementation": implementation,
                "files": files,
                "dependencies": dependencies,
            }
        ),
        "files": files,
        "dependencies": dependencies,
        "coverage": {
            "files": len(files),
            "scope": "registered sources only",
            "errors": [],
        },
    }
