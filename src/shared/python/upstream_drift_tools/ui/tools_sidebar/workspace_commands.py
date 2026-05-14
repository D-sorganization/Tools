"""Explicit, bounded workspace commands for Sidekick runtime tabs."""

from __future__ import annotations

import ast
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calculator_workspace import (
    CALCULATOR_WORKSPACE_SCOPE,
    GLOBAL_WORKSPACE_SCOPE,
    CalculatorWorkspaceController,
    CalculatorWorkspaceFacade,
    CalculatorWorkspaceLoadResult,
    validate_calculator_workspace_path,
)
from .registry import WorkspaceRegistry

_ASSIGNMENT_RE = re.compile(
    r"^(local|global)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class WorkspaceCommandResult:
    """User-facing result of a bounded workspace command."""

    message: str
    mutated: bool = False
    scope: str | None = None


class WorkspaceCommandExecutor:
    """Execute limited workspace commands without arbitrary code evaluation."""

    def __init__(
        self,
        *,
        workspace: CalculatorWorkspaceFacade,
        local_controller: CalculatorWorkspaceController,
        global_registry: WorkspaceRegistry,
        global_storage_path: str | Path,
    ) -> None:
        if workspace is None:
            raise ValueError("workspace must be provided")
        if local_controller is None:
            raise ValueError("local_controller must be provided")
        if global_registry is None:
            raise ValueError("global_registry must be provided")
        self._workspace = workspace
        self._local_controller = local_controller
        self._global_controller = _GlobalWorkspaceController(
            global_registry,
            storage_path=global_storage_path,
        )

    def execute(self, command: str) -> WorkspaceCommandResult:
        """Execute one explicit workspace command."""
        normalized = _normalize_command(command)
        assignment = _ASSIGNMENT_RE.match(normalized)
        if assignment is not None:
            return self._execute_assignment(assignment)
        try:
            tokens = shlex.split(normalized)
        except ValueError as exc:
            raise ValueError("Unsupported workspace command syntax") from exc
        if not tokens:
            raise ValueError("Unsupported workspace command")
        action = tokens[0].lower()
        if action == "show":
            return self._show(tokens)
        if action == "delete":
            return self._delete(tokens)
        if action == "clear":
            return self._clear(tokens)
        if action == "save":
            return self._save(tokens)
        if action == "load":
            return self._load(tokens)
        raise ValueError("Unsupported workspace command")

    def _execute_assignment(self, match: re.Match[str]) -> WorkspaceCommandResult:
        scope = _normalize_scope(match.group(1))
        name = _normalize_name(match.group(2))
        try:
            value = ast.literal_eval(match.group(3))
        except (SyntaxError, ValueError) as exc:
            raise ValueError("assignment value must be a Python literal") from exc
        variable = self._target_registry(scope).set(name, value)
        return WorkspaceCommandResult(
            message=f"{scope} {name} = {variable.preview}",
            mutated=True,
            scope=scope,
        )

    def _show(self, tokens: list[str]) -> WorkspaceCommandResult:
        if len(tokens) != 3:
            raise ValueError("show command must be: show <scope> <name>")
        scope = _normalize_scope(tokens[1])
        name = _normalize_name(tokens[2])
        variable = self._target_registry(scope).describe(name)
        details = [_display_summary(variable)]
        if variable.dtype:
            details.append(variable.dtype)
        if variable.size is not None:
            details.append(f"size={variable.size}")
        return WorkspaceCommandResult(
            message=(
                f"{scope} {name}: {variable.type_name} "
                f"({', '.join(details)}) {variable.preview}"
            ),
            scope=scope,
        )

    def _delete(self, tokens: list[str]) -> WorkspaceCommandResult:
        if len(tokens) < 4:
            raise PermissionError("delete requires confirm")
        if len(tokens) != 4:
            raise ValueError("delete command must be: delete <scope> <name> confirm")
        scope = _normalize_scope(tokens[1])
        name = _normalize_name(tokens[2])
        if tokens[3].lower() != "confirm":
            raise PermissionError("delete requires confirm")
        removed = self._target_registry(scope).remove(name)
        return WorkspaceCommandResult(
            message=(
                f"Removed {scope} {name}."
                if removed
                else f"{scope} {name} was not set."
            ),
            mutated=removed,
            scope=scope,
        )

    def _clear(self, tokens: list[str]) -> WorkspaceCommandResult:
        if len(tokens) < 3:
            raise PermissionError("clear requires confirm")
        if len(tokens) != 3:
            raise ValueError("clear command must be: clear <scope> confirm")
        scope = _normalize_scope(tokens[1])
        if tokens[2].lower() != "confirm":
            raise PermissionError("clear requires confirm")
        if scope == GLOBAL_WORKSPACE_SCOPE:
            self._workspace.global_registry.clear()
        else:
            self._local_controller.clear(confirm_clear=True)
        return WorkspaceCommandResult(
            message=f"Cleared {scope} workspace.",
            mutated=True,
            scope=scope,
        )

    def _save(self, tokens: list[str]) -> WorkspaceCommandResult:
        scope, path = _parse_path_command(tokens, action="save", keyword="to")
        saved = self._controller(scope).save(path)
        return WorkspaceCommandResult(
            message=f"Workspace saved: {saved}",
            scope=scope,
        )

    def _load(self, tokens: list[str]) -> WorkspaceCommandResult:
        scope, path, replace = _parse_load_command(tokens)
        result = self._controller(scope).load(
            path,
            replace=replace,
            confirm_replace=replace,
        )
        return WorkspaceCommandResult(
            message=result.summary,
            mutated=bool(result.variables) or replace,
            scope=scope,
        )

    def _target_registry(self, scope: str) -> WorkspaceRegistry:
        if scope == GLOBAL_WORKSPACE_SCOPE:
            return self._workspace.global_registry
        if scope == CALCULATOR_WORKSPACE_SCOPE:
            return self._workspace.local_registry
        raise ValueError(f"Unsupported workspace scope: {scope}")

    def _controller(self, scope: str) -> Any:
        if scope == GLOBAL_WORKSPACE_SCOPE:
            return self._global_controller
        if scope == CALCULATOR_WORKSPACE_SCOPE:
            return self._local_controller
        raise ValueError(f"Unsupported workspace scope: {scope}")


class _GlobalWorkspaceController:
    """Persist and restore the shared global workspace."""

    def __init__(
        self,
        registry: WorkspaceRegistry,
        *,
        storage_path: str | Path,
    ) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        self._registry = registry
        self._storage_path = Path(storage_path)

    def save(self, path: str | Path | None = None) -> Path:
        target = validate_calculator_workspace_path(path or self._storage_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._registry.to_dict()
        payload["scope"] = GLOBAL_WORKSPACE_SCOPE
        temp = target.with_name(f".{target.name}.tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(target)
        return target

    def load(
        self,
        path: str | Path | None = None,
        *,
        replace: bool = False,
        confirm_replace: bool = False,
    ) -> CalculatorWorkspaceLoadResult:
        if replace and not confirm_replace:
            raise PermissionError("replace load requires explicit confirmation")
        source = validate_calculator_workspace_path(path or self._storage_path)
        incoming = _load_global_registry(source)
        imported = tuple(incoming.variables())
        if replace:
            self._registry.clear()
        for variable in imported:
            self._registry.set(variable.name, incoming.get(variable.name))
        return CalculatorWorkspaceLoadResult(imported, replaced=replace)


def _load_global_registry(path: Path) -> WorkspaceRegistry:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("workspace file is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("workspace file must contain an object")
    if payload.get("scope") != GLOBAL_WORKSPACE_SCOPE:
        raise ValueError("workspace scope must be global")
    try:
        return WorkspaceRegistry.load_json(path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("workspace file is not valid JSON") from exc


def _normalize_command(command: str) -> str:
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    normalized = command.strip()
    if not normalized:
        raise ValueError("command must not be blank")
    return normalized


def _normalize_scope(scope: str) -> str:
    normalized = scope.strip().lower()
    if normalized == "local":
        return CALCULATOR_WORKSPACE_SCOPE
    if normalized == "global":
        return GLOBAL_WORKSPACE_SCOPE
    raise ValueError(f"Unsupported workspace scope: {scope}")


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not _NAME_RE.fullmatch(normalized):
        raise ValueError("workspace variable names must be identifier-like")
    return normalized


def _parse_path_command(
    tokens: list[str],
    *,
    action: str,
    keyword: str,
) -> tuple[str, str | Path | None]:
    if len(tokens) not in {2, 4}:
        raise ValueError(f"{action} command syntax is invalid")
    scope = _normalize_scope(tokens[1])
    if len(tokens) == 2:
        return scope, None
    if tokens[2].lower() != keyword:
        raise ValueError(f"{action} command syntax is invalid")
    return scope, tokens[3]


def _parse_load_command(tokens: list[str]) -> tuple[str, str | Path | None, bool]:
    if len(tokens) < 2:
        raise ValueError("load command syntax is invalid")
    scope = _normalize_scope(tokens[1])
    path: str | Path | None = None
    replace = False
    index = 2
    while index < len(tokens):
        token = tokens[index].lower()
        if token == "from" and index + 1 < len(tokens):
            path = tokens[index + 1]
            index += 2
            continue
        if token == "replace":
            replace = True
            index += 1
            continue
        if token == "confirm":
            index += 1
            continue
        raise ValueError("load command syntax is invalid")
    return scope, path, replace


def _display_summary(variable: Any) -> str:
    shape = getattr(variable, "shape", None)
    if shape is not None and len(shape) == 1:
        return f"length={shape[0]}"
    return str(variable.summary)
