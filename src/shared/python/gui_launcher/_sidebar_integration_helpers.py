"""Internal helpers and reflection utilities for Tools Sidebar host integration."""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_SIDEBAR_MODULE_CANDIDATES = (
    "sidekick.ui.tools_sidebar",
    "shared.python.sidekick.ui.tools_sidebar",
)
_SIDEBAR_CLASS_CANDIDATES = (
    "ToolsSidebar",
    "UnifiedToolsSidebar",
    "ToolsSidebarWidget",
)
_FILE_OPEN_SIGNAL_CANDIDATES = (
    "file_open_requested",
    "open_file_requested",
    "fileRequested",
    "openRequested",
)
_FILE_OPEN_METHOD_CANDIDATES = (
    "open_file",
    "load_file",
    "_open_file",
    "_load_file",
    "load_data_file",
)


def import_sidebar_module() -> Any | None:
    """Attempt importing the candidate tools sidebar modules."""
    for module_name in _SIDEBAR_MODULE_CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def create_sidebar_from_module(
    module: Any,
    *,
    main_window: Any,
    project_root: Path | None,
    context_provider: Callable[[], Any] | None,
) -> Any | None:
    """Create a sidebar instance using module factory or standard class names."""
    factory = getattr(module, "create_tools_sidebar", None)
    if callable(factory):
        return call_sidebar_factory(
            factory,
            main_window=main_window,
            project_root=project_root,
            context_provider=context_provider,
        )

    for class_name in _SIDEBAR_CLASS_CANDIDATES:
        sidebar_class = getattr(module, class_name, None)
        if callable(sidebar_class):
            return call_sidebar_factory(
                sidebar_class,
                main_window=main_window,
                project_root=project_root,
                context_provider=context_provider,
            )
    return None


def sidebar_factory_kwargs(
    *,
    main_window: Any,
    project_root: Path | None,
    context_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    return {
        "parent": main_window,
        "project_root": project_root,
        "context_provider": context_provider,
        "sidekick_tokens": get_sidekick_tokens(),
    }


def get_sidekick_tokens() -> dict[str, str]:
    try:
        from src.shared.python.theme.sidekick_tokens import get_current_sidekick_tokens

        return cast(dict[str, str], get_current_sidekick_tokens())
    except Exception:  # noqa: BLE001 - sidebar startup must stay optional
        return {}


def call_shared_installer(
    installer: Callable[..., Any],
    *,
    main_window: Any,
    project_root: Path | None,
    context_provider: Callable[[], Any] | None,
) -> Any:
    kwargs: dict[str, Any] = sidebar_factory_kwargs(
        main_window=main_window,
        project_root=project_root,
        context_provider=context_provider,
    )
    try:
        signature = inspect.signature(installer)
    except (TypeError, ValueError):
        return installer(
            main_window, project_root=project_root, context_provider=context_provider
        )

    accepted = set(signature.parameters)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    call_kwargs = {
        key: value for key, value in kwargs.items() if accepts_kwargs or key in accepted
    }
    return installer(main_window, **call_kwargs)


def call_sidebar_factory(
    factory: Callable[..., Any],
    *,
    main_window: Any,
    project_root: Path | None,
    context_provider: Callable[[], Any] | None,
) -> Any:
    kwargs = sidebar_factory_kwargs(
        main_window=main_window,
        project_root=project_root,
        context_provider=context_provider,
    )
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(
            **{key: value for key, value in kwargs.items() if value is not None}
        )

    accepted = {
        name
        for name, param in signature.parameters.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        call_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    else:
        call_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
    return factory(**call_kwargs)


def ensure_dock_widget(sidebar: Any, main_window: Any) -> Any:
    if looks_like_dock_widget(sidebar):
        return sidebar

    try:
        from PyQt6.QtWidgets import QDockWidget
    except ImportError as exc:
        raise RuntimeError("PyQt6 is required to wrap sidebar widgets") from exc

    dock = QDockWidget("Tools", main_window)
    dock.setObjectName("unifiedToolsSidebarDock")
    dock.setWidget(sidebar)
    return dock


def looks_like_dock_widget(widget: Any) -> bool:
    return hasattr(widget, "setWidget") and hasattr(widget, "toggleViewAction")


def add_dock_widget(main_window: Any, dock: Any) -> None:
    main_window.addDockWidget(right_dock_area(), dock)


def right_dock_area() -> Any:
    try:
        from PyQt6.QtCore import Qt

        return Qt.DockWidgetArea.RightDockWidgetArea
    except ImportError:
        return "right"


def connect_file_open_request(sidebar: Any, main_window: Any) -> bool:
    signal_owner, signal = find_signal(sidebar)
    if signal is None and looks_like_dock_widget(sidebar):
        widget = sidebar.widget() if hasattr(sidebar, "widget") else None
        signal_owner, signal = find_signal(widget)
    if signal is None:
        return False

    handler = find_file_open_handler(main_window)
    if handler is None:
        handler = build_status_handler(main_window)

    try:
        signal.connect(handler)
    except (AttributeError, TypeError) as exc:
        logger.debug("Could not connect sidebar file-open signal: %s", exc)
        return False

    logger.debug(
        "Connected tools sidebar file-open signal from %s",
        type(signal_owner).__name__,
    )
    return True


def find_signal(candidate: Any) -> tuple[Any | None, Any | None]:
    if candidate is None:
        return None, None
    for attr_name in _FILE_OPEN_SIGNAL_CANDIDATES:
        signal = getattr(candidate, attr_name, None)
        if hasattr(signal, "connect"):
            return candidate, signal
    return None, None


def find_file_open_handler(main_window: Any) -> Callable[[Any], Any] | None:
    candidates = [main_window]
    central_widget = maybe_call(main_window, "centralWidget")
    if central_widget is not None:
        candidates.append(central_widget)
    inner_window = maybe_call(main_window, "inner_main_window")
    if inner_window is not None:
        candidates.append(inner_window)

    for target in candidates:
        for method_name in _FILE_OPEN_METHOD_CANDIDATES:
            method = getattr(target, method_name, None)
            if callable(method) and callable_accepts_path(method):
                return cast(Callable[[Any], Any], method)
    return None


def callable_accepts_path(method: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return True

    positional = [
        param
        for param in signature.parameters.values()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in signature.parameters.values()
    ) or bool(positional)


def build_status_handler(main_window: Any) -> Callable[[Any], None]:
    def _handle_requested_file(path: Any) -> None:
        message = f"Tools sidebar requested file open: {path}"
        status_bar = maybe_call(main_window, "statusBar")
        if status_bar is not None and hasattr(status_bar, "showMessage"):
            try:
                status_bar.showMessage(message, 5000)
                return
            except TypeError:
                status_bar.showMessage(message)
                return
        logger.info(message)

    return _handle_requested_file


def maybe_call(obj: Any, method_name: str) -> Any | None:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except TypeError:
        return None


def status_sidebar(installed: Any) -> Any:
    if installed is None:
        return None
    sidebar = getattr(installed, "sidebar", None)
    if sidebar is not None:
        return sidebar
    if hasattr(installed, "widget") and callable(installed.widget):
        try:
            return installed.widget()
        except TypeError:
            return installed
    return installed
