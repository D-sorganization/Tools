"""Transaction logging system for rollback capability."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TransactionLog:
    """Logs all file operations for potential rollback."""

    def __init__(self, log_path: Path | None = None):
        """
        Initialize transaction log.

        Args:
            log_path: Path to transaction log file. If None, uses default.
        """
        if log_path is None:
            log_path = Path.cwd() / "pdf_renamer_transactions.jsonl"
        self.log_path = log_path
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_rename(
        self, original_path: Path, new_path: Path, success: bool, error: str = ""
    ) -> None:
        """
        Log a rename operation.

        Args:
            original_path: Original file path
            new_path: New file path
            success: Whether operation succeeded
            error: Error message if failed
        """
        entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "operation": "rename",
            "original_path": str(original_path),
            "new_path": str(new_path),
            "success": success,
            "error": error,
        }
        self._write_entry(entry)

    def log_delete(self, file_path: Path, success: bool, error: str = "") -> None:
        """
        Log a delete operation.

        Args:
            file_path: Path to deleted file
            success: Whether operation succeeded
            error: Error message if failed
        """
        entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "operation": "delete",
            "file_path": str(file_path),
            "success": success,
            "error": error,
        }
        self._write_entry(entry)

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a log entry to the transaction log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except (IOError, PermissionError, OSError) as e:
            logger.error(f"Failed to write transaction log: {e}")

    def get_session_operations(
        self, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve all operations from a session.

        Args:
            session_id: Session ID to retrieve. If None, uses current session.

        Returns:
            List of operation dictionaries
        """
        if session_id is None:
            session_id = self.session_id

        operations: list[dict[str, Any]] = []
        if not self.log_path.exists():
            return operations

        try:
            with open(self.log_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("session_id") == session_id:
                            operations.append(entry)
                    except json.JSONDecodeError:
                        continue
        except (IOError, PermissionError, OSError) as e:
            logger.error(f"Failed to read transaction log: {e}")

        return operations

    def rollback_session(
        self, session_id: str | None = None, dry_run: bool = True
    ) -> None:
        """
        Rollback all successful operations from a session.

        Args:
            session_id: Session ID to rollback. If None, uses current session.
            dry_run: If True, only show what would be rolled back
        """
        operations = self.get_session_operations(session_id)

        # Reverse the operations to undo in reverse order
        for op in reversed(operations):
            if not op.get("success"):
                continue

            if op["operation"] == "rename":
                original = Path(op["original_path"])
                new = Path(op["new_path"])

                if new.exists():
                    if dry_run:
                        logger.info(f"[DRY RUN] Would rename {new} -> {original}")
                    else:
                        try:
                            new.rename(original)
                            logger.info(f"Rolled back: {new} -> {original}")
                        except OSError as e:
                            logger.error(f"Failed to rollback rename: {e}")
                else:
                    logger.warning(f"Cannot rollback: {new} doesn't exist")

            elif op["operation"] == "delete":
                logger.warning(
                    f"Cannot rollback delete: {op['file_path']} "
                    "(file permanently deleted)"
                )
