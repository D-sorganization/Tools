"""SQLite-backed event logger and PyQt6 log viewer widget.

Saves and queries system events (clicks, setpoints, alarms, login/logout).
"""

from __future__ import annotations

import csv
import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timedelta
from typing import Any

from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_INSERT_SQL = """
    INSERT INTO event_logs (
        timestamp, event_type, severity, operator, description, details
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """


class BatchedEventWriter:
    """Serialise event rows onto a background thread with one open connection.

    The desktop HMI used to open a fresh SQLite connection, insert and
    fsync-commit on the Qt GUI thread for *every* alarm transition. With a tag
    dithering on its trip point that is ~10 commits/second of blocking I/O on
    the thread that repaints the E-stop button (issue #4022).

    Rows submitted here are queued (non-blocking for the caller), drained by a
    daemon thread, and committed in batches through a single persistent
    connection.
    """

    def __init__(
        self,
        db_path: str,
        flush_interval_s: float = 0.25,
        max_batch: int = 200,
    ) -> None:
        """Start the writer thread.

        Args:
            db_path: SQLite file to write to.
            flush_interval_s: Maximum time a queued row waits before commit.
            max_batch: Maximum rows coalesced into one transaction.

        Raises:
            TypeError: If ``db_path`` is not a string.
            ValueError: If ``flush_interval_s`` or ``max_batch`` is not > 0.
        """
        if not isinstance(db_path, str):
            raise TypeError(f"db_path must be a str, got {type(db_path).__name__}")
        if flush_interval_s <= 0:
            raise ValueError(f"flush_interval_s must be > 0, got {flush_interval_s}")
        if max_batch <= 0:
            raise ValueError(f"max_batch must be > 0, got {max_batch}")

        self.db_path = db_path
        self.flush_interval_s = float(flush_interval_s)
        self.max_batch = int(max_batch)

        self._queue: queue.Queue[tuple[Any, ...] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._drained = threading.Condition(self._lock)
        self._pending = 0
        self._stopping = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="p1am-event-log-writer", daemon=True
        )
        self.thread.start()

    def submit(self, row: tuple[Any, ...]) -> None:
        """Queue one already-validated row. Never blocks the caller.

        Raises:
            TypeError: If ``row`` is not a 6-tuple.
        """
        if not isinstance(row, tuple) or len(row) != 6:
            raise TypeError("row must be a 6-tuple of insert parameters")
        if self._stopping.is_set():
            raise RuntimeError("writer is stopped")
        with self._lock:
            self._pending += 1
        self._queue.put(row)

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until every queued row has been committed.

        Returns:
            ``True`` if the writer went idle within ``timeout``.
        """
        with self._drained:
            return self._drained.wait_for(lambda: self._pending <= 0, timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """Drain and shut the writer thread down."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        self._queue.put(None)
        self.thread.join(timeout)

    def _run(self) -> None:
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            while True:
                item = self._queue.get()
                if item is None:
                    break
                batch = [item]
                stopping = False
                deadline = time.monotonic() + self.flush_interval_s
                while len(batch) < self.max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        nxt = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if nxt is None:
                        stopping = True
                        break
                    batch.append(nxt)
                self._commit(conn, batch)
                self._settle(len(batch))
                if stopping:
                    return
        except Exception:  # pragma: no cover - background thread safety net
            logger.exception("Event-log writer thread failed")
        finally:
            self._settle(self._pending)
            if conn is not None:
                conn.close()

    def _settle(self, completed: int) -> None:
        """Decrement the in-flight count and wake any waiting :meth:`flush`."""
        with self._drained:
            self._pending = max(0, self._pending - max(0, completed))
            if self._pending <= 0:
                self._drained.notify_all()

    @staticmethod
    def _commit(conn: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
        try:
            conn.executemany(_INSERT_SQL, batch)
            conn.commit()
        except Exception:
            logger.exception("Failed to persist %d event rows", len(batch))
            try:
                conn.rollback()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Event-log rollback failed")


class EventLogger:
    """Manages the SQLite database for event logs, executing parameterized queries."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the database and ensure tables are created.

        Args:
            db_path: Path to the SQLite database file. Defaults to EVENT_LOG_DB_PATH
                     or 'p1am_event_log.db' in the current directory.
        """
        if db_path is None:
            db_path = os.environ.get("EVENT_LOG_DB_PATH", "p1am_event_log.db")
        self.db_path = db_path
        self._writer: BatchedEventWriter | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Create the event logs table if it does not already exist."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    operator TEXT,
                    description TEXT NOT NULL,
                    details TEXT
                )
                """)
            conn.commit()

    def log_event(
        self,
        event_type: str,
        severity: str,
        operator: str | None,
        description: str,
        details: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Insert a log record into the SQLite database.

        Uses parameterized queries to prevent SQL injection.

        Args:
            event_type: Category of event (e.g. 'button_click').
            severity: Importance levels ('INFO', 'WARNING', 'ERROR', 'CRITICAL').
            operator: Name or role of the active user.
            description: Narrative summary of the event.
            details: Optional stringified metadata/parameters.
            timestamp: Event time. Defaults to now.
        """
        row = self._build_row(
            event_type, severity, operator, description, details, timestamp
        )

        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(_INSERT_SQL, row)
            conn.commit()

    @staticmethod
    def _build_row(
        event_type: str,
        severity: str,
        operator: str | None,
        description: str,
        details: str | None,
        timestamp: datetime | None,
    ) -> tuple[Any, ...]:
        """Validate and normalise one insert row.

        Raises:
            ValueError: If ``event_type``, ``severity`` or ``description`` is
                empty.
        """
        if not event_type or not severity or not description:
            raise ValueError(
                "event_type, severity, and description must be non-empty strings"
            )
        stamp = datetime.now() if timestamp is None else timestamp
        return (
            stamp.isoformat(),
            event_type,
            severity,
            operator,
            description,
            details,
        )

    @property
    def async_writer(self) -> BatchedEventWriter:
        """The lazily-started background writer backing :meth:`log_event_async`."""
        if self._writer is None:
            self._writer = BatchedEventWriter(self.db_path)
        return self._writer

    def log_event_async(
        self,
        event_type: str,
        severity: str,
        operator: str | None,
        description: str,
        details: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Queue an event for batched insertion on the writer thread.

        Same validation and semantics as :meth:`log_event`, but the SQLite
        connect/insert/commit never runs on the caller's thread. Use this from
        GUI slots; use :meth:`log_event` when the row must be readable
        immediately.

        Raises:
            ValueError: If ``event_type``, ``severity`` or ``description`` is
                empty.
        """
        row = self._build_row(
            event_type, severity, operator, description, details, timestamp
        )
        self.async_writer.submit(row)

    def flush_async(self, timeout: float = 5.0) -> bool:
        """Block until queued async events have been committed."""
        if self._writer is None:
            return True
        return self._writer.flush(timeout)

    def close(self) -> None:
        """Drain and stop the background writer. Idempotent."""
        if self._writer is not None:
            self._writer.stop()
            self._writer = None

    def purge_older_than(self, retention_days: int) -> int:
        """Delete event rows older than ``retention_days``.

        A 10 Hz plant logs enough alarm traffic to grow the SQLite file without
        bound on a Raspberry Pi's SD card; the History tab's full-table requery
        then gets slower forever (issue #4022).

        Args:
            retention_days: Age threshold in days. ``0`` deletes everything
                older than now.

        Returns:
            Number of rows deleted.

        Raises:
            TypeError: If ``retention_days`` is not an int.
            ValueError: If ``retention_days`` is negative.
        """
        if isinstance(retention_days, bool) or not isinstance(retention_days, int):
            raise TypeError(
                f"retention_days must be an int, got {type(retention_days).__name__}"
            )
        if retention_days < 0:
            raise ValueError(f"retention_days must be >= 0, got {retention_days}")

        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM event_logs WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
        return max(0, deleted)

    def log_button_click(self, operator: str | None, button_name: str) -> None:
        """Log a button click event."""
        self.log_event(
            event_type="button_click",
            severity="INFO",
            operator=operator,
            description=f"Button clicked: {button_name}",
        )

    def log_setpoint_modification(
        self,
        operator: str | None,
        setpoint_name: str,
        old_value: float,
        new_value: float,
    ) -> None:
        """Log a setpoint modification event."""
        self.log_event(
            event_type="setpoint_modification",
            severity="INFO",
            operator=operator,
            description=(
                f"Setpoint '{setpoint_name}' changed from {old_value} to {new_value}"
            ),
            details=f"old_value={old_value}, new_value={new_value}",
        )

    def log_alarm_trip(
        self, alarm_name: str, condition: str, severity: str = "WARNING"
    ) -> None:
        """Log an alarm trip event."""
        self.log_event(
            event_type="alarm_trip",
            severity=severity,
            operator=None,
            description=f"Alarm tripped: {alarm_name}",
            details=f"condition={condition}",
        )

    def log_alarm_acknowledgment(self, operator: str | None, alarm_name: str) -> None:
        """Log an alarm acknowledgment event."""
        self.log_event(
            event_type="alarm_acknowledgment",
            severity="INFO",
            operator=operator,
            description=f"Alarm acknowledged: {alarm_name}",
        )

    def log_login(self, operator: str) -> None:
        """Log operator login event."""
        self.log_event(
            event_type="operator_login",
            severity="INFO",
            operator=operator,
            description=f"Operator '{operator}' logged in",
        )

    def log_logout(self, operator: str) -> None:
        """Log operator logout event."""
        self.log_event(
            event_type="operator_logout",
            severity="INFO",
            operator=operator,
            description=f"Operator '{operator}' logged out",
        )

    def fetch_logs(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        severity: str | None = None,
        event_type: str | None = None,
        keyword: str | None = None,
    ) -> list[tuple[int, str, str, str, str | None, str, str | None]]:
        """Fetch logs from SQLite database with optional filters.

        Uses parameterized queries to prevent SQL injection.

        Args:
            start_date: Filter logs starting from this date.
            end_date: Filter logs up to this date.
            severity: Filter logs by exact severity.
            event_type: Filter logs by exact event type.
            keyword: Case-insensitive search on description, details, or operator.

        Returns:
            List of log rows (tuples).
        """
        query = (
            "SELECT id, timestamp, event_type, severity, operator, "
            "description, details FROM event_logs WHERE 1=1"
        )
        params: list[Any] = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        if severity and severity != "All":
            query += " AND severity = ?"
            params.append(severity)
        if event_type and event_type != "All":
            query += " AND event_type = ?"
            params.append(event_type)
        if keyword:
            query += " AND (description LIKE ? OR details LIKE ? OR operator LIKE ?)"
            like_pattern = f"%{keyword}%"
            params.extend([like_pattern, like_pattern, like_pattern])

        query += " ORDER BY timestamp DESC"

        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def get_unique_event_types(self) -> list[str]:
        """Retrieve list of unique event types stored in the database."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT event_type FROM event_logs ORDER BY event_type ASC"
            )
            return [row[0] for row in cursor.fetchall() if row[0]]

    def clear_logs(self) -> None:
        """Clear all event logs in the database."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM event_logs")
            conn.commit()


class EventLogViewerWidget(QWidget):
    """PyQt6 widget to display event logs with filters and export capabilities."""

    def __init__(
        self, logger_instance: EventLogger | None = None, parent: QWidget | None = None
    ) -> None:
        """Initialize the viewer widget.

        Args:
            logger_instance: Custom EventLogger instance. Creates default if None.
            parent: Parent QWidget.
        """
        super().__init__(parent)
        self.logger = logger_instance or EventLogger()
        self._init_ui()
        self.apply_filters()

    def _init_ui(self) -> None:
        """Initialize the user interface elements."""
        main_layout = QVBoxLayout(self)

        # Filters Group
        filters_group = QGroupBox("Filters")
        grid_layout = QGridLayout(filters_group)

        # Start Date
        self.start_date_checkbox = QCheckBox("Start Date:")
        self.start_date_checkbox.setChecked(True)
        self.start_date_edit = QDateTimeEdit(QDateTime.currentDateTime().addDays(-7))
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_checkbox.toggled.connect(self.start_date_edit.setEnabled)

        grid_layout.addWidget(self.start_date_checkbox, 0, 0)
        grid_layout.addWidget(self.start_date_edit, 0, 1)

        # End Date
        self.end_date_checkbox = QCheckBox("End Date:")
        self.end_date_checkbox.setChecked(True)
        self.end_date_edit = QDateTimeEdit(QDateTime.currentDateTime().addDays(1))
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_checkbox.toggled.connect(self.end_date_edit.setEnabled)

        grid_layout.addWidget(self.end_date_checkbox, 0, 2)
        grid_layout.addWidget(self.end_date_edit, 0, 3)

        # Severity
        grid_layout.addWidget(QLabel("Severity:"), 1, 0)
        self.severity_combo = QComboBox()
        self.severity_combo.addItems(["All", "INFO", "WARNING", "ERROR", "CRITICAL"])
        grid_layout.addWidget(self.severity_combo, 1, 1)

        # Event Type
        grid_layout.addWidget(QLabel("Event Type:"), 1, 2)
        self.event_type_combo = QComboBox()
        grid_layout.addWidget(self.event_type_combo, 1, 3)

        # Search Query
        grid_layout.addWidget(QLabel("Search Keyword:"), 2, 0)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search descriptions/details...")
        grid_layout.addWidget(self.search_input, 2, 1, 1, 2)

        # Filter Buttons
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Filters")
        self.apply_btn.clicked.connect(self.apply_filters)
        self.reset_btn = QPushButton("Reset Filters")
        self.reset_btn.clicked.connect(self.reset_filters)

        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.reset_btn)
        grid_layout.addLayout(btn_layout, 2, 3)

        main_layout.addWidget(filters_group)

        # Table Widget
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            [
                "ID",
                "Timestamp",
                "Event Type",
                "Severity",
                "Operator",
                "Description",
                "Details",
            ]
        )
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        main_layout.addWidget(self.table)

        # Bottom Bar
        bottom_layout = QHBoxLayout()
        self.status_label = QLabel("Showing 0 logs")
        bottom_layout.addWidget(self.status_label)

        bottom_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.apply_filters)
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self.export_to_csv)

        bottom_layout.addWidget(self.refresh_btn)
        bottom_layout.addWidget(self.export_btn)

        main_layout.addLayout(bottom_layout)

    def update_event_types_combobox(self) -> None:
        """Dynamically populate unique event types from the database."""
        current_selection = self.event_type_combo.currentText()
        self.event_type_combo.clear()
        self.event_type_combo.addItem("All")

        try:
            event_types = self.logger.get_unique_event_types()
            for et in event_types:
                self.event_type_combo.addItem(et)
        except Exception:
            logger.exception("Failed to populate event-type filter from DB")

        idx = self.event_type_combo.findText(current_selection)
        if idx >= 0:
            self.event_type_combo.setCurrentIndex(idx)
        else:
            self.event_type_combo.setCurrentIndex(0)

    def apply_filters(self) -> None:
        """Fetch and display logs based on filter criteria."""
        self.update_event_types_combobox()

        start_date = None
        if self.start_date_checkbox.isChecked():
            start_date = self.start_date_edit.dateTime().toPyDateTime()

        end_date = None
        if self.end_date_checkbox.isChecked():
            end_date = self.end_date_edit.dateTime().toPyDateTime()

        severity = self.severity_combo.currentText()
        event_type = self.event_type_combo.currentText()
        keyword = self.search_input.text().strip() or None

        try:
            logs = self.logger.fetch_logs(
                start_date=start_date,
                end_date=end_date,
                severity=severity,
                event_type=event_type,
                keyword=keyword,
            )
            self._populate_table(logs)
            self.status_label.setText(f"Showing {len(logs)} logs")
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to fetch logs: {e}")

    def reset_filters(self) -> None:
        """Reset filters to default values."""
        self.start_date_checkbox.setChecked(True)
        self.start_date_edit.setEnabled(True)
        self.start_date_edit.setDateTime(QDateTime.currentDateTime().addDays(-7))

        self.end_date_checkbox.setChecked(True)
        self.end_date_edit.setEnabled(True)
        self.end_date_edit.setDateTime(QDateTime.currentDateTime().addDays(1))

        self.severity_combo.setCurrentIndex(0)
        self.event_type_combo.setCurrentIndex(0)
        self.search_input.clear()
        self.apply_filters()

    def _populate_table(
        self, logs: list[tuple[int, str, str, str, str | None, str, str | None]]
    ) -> None:
        """Fill QTableWidget with log entries and color-code by severity."""
        self.table.setRowCount(0)
        self.table.setRowCount(len(logs))

        for row_idx, log in enumerate(logs):
            severity = str(log[3]).upper()

            # Soft background coloring for distinct visual severity
            bg_color = None
            fg_color = None
            if severity == "CRITICAL":
                bg_color = QColor(255, 200, 200)
                fg_color = QColor(150, 0, 0)
            elif severity == "ERROR":
                bg_color = QColor(255, 220, 220)
                fg_color = QColor(180, 0, 0)
            elif severity == "WARNING":
                bg_color = QColor(255, 243, 205)
                fg_color = QColor(133, 100, 4)
            elif severity == "INFO":
                if "login" in str(log[2]).lower():
                    bg_color = QColor(212, 239, 223)
                    fg_color = QColor(21, 101, 41)

            for col_idx, val in enumerate(log):
                item = QTableWidgetItem(str(val) if val is not None else "")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                if bg_color:
                    item.setBackground(bg_color)
                if fg_color:
                    item.setForeground(fg_color)

                self.table.setItem(row_idx, col_idx, item)

    def export_to_csv(self) -> None:
        """Export the currently filtered list of logs to a CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs to CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        start_date = None
        if self.start_date_checkbox.isChecked():
            start_date = self.start_date_edit.dateTime().toPyDateTime()

        end_date = None
        if self.end_date_checkbox.isChecked():
            end_date = self.end_date_edit.dateTime().toPyDateTime()

        severity = self.severity_combo.currentText()
        event_type = self.event_type_combo.currentText()
        keyword = self.search_input.text().strip() or None

        try:
            logs = self.logger.fetch_logs(
                start_date=start_date,
                end_date=end_date,
                severity=severity,
                event_type=event_type,
                keyword=keyword,
            )
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "ID",
                        "Timestamp",
                        "Event Type",
                        "Severity",
                        "Operator",
                        "Description",
                        "Details",
                    ]
                )
                writer.writerows(logs)
            QMessageBox.information(
                self,
                "Export Success",
                f"Successfully exported {len(logs)} log entries to CSV.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export logs: {e}")
