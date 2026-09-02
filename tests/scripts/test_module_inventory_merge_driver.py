"""Tests for the module-inventory git merge driver (#4818).

``manuals/tools/manifests/module-inventory.json`` and its shards are a pure
function of the tracked working tree (see
``scripts/build_tools_module_inventory.py``), so any textual conflict on
those generated paths should be resolved by regenerating rather than by
inspecting the conflicting versions' content. These tests exercise
``scripts/git/module_inventory_merge_driver.py`` and
``scripts/git/install_merge_drivers.py`` directly, without invoking a real
``git merge`` (an end-to-end reproduction lives outside the test suite --
see the PR description for #4818).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.git import install_merge_drivers
from scripts.git import module_inventory_merge_driver as driver
from scripts.git import regenerate_module_inventory_during_merge as merge_fixup


class TestResolve:
    """``resolve()`` must ignore %O/%A/%B content and regenerate instead."""

    def test_overwrites_current_with_regenerated_content(self, tmp_path):
        root = tmp_path
        resolved_relative = "manuals/tools/manifests/module-inventory.json"
        regenerated_file = root / resolved_relative
        regenerated_file.parent.mkdir(parents=True)
        regenerated_file.write_text('{"regenerated": true}\n', encoding="utf-8")

        current_path = tmp_path / "current-A-side.json"
        current_path.write_text("<<<<<<< conflict garbage\n", encoding="utf-8")

        with patch.object(
            driver.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            ),
        ) as mock_run:
            exit_code = driver.resolve(current_path, resolved_relative, root)

        assert exit_code == 0
        assert current_path.read_text(encoding="utf-8") == '{"regenerated": true}\n'
        # The conflicting O/A/B content was never consulted for the result.
        assert "conflict garbage" not in current_path.read_text(encoding="utf-8")
        mock_run.assert_called_once()
        called_cmd = mock_run.call_args.args[0]
        assert called_cmd[1:3] == ["-m", driver._REGEN_MODULE]
        assert mock_run.call_args.kwargs["cwd"] == root

    def test_regeneration_failure_falls_back_to_conflicted_merge(self, tmp_path):
        current_path = tmp_path / "current-A-side.json"
        current_path.write_text("original\n", encoding="utf-8")

        with patch.object(
            driver.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="boom"
            ),
        ):
            exit_code = driver.resolve(
                current_path,
                "manuals/tools/manifests/module-inventory.json",
                tmp_path,
            )

        assert exit_code == 1
        # %A was left untouched so git's normal conflict handling applies.
        assert current_path.read_text(encoding="utf-8") == "original\n"

    def test_missing_shard_after_regeneration_falls_back_instead_of_guessing(
        self, tmp_path
    ):
        """A shard the merged tree no longer needs can't be silently deleted."""
        current_path = tmp_path / "current-A-side.json"
        current_path.write_text("original\n", encoding="utf-8")

        with patch.object(
            driver.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            ),
        ):
            exit_code = driver.resolve(
                current_path,
                "manuals/tools/manifests/module-inventory/entries-099.json",
                tmp_path,
            )

        assert exit_code == 1
        assert current_path.read_text(encoding="utf-8") == "original\n"


class TestMain:
    """``main()`` must match git's ``driver %O %A %B %L %P`` contract."""

    def test_too_few_arguments_is_a_usage_error(self):
        assert driver.main(["only", "two"]) == 2

    def test_dispatches_to_resolve_with_ancestor_and_other_ignored(self, tmp_path):
        with (
            patch.object(driver, "_repo_root", return_value=tmp_path),
            patch.object(driver, "resolve", return_value=0) as mock_resolve,
        ):
            exit_code = driver.main(
                [
                    "/tmp/ancestor",
                    "/tmp/current",
                    "/tmp/other",
                    "7",
                    "manuals/tools/manifests/module-inventory.json",
                ]
            )

        assert exit_code == 0
        mock_resolve.assert_called_once_with(
            Path("/tmp/current"),
            "manuals/tools/manifests/module-inventory.json",
            tmp_path,
        )


class TestInstallMergeDrivers:
    """The installer must register local git config, not repo-tracked state."""

    def test_registers_driver_and_name_in_local_git_config(self, tmp_path):
        calls: list[list[str]] = []

        def fake_run(cmd, check):  # noqa: ARG001 - test double signature
            calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        with patch.object(
            install_merge_drivers.subprocess, "run", side_effect=fake_run
        ):
            driver_command = install_merge_drivers.install(
                python_executable="/usr/bin/python3"
            )

        assert len(calls) == 2
        assert calls[0][:3] == ["git", "config", "merge.module-inventory-regen.driver"]
        assert calls[0][3] == driver_command
        assert calls[1][:3] == ["git", "config", "merge.module-inventory-regen.name"]
        # The registered command re-invokes the same interpreter that ran
        # the installer, quoted, and passes through git's placeholders
        # unmodified for git to substitute.
        assert '"/usr/bin/python3"' in driver_command
        assert "module_inventory_merge_driver.py %O %A %B %L %P" in driver_command


class TestMergeFixup:
    """The pre-merge-commit-stage fixup is the authoritative correctness layer.

    It is only ever invoked by git's ``pre-merge-commit`` hook (see
    ``.pre-commit-config.yaml``'s ``stages: [pre-merge-commit]``), which by
    definition only fires while a merge commit is being created -- so
    ``fixup()`` has no "is this actually a merge" branch to test. An
    earlier version of this script tried to gate on ``MERGE_HEAD``, but
    that file was confirmed (via a raw, framework-free hook script) to not
    exist yet at the point ``pre-merge-commit`` fires; that guard always
    read false and silently no-op'd the entire hook.
    """

    def test_noop_when_already_fresh(self, tmp_path):
        fresh_check = subprocess.CompletedProcess(args=[], returncode=0)
        with patch.object(
            merge_fixup.subprocess, "run", return_value=fresh_check
        ) as mock_run:
            exit_code = merge_fixup.fixup(tmp_path)

        assert exit_code == 0
        # Only the --check call, no write-mode regeneration and no `git add`.
        mock_run.assert_called_once()
        assert "--check" in mock_run.call_args.args[0]

    def test_regenerates_and_stages_when_stale(self, tmp_path):
        stale_check = subprocess.CompletedProcess(args=[], returncode=1)
        successful_regen = subprocess.CompletedProcess(args=[], returncode=0)
        successful_add = subprocess.CompletedProcess(args=[], returncode=0)

        with patch.object(
            merge_fixup.subprocess,
            "run",
            side_effect=[stale_check, successful_regen, successful_add],
        ) as mock_run:
            exit_code = merge_fixup.fixup(tmp_path)

        assert exit_code == 0
        assert mock_run.call_count == 3
        check_cmd, regen_cmd, add_cmd = (c.args[0] for c in mock_run.call_args_list)
        assert "--check" in check_cmd
        assert "--check" not in regen_cmd
        assert add_cmd[:2] == ["git", "add"]

    def test_regeneration_failure_blocks_the_commit(self, tmp_path):
        stale_check = subprocess.CompletedProcess(args=[], returncode=1, stdout="")
        failed_regen = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="boom"
        )

        with patch.object(
            merge_fixup.subprocess,
            "run",
            side_effect=[stale_check, failed_regen],
        ):
            exit_code = merge_fixup.fixup(tmp_path)

        assert exit_code == 1
