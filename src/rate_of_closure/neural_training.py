"""Private campaign neural-training request and configuration contracts."""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingOptions:
    """User-selectable neural training hyperparameters."""

    vendor: str
    features: tuple[str, ...]
    targets: tuple[str, ...]
    hidden_layers: tuple[int, ...]
    activation: str
    alpha: float
    seed: int
    epochs: int
    holdout: float
    split_group: str = "shot_id"


@dataclass(frozen=True)
class TrainingRequest:
    """Auditable external command and environment for one training run."""

    program: str
    arguments: tuple[str, ...]
    working_directory: Path
    python_path: str
    config_path: Path


NEURAL_CAMPAIGN_ENVIRONMENT_VARIABLE = "LAUNCH_MONITOR_NEURAL_REPO"


def discover_training_repository(data_root: Path | None) -> Path | None:
    """Locate a private campaign checkout that exposes the neural CLI."""

    configured = os.environ.get(NEURAL_CAMPAIGN_ENVIRONMENT_VARIABLE)
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.home()
        / "Repositories"
        / "Launch-Monitor-Flight-Model-Campaign-worktrees"
        / "neural-surrogate",
        data_root,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        neural_module = resolved / "src" / "lm_flight_campaign" / "neural_surrogate.py"
        if (resolved / "campaign.toml").is_file() and neural_module.is_file():
            return resolved
    return None


def parse_hidden_layers(text: str) -> tuple[int, ...]:
    """Parse a comma-separated architecture with bounded positive widths."""

    try:
        widths = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("hidden layers must be comma-separated integers") from exc
    if (
        not widths
        or len(widths) > 8
        or any(width < 1 or width > 4096 for width in widths)
    ):
        raise ValueError("use 1-8 hidden layers with widths from 1 to 4096")
    return widths


def _quoted(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _names(values: tuple[str, ...], label: str) -> str:
    if not values or any(
        not value.strip() or not value.isprintable() for value in values
    ):
        raise ValueError(f"{label} must contain printable, named dataset columns")
    return "[" + ", ".join(_quoted(value) for value in values) + "]"


def _validate_options(options: TrainingOptions) -> None:
    if options.activation not in {"relu", "tanh"}:
        raise ValueError("unsupported training activation")
    if not 0.05 <= options.holdout <= 0.5 or options.epochs < 10:
        raise ValueError("holdout or epoch count is outside supported bounds")


def write_training_config(
    path: Path,
    *,
    dataset_path: Path,
    output_path: Path,
    options: TrainingOptions,
) -> None:
    """Write inspectable TOML consumed by the private training CLI."""

    _validate_options(options)
    vendor = options.vendor.lower().replace("-comparable", "")
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    validation_fraction = options.holdout / 2.0
    train_fraction = 1.0 - options.holdout
    output_root = output_path.resolve().parent
    stem = output_path.stem.removesuffix(".nn")
    content = "\n".join(
        (
            "[surrogate]",
            f"vendor = {_quoted(vendor)}",
            f"dataset = {_quoted(str(dataset_path.resolve()))}",
            f"expected_dataset_sha256 = {_quoted(dataset_hash)}",
            f"random_seed = {options.seed}",
            f"split_group = {_quoted(options.split_group)}",
            f"train_fraction = {train_fraction:.12g}",
            f"validation_fraction = {validation_fraction:.12g}",
            f"max_iter = {options.epochs}",
            "architectures = [[" + ", ".join(map(str, options.hidden_layers)) + "]]",
            "learning_curve_fractions = [0.25, 0.5, 1.0]",
            f"activation = {_quoted(options.activation)}",
            f"alpha = {options.alpha:.12g}",
            "",
            "[columns]",
            f"features = {_names(options.features, 'features')}",
            f"targets = {_names(options.targets, 'targets')}",
            "",
            "[outputs]",
            f"bundle = {_quoted(str(output_path.resolve()))}",
            f"index = {_quoted(str(output_root / (stem + '_index.json')))}",
            f"metrics = {_quoted(str(output_root / (stem + '_metrics.csv')))}",
            f"predictions = {_quoted(str(output_root / (stem + '_predictions.csv')))}",
            "learning_curve = "
            + _quoted(str(output_root / (stem + "_learning_curve.csv"))),
            f"manifest = {_quoted(str(output_root / (stem + '_manifest.json')))}",
            f"report = {_quoted(str(output_root / (stem + '_report.md')))}",
            "",
        )
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def build_training_request(campaign_root: Path, config_path: Path) -> TrainingRequest:
    """Build the private CLI request without starting a process."""

    root = campaign_root.resolve()
    source_root = root / "src"
    if not (root / "campaign.toml").is_file():
        raise ValueError("private campaign root must contain campaign.toml")
    return TrainingRequest(
        program=sys.executable,
        arguments=(
            "-m",
            "lm_flight_campaign.cli",
            "--config",
            str(root / "campaign.toml"),
            "neural-train",
            "--training-config",
            str(config_path.resolve()),
        ),
        working_directory=root,
        python_path=str(source_root),
        config_path=config_path.resolve(),
    )


__all__ = [
    "TrainingOptions",
    "TrainingRequest",
    "build_training_request",
    "discover_training_repository",
    "parse_hidden_layers",
    "write_training_config",
]
