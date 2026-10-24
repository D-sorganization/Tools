# .oss-scanner/Dockerfile — offline security-audit image for D-sorganization/Tools.
#
# Built by the fleet security scanner (anthropics/oss-scanner contract) with this repository as the
# build context. Network is available ONLY while this file builds; the audit that follows runs
# offline, so every dependency an auditor needs to import, run, or test the code is fetched here.
#
# Source of truth for dependencies: pyproject.toml (extras), Dockerfile.dev, Cargo.toml.
# No secrets, tokens, or credentials are used or baked in. Qt runs headless (offscreen).
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    MUJOCO_GL=egl \
    SDL_VIDEODRIVER=dummy \
    SKIP_UI_BUILD=1 \
    CI=1 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# System packages: compilers for native wheels and Rust crates, plus the shared libraries PyQt6,
# OpenCV-headless and MuJoCo need to import under QT_QPA_PLATFORM=offscreen.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config clang \
      libffi-dev libssl-dev libjpeg-dev zlib1g-dev \
      libgl1 libegl1 libglib2.0-0 libdbus-1-3 libfontconfig1 libfreetype6 \
      libxkbcommon0 libxkbcommon-x11-0 libx11-6 libx11-xcb1 libxcb1 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (stable, as pinned by rust-toolchain.toml) for the rust_core/ workspace.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path \
    && chmod -R a+rX "$RUSTUP_HOME" "$CARGO_HOME"

# scanner contract: the checkout lives inside the image, at /src. The scanner runs as uid 65532,
# so ownership is set here and in the layers that create files; a final `chown -R /src` would
# copy the whole tree (and the Rust target dir) into a new overlay layer.
ARG SCAN_UID=65532
COPY --chown=${SCAN_UID}:${SCAN_UID} . /src
WORKDIR /src

# Python: the core package with GUI, test, dev and P1AM backend extras (mirrors Dockerfile.dev's
# `.[all,dev]` plus the nightly full-suite `.[dev,test]` set). A failing install fails the build.
RUN pip install --upgrade pip "setuptools>=61,<85" wheel \
    && pip install -e ".[all,gui,test,dev,p1am]" \
    && find /src -maxdepth 4 -name '*.egg-info' -prune -exec chown -R "${SCAN_UID}:${SCAN_UID}" {} +

# Rust: fetch every crate now (the audit is offline), build the workspace with debug info and run
# its tests. The crates are optional accelerators for the Python package, so failures are
# reported but do not abort the image. Ownership is fixed in the same layer to avoid a copy-up.
ENV CARGO_PROFILE_DEV_DEBUG=true \
    RUSTFLAGS="-C force-frame-pointers=yes"
RUN rustup show active-toolchain \
    && cargo fetch \
    && (cargo fetch --manifest-path src/pendulum_simulator/pendulum-core/Cargo.toml \
        || echo "WARNING: pendulum-core fetch failed") \
    && (cargo build --workspace --all-targets || echo "WARNING: cargo build reported failures") \
    && (timeout 900 cargo test --workspace --no-fail-fast \
        || echo "WARNING: cargo test reported failures") \
    && chown -R "${SCAN_UID}:${SCAN_UID}" /src/target /src/Cargo.lock \
    && chmod -R a+rwX "$CARGO_HOME"
ENV CARGO_NET_OFFLINE=true

# Cheap smoke tests so we know the image works; failures are reported, never fatal. These cover
# the network-facing backends and file importers that threat_model.md calls out.
RUN timeout 1500 python -m pytest -q -p no:cacheprovider -n 4 --timeout 120 \
        tests/calc_backend tests/p1am_control_system tests/data_processor_io \
    || echo "WARNING: pytest smoke subset reported failures"

# The scanner executes as uid 65532 with HOME on tmpfs.
RUN mkdir -p /tmp/home && chown "${SCAN_UID}:${SCAN_UID}" /tmp/home
ENV HOME=/tmp/home

# Phase B runs as uid 65532 (the Runner_Dashboard scan sandbox forces it);
# declare it so the image never defaults to root (Trivy DS-0002).
USER 65532:65532
