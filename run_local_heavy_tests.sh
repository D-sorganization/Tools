#!/bin/bash
set -e

echo "=========================================================="
echo " Starting Tools Virtual Heavy Physics Testing Run "
echo "=========================================================="

echo "[1/2] Building local testing image (upstream-heavy)..."
echo "      (This includes MuJoCo, Drake, Pinocchio, OpenSim, etc.)"
# Use --network=host to bypass common WSL2 internal DNS routing issues with pure docker setups.
docker build --network=host -t upstream-heavy -f Dockerfile.heavy_test .

echo ""
echo "[2/2] Running rigorous integration test suite internally..."
# Use Xvfb to emulate a display since we are opening GUI tools / Physics Viewers
docker run --rm --network=host -v "$(pwd):/app" upstream-heavy xvfb-run -a pytest -v -m "live_simulation" tests/test_heavy_physics_suite.py

echo ""
echo "=========================================================="
echo " Process Finished Successfully! "
echo "=========================================================="
