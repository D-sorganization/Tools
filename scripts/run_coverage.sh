#!/bin/bash
# Run comprehensive coverage measurement with all report formats
#
# Usage:
#   ./scripts/run_coverage.sh [MIN_COVERAGE] [--baseline-comparison]
#   MIN_COVERAGE defaults to pyproject [tool.coverage.report] fail_under (Tools #4913).
#
# Output:
#   - htmlcov/index.html — interactive HTML coverage report
#   - coverage.xml — machine-readable coverage (for CI)
#   - coverage.json — JSON coverage metrics
#   - coverage_reports/coverage_report.json — policy comparison
#
# Prerequisites:
#   - pytest and pytest-cov installed
#   - core test suite passing

set -e

MIN_COVERAGE="${1:-$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['tool']['coverage']['report']['fail_under'])")}"
BASELINE_COMPARISON="${2:-}"

echo "================================"
echo "Coverage Measurement Pipeline"
echo "================================"
echo "Minimum coverage threshold: ${MIN_COVERAGE}%"

# Run pytest with coverage
echo ""
echo "Running tests with coverage measurement..."
python3 -m pytest \
  tests/ \
  -m "not live_simulation" \
  --import-mode=importlib \
  --cov=src \
  --cov-report=html \
  --cov-report=xml:coverage.xml \
  --cov-report=json:coverage.json \
  --cov-report=term-missing \
  --cov-fail-under="${MIN_COVERAGE}" \
  -v \
  --tb=short \
  -n auto \
  2>&1 | tee coverage.log

COVERAGE_EXIT=$?

if [ $COVERAGE_EXIT -eq 0 ]; then
  echo ""
  echo "✓ Coverage requirement met (>=${MIN_COVERAGE}%)"
else
  echo ""
  echo "✗ Coverage below threshold"
fi

# Generate comparison report if baseline comparison requested
if [ -n "$BASELINE_COMPARISON" ] && [ "$BASELINE_COMPARISON" = "--baseline" ]; then
  echo ""
  echo "Comparing against baseline..."
  python3 scripts/measure_coverage.py \
    --coverage-file coverage.xml \
    --baseline-file config/coverage_baseline.json \
    --policy-file config/coverage_policy.json \
    --output-dir coverage_reports
fi

echo ""
echo "Coverage Reports:"
echo "  - Interactive HTML: htmlcov/index.html"
echo "  - XML (CI):        coverage.xml"
echo "  - JSON:            coverage.json"

exit $COVERAGE_EXIT
