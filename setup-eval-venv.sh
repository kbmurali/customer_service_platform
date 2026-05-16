#!/bin/bash
# Setup script for CSIP AI Evaluation venv
# Usage: bash setup-eval-venv.sh

set -e

echo "=== Installing deepeval ==="
pip install "deepeval>=3.5.0" "pytest-timeout>=2.2.0"

echo "=== Pinning posthog for Chroma compatibility ==="
# DeepEval uses posthog only for telemetry — metrics work with any version.
# Chroma requires posthog<6. Force the downgrade after deepeval installs.
pip install "posthog==5.1.0" --force-reinstall --no-deps

echo "=== Verifying ==="
python -c "from deepeval.metrics import FaithfulnessMetric; print('DeepEval OK')"
python -c "from databases.chroma_experience_store import ChromaExperienceStore; print('Chroma OK')"
python -c "from databases.connections import get_mysql; print('MySQL OK')"

echo ""
echo "=== Eval venv ready ==="
echo "Run evals with: pytest tests/evals/ -s -v --timeout=300"
