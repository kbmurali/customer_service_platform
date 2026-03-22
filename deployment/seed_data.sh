#!/bin/bash
# ==========================================================================
# seed_data.sh — Seeds LangFuse prompt templates and Chroma experience
#                collections after the CSIP Docker stack is healthy.
#
# Prerequisites:
#   1. The Docker stack (health_insurance) is deployed and all services
#      show 1/1 replicas in 'docker stack services health_insurance'.
#   2. LangFuse is accessible at http://localhost:3001.
#   3. A LangFuse project has been created and API keys (Public + Secret)
#      have been registered as Docker Swarm secrets:
#        docker secret create LANGFUSE_PUBLIC_KEY -
#        docker secret create LANGFUSE_SECRET_KEY -
#   4. The agentic-access-api service is running (it contains all
#      Python dependencies and network access needed by both scripts).
#
# Usage:
#   cd deployment/
#   chmod +x seed_data.sh
#   ./seed_data.sh
#
# Both scripts are idempotent — safe to run multiple times.
# ==========================================================================

set -e

STACK_NAME="health_insurance"
API_SERVICE="${STACK_NAME}_agentic-access-api"

echo "=============================================="
echo "  CSIP Data Seeding"
echo "=============================================="
echo ""

# ── Step 1: Locate the agentic-access-api container ──────────────────────

echo "[1/4] Locating agentic-access-api container..."
CONTAINER_ID=$(docker ps -q -f name="${API_SERVICE}" | head -n 1)

if [ -z "$CONTAINER_ID" ]; then
    echo "ERROR: Cannot find a running container for ${API_SERVICE}."
    echo "       Ensure the stack is deployed and the API service is healthy:"
    echo "       docker stack services ${STACK_NAME}"
    exit 1
fi

echo "       Found container: ${CONTAINER_ID}"
echo ""

# ── Step 2: Verify Chroma is reachable from inside the container ─────────

echo "[2/4] Verifying Chroma connectivity..."
if docker exec "$CONTAINER_ID" python3 -c "
from databases.connections import get_chroma
c = get_chroma().connect()
c.heartbeat()
print('Chroma heartbeat OK')
" 2>/dev/null; then
    echo "       Chroma is healthy."
else
    echo "ERROR: Chroma is not reachable from the API container."
    echo "       Wait for the chroma service to become healthy and retry."
    exit 1
fi
echo ""

# ── Step 3: Seed LangFuse prompt templates ───────────────────────────────

echo "[3/4] Seeding LangFuse prompt templates..."
if docker exec "$CONTAINER_ID" bash -c '
    export OPENAI_API_KEY="$(cat /run/secrets/OPENAI_API_KEY 2>/dev/null || echo "")"
    export ANTHROPIC_API_KEY="$(cat /run/secrets/ANTHROPIC_API_KEY 2>/dev/null || echo "")"
    export LANGFUSE_SECRET_KEY="$(cat /run/secrets/LANGFUSE_SECRET_KEY 2>/dev/null || echo "")"
    export LANGFUSE_PUBLIC_KEY="$(cat /run/secrets/LANGFUSE_PUBLIC_KEY 2>/dev/null || echo "")"
    python3 data/seed_langfuse_prompts.py
' 2>&1; then
    echo "       LangFuse prompts seeded successfully."
else
    echo "WARNING: LangFuse prompt seeding failed."
    echo "         This is non-fatal if LangFuse is not yet configured."
    echo "         You can re-run this script after setting up LangFuse."
fi
echo ""

# ── Step 4: Seed Chroma experience collections ──────────────────────────

echo "[4/4] Seeding Chroma experience collections..."
if docker exec "$CONTAINER_ID" bash -c '
    export OPENAI_API_KEY="$(cat /run/secrets/OPENAI_API_KEY 2>/dev/null || echo "")"
    python3 data/seed_experience_collections.py
' 2>&1; then
    echo "       Experience collections seeded successfully."
else
    echo "WARNING: Experience collection seeding failed."
    echo "         Verify Chroma is healthy and retry."
fi
echo ""

echo "=============================================="
echo "  Seeding complete."
echo ""
echo "  Next steps:"
echo "    - Deploy per-service compose files (team MCP + A2A servers)"
echo "    - Deploy the Agentic Access API compose file last"
echo "    - Run: docker stack services ${STACK_NAME}"
echo "=============================================="
