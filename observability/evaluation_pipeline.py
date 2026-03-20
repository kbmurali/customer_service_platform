"""
Agent Evaluation Pipeline.

Runs as a scheduled background task (called from metrics_persister.py every
30 seconds) to compute agent quality metrics and write them to:

  1. Prometheus gauges (visible in Grafana immediately)
  2. Redis DB 3 (consumed by the management REST API)

Metrics computed:
  - planning_routing_accuracy   : fraction of steps where routing LLM
                                   confirmed the planner's assignment
                                   (vs VALID_REMOTE_AGENTS fallback fired)
  - positive_feedback_rate      : fraction of 'correct' ratings in last 7 days
  - avg_plan_goals_per_query    : mean goal count per planning trace (drift signal)
  - avg_plan_steps_per_query    : mean step count per planning trace
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (created here, registered once)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Gauge

    planning_routing_accuracy = Gauge(
        "csip_planning_routing_accuracy",
        "Fraction of steps routed to the planner-assigned agent without fallback",
    )
    positive_feedback_rate = Gauge(
        "csip_positive_feedback_rate",
        "Fraction of CSR feedback ratings marked correct (7-day window)",
    )
    avg_plan_goals = Gauge(
        "csip_avg_plan_goals_per_query",
        "Rolling average number of goals per central plan (drift signal)",
    )
    avg_plan_steps = Gauge(
        "csip_avg_plan_steps_per_query",
        "Rolling average number of steps per central plan (drift signal)",
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("EvaluationPipeline: prometheus_client not available, skipping gauges")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation_cycle(redis_client=None) -> None:
    """
    Compute all quality metrics and write them to Prometheus + Redis.

    This function is intentionally defensive — any failure in one metric
    is caught and logged without preventing the others from running.

    Args:
        redis_client: Optional redis.Redis instance (DB 3) for writing
                      metric values for the REST API.  If None, Redis
                      writes are skipped silently.
    """
    _compute_feedback_rate(redis_client)
    _compute_plan_complexity(redis_client)


# ---------------------------------------------------------------------------
# Individual metric computations
# ---------------------------------------------------------------------------

def _compute_feedback_rate(redis_client=None) -> None:
    """Read positive feedback rate from MySQL and publish it."""
    try:
        from databases.feedback_data_access import get_feedback_data_access
        rate = get_feedback_data_access().get_positive_feedback_rate(window_days=7)

        if _PROMETHEUS_AVAILABLE:
            positive_feedback_rate.set(rate)

        if redis_client:
            try:
                redis_client.setex(
                    "metrics:positive_feedback_rate", 86400, str(round(rate, 4))
                )
            except Exception as r_exc:
                logger.debug("EvaluationPipeline: Redis write failed: %s", r_exc)

        logger.info("EvaluationPipeline: positive_feedback_rate=%.3f", rate)

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_feedback_rate failed: %s", exc)


def _compute_plan_complexity(redis_client=None) -> None:
    """
    Derive average goal and step counts from recent CG Plan nodes.

    Queries the last 100 central plans from Neo4j CG.  This gives a
    rolling signal for planning LLM behaviour drift without requiring
    LangFuse API access.
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access

        cg = get_cg_data_access()

        # Cypher: fetch goal and step counts for the 100 most recent central plans.
        # Uses cg.conn.execute_query() — the standard CG query pattern.
        # Note: Plan/Goal/Step use plain labels (no CG: prefix).
        query = """
            MATCH (p:Plan {planType: 'central'})
            OPTIONAL MATCH (p)-[:HAS_GOAL]->(g:Goal)
            OPTIONAL MATCH (g)-[:HAS_STEP]->(s:Step)
            WITH p, count(DISTINCT g) AS goalCount, count(DISTINCT s) AS stepCount
            ORDER BY p.createdAt DESC
            LIMIT 100
            RETURN avg(goalCount) AS avgGoals, avg(stepCount) AS avgSteps
        """
        result = cg.conn.execute_query(query)
        if not result:
            return
        record = result[0]

        avg_g = float(record.get("avgGoals") or 0.0)
        avg_s = float(record.get("avgSteps") or 0.0)

        if _PROMETHEUS_AVAILABLE:
            avg_plan_goals.set(avg_g)
            avg_plan_steps.set(avg_s)

        if redis_client:
            try:
                redis_client.setex("metrics:avg_plan_goals",  86400, str(round(avg_g, 2)))
                redis_client.setex("metrics:avg_plan_steps",  86400, str(round(avg_s, 2)))
            except Exception as r_exc:
                logger.debug("EvaluationPipeline: Redis write failed: %s", r_exc)

        logger.info(
            "EvaluationPipeline: avg_plan_goals=%.2f avg_plan_steps=%.2f",
            avg_g, avg_s,
        )

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_plan_complexity failed: %s", exc)
