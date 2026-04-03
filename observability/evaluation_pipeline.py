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
    from prometheus_client import Gauge, Counter

    # --- Existing Tier 2 gauges ---
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

    # --- Tier 1: LLM & Cost ---
    estimated_tokens_per_query = Gauge(
        "csip_estimated_tokens_per_query",
        "Estimated average token consumption per user query (input+output chars/4)",
    )
    llm_calls_per_query = Gauge(
        "csip_llm_calls_per_query",
        "Average number of LLM agent executions per user session",
    )

    # --- Tier 1: Tool & Output Quality ---
    tool_success_rate = Gauge(
        "csip_tool_success_rate",
        "Fraction of tool executions that completed successfully (vs error/pending)",
    )

    # --- Tier 2: Agent Health ---
    agent_error_rate = Gauge(
        "csip_agent_error_rate",
        "Fraction of agent executions that encountered errors",
    )
    avg_agent_latency = Gauge(
        "csip_avg_agent_latency_seconds",
        "Average agent execution duration in seconds (team-level supervisors)",
    )
    avg_e2e_latency = Gauge(
        "csip_avg_e2e_latency_seconds",
        "Average end-to-end query latency from Session start to plan completion",
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
    """
    # Tier 2 — existing
    _compute_feedback_rate(redis_client)
    _compute_plan_complexity(redis_client)
    _compute_routing_accuracy(redis_client)

    # Tier 1 — LLM cost & quality
    _compute_token_estimates(redis_client)
    _compute_llm_calls_per_query(redis_client)
    _compute_tool_success_rate(redis_client)

    # Tier 2 — agent health
    _compute_agent_error_rate(redis_client)
    _compute_agent_latency(redis_client)
    _compute_e2e_latency(redis_client)

    # Experience extraction (may not be available in A2A containers)
    _run_experience_extraction(redis_client)

    # Feedback pattern analysis (daily, throttled via Redis TTL)
    _run_pattern_analysis(redis_client)

    # Persist aggregated metrics to MySQL (hourly, throttled via Redis TTL)
    _persist_metrics_to_mysql(redis_client)


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


def _compute_routing_accuracy(redis_client=None) -> None:
    """
    Compute planning/routing accuracy from CG Step nodes.

    Accuracy = completed_steps / (completed_steps + skipped_steps)

    A Step with status 'completed' means the planner's routing was correct —
    the assigned worker executed the tool successfully. A Step with status
    'skipped' means the worker couldn't handle the request (e.g. missing
    required data, wrong tool assignment).

    Queries the last 200 team-level steps (planType='team') so the metric
    reflects recent routing behaviour rather than all-time.
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access

        cg = get_cg_data_access()
        query = """
            MATCH (p:Plan {planType: 'team'})-[:HAS_GOAL]->(:Goal)-[:HAS_STEP]->(s:Step)
            WHERE s.status IN ['completed', 'skipped']
            WITH s ORDER BY s.createdAt DESC LIMIT 200
            RETURN
                count(CASE WHEN s.status = 'completed' THEN 1 END) AS completed,
                count(CASE WHEN s.status = 'skipped'   THEN 1 END) AS skipped
        """
        result = cg.conn.execute_query(query)
        if not result:
            return
        record = result[0]
        completed = int(record.get("completed") or 0)
        skipped = int(record.get("skipped") or 0)
        total = completed + skipped

        accuracy = completed / total if total > 0 else 0.0

        if _PROMETHEUS_AVAILABLE:
            planning_routing_accuracy.set(accuracy)

        if redis_client:
            try:
                redis_client.setex("metrics:planning_routing_accuracy", 86400, str(round(accuracy, 4)))
            except Exception as r_exc:
                logger.debug("EvaluationPipeline: Redis write failed: %s", r_exc)

        logger.info(
            "EvaluationPipeline: planning_routing_accuracy=%.3f (%d/%d)",
            accuracy, completed, total,
        )

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_routing_accuracy failed: %s", exc)


# ---------------------------------------------------------------------------
# Tier 1: Token estimation (from CG Session conversation lengths)
# ---------------------------------------------------------------------------

def _compute_token_estimates(redis_client=None) -> None:
    """
    Estimate average token consumption per query from CG Session nodes.

    Uses character count / 4 as a rough token estimate across the
    conversation messages (input + output). Queries the 100 most recent
    sessions.  This is an estimate — for precise tracking, LLM providers
    would need to return usage_metadata on each call.
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()
        result = cg.conn.execute_query(
            """
            MATCH (s:Session)
            WHERE s.conversationMessages IS NOT NULL
            WITH s ORDER BY s.startTime DESC LIMIT 100
            RETURN avg(size(s.conversationMessages)) AS avgChars
            """
        )
        if not result:
            return
        avg_chars = float(result[0].get("avgChars") or 0)
        avg_tokens = avg_chars / 4.0  # rough char→token ratio

        if _PROMETHEUS_AVAILABLE:
            estimated_tokens_per_query.set(round(avg_tokens, 1))

        if redis_client:
            try:
                redis_client.setex("metrics:estimated_tokens_per_query", 86400, str(round(avg_tokens, 1)))
            except Exception:
                pass

        logger.info("EvaluationPipeline: estimated_tokens_per_query=%.1f", avg_tokens)

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_token_estimates failed: %s", exc)


# ---------------------------------------------------------------------------
# Tier 1: LLM calls per query (AgentExecution count per Session)
# ---------------------------------------------------------------------------

def _compute_llm_calls_per_query(redis_client=None) -> None:
    """
    Count average number of AgentExecution nodes per Session.

    Each AgentExecution represents an LLM invocation — planner, router,
    team supervisor, worker, consolidation.  A high value signals plan
    complexity or excessive retries.
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()
        result = cg.conn.execute_query(
            """
            MATCH (s:Session)-[:HAS_EXECUTION]->(top:AgentExecution)
            WHERE s.startTime IS NOT NULL
            WITH s ORDER BY s.startTime DESC LIMIT 100
            OPTIONAL MATCH (s)-[:HAS_EXECUTION]->(:AgentExecution)-[:HAS_PLAN]->(:Plan)
                           -[:HAS_GOAL]->(:Goal)-[:HAS_STEP]->(:Step)
                           -[:EXECUTED_BY]->(ae:AgentExecution)
            WITH s, count(DISTINCT ae) + 1 AS execCount
            RETURN avg(execCount) AS avgCalls
            """
        )
        if not result:
            return
        avg_calls = float(result[0].get("avgCalls") or 0)

        if _PROMETHEUS_AVAILABLE:
            llm_calls_per_query.set(round(avg_calls, 2))

        if redis_client:
            try:
                redis_client.setex("metrics:llm_calls_per_query", 86400, str(round(avg_calls, 2)))
            except Exception:
                pass

        logger.info("EvaluationPipeline: llm_calls_per_query=%.2f", avg_calls)

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_llm_calls_per_query failed: %s", exc)


# ---------------------------------------------------------------------------
# Tier 1: Tool execution success rate
# ---------------------------------------------------------------------------

def _compute_tool_success_rate(redis_client=None) -> None:
    """
    Compute fraction of ToolExecution nodes with status 'success'
    vs total (success + error + pending_approval).
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()
        result = cg.conn.execute_query(
            """
            MATCH (te:ToolExecution)
            WITH te ORDER BY te.timestamp DESC LIMIT 200
            RETURN
                count(CASE WHEN te.status = 'success' THEN 1 END) AS successes,
                count(te) AS total
            """
        )
        if not result:
            return
        successes = int(result[0].get("successes") or 0)
        total = int(result[0].get("total") or 0)
        rate = successes / total if total > 0 else 0.0

        if _PROMETHEUS_AVAILABLE:
            tool_success_rate.set(round(rate, 4))

        if redis_client:
            try:
                redis_client.setex("metrics:tool_success_rate", 86400, str(round(rate, 4)))
            except Exception:
                pass

        logger.info("EvaluationPipeline: tool_success_rate=%.3f (%d/%d)", rate, successes, total)

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_tool_success_rate failed: %s", exc)


# ---------------------------------------------------------------------------
# Tier 2: Agent error rate
# ---------------------------------------------------------------------------

def _compute_agent_error_rate(redis_client=None) -> None:
    """
    Compute fraction of AgentExecution nodes that have a HAD_ERROR
    relationship or status containing 'error'/'skipped'/'failed'.
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()
        result = cg.conn.execute_query(
            """
            MATCH (ae:AgentExecution)
            WHERE ae.startTime IS NOT NULL
            WITH ae ORDER BY ae.startTime DESC LIMIT 200
            OPTIONAL MATCH (ae)-[:CALLED_TOOL]->(te:ToolExecution)-[:HAD_ERROR]->(err)
            WITH ae, count(err) AS errorCount
            RETURN
                count(CASE WHEN ae.status IN ['failed', 'error', 'skipped'] OR errorCount > 0 THEN 1 END) AS errors,
                count(ae) AS total
            """
        )
        if not result:
            return
        errors = int(result[0].get("errors") or 0)
        total = int(result[0].get("total") or 0)
        rate = errors / total if total > 0 else 0.0

        if _PROMETHEUS_AVAILABLE:
            agent_error_rate.set(round(rate, 4))

        if redis_client:
            try:
                redis_client.setex("metrics:agent_error_rate", 86400, str(round(rate, 4)))
            except Exception:
                pass

        logger.info("EvaluationPipeline: agent_error_rate=%.3f (%d/%d)", rate, errors, total)

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_agent_error_rate failed: %s", exc)


# ---------------------------------------------------------------------------
# Tier 2: Agent latency (team-level supervisor duration)
# ---------------------------------------------------------------------------

def _compute_agent_latency(redis_client=None) -> None:
    """
    Compute average agent execution duration from CG AgentExecution nodes
    that have the 'duration' property (set by team supervisors in ms).
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()
        result = cg.conn.execute_query(
            """
            MATCH (ae:AgentExecution)
            WHERE ae.duration IS NOT NULL AND ae.agentType = 'supervisor'
            WITH ae ORDER BY ae.startTime DESC LIMIT 200
            RETURN avg(ae.duration) AS avgDurationMs
            """
        )
        if not result:
            return
        avg_ms = float(result[0].get("avgDurationMs") or 0)
        avg_sec = avg_ms / 1000.0

        if _PROMETHEUS_AVAILABLE:
            avg_agent_latency.set(round(avg_sec, 3))

        if redis_client:
            try:
                redis_client.setex("metrics:avg_agent_latency_seconds", 86400, str(round(avg_sec, 3)))
            except Exception:
                pass

        logger.info("EvaluationPipeline: avg_agent_latency=%.3fs (%.1fms)", avg_sec, avg_ms)

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_agent_latency failed: %s", exc)


# ---------------------------------------------------------------------------
# Tier 2: End-to-end latency (Session start to Plan completion)
# ---------------------------------------------------------------------------

def _compute_e2e_latency(redis_client=None) -> None:
    """
    Compute average end-to-end latency from Session startTime to
    the central Plan completedAt for the most recent 100 sessions.
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()
        result = cg.conn.execute_query(
            """
            MATCH (s:Session)-[:HAS_EXECUTION]->(:AgentExecution)-[:HAS_PLAN]->(p:Plan {planType:'central'})
            WHERE s.startTime IS NOT NULL AND p.completedAt IS NOT NULL
            WITH s, p ORDER BY s.startTime DESC LIMIT 100
            RETURN avg(duration.between(s.startTime, p.completedAt).seconds) AS avgSeconds
            """
        )
        if not result:
            return
        avg_sec = float(result[0].get("avgSeconds") or 0)

        if _PROMETHEUS_AVAILABLE:
            avg_e2e_latency.set(round(avg_sec, 2))

        if redis_client:
            try:
                redis_client.setex("metrics:avg_e2e_latency_seconds", 86400, str(round(avg_sec, 2)))
            except Exception:
                pass

        logger.info("EvaluationPipeline: avg_e2e_latency=%.2fs", avg_sec)

    except Exception as exc:
        logger.error("EvaluationPipeline._compute_e2e_latency failed: %s", exc)


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------

# Prometheus gauges for experience store health
try:
    _experience_store_size = Gauge(
        "csip_experience_store_size",
        "Number of documents in the successful_experiences Chroma collection",
    )
    _experience_hit_rate = Gauge(
        "csip_experience_hit_rate",
        "Fraction of planning calls where at least one experience was retrieved",
    )
except Exception:
    _experience_store_size = None
    _experience_hit_rate = None


def _run_experience_extraction(redis_client=None) -> None:
    """
    Extract experiences from correct-rated sessions into the Chroma store.

    Non-fatal — any failure is caught and logged without affecting
    the other metrics in the evaluation cycle.

    In A2A server containers, the experience_extraction_pipeline module
    is not present (intentionally — extraction runs only in the main API).
    The ImportError is caught at debug level to avoid log spam.
    """
    try:
        from observability.experience_extraction_pipeline import run_extraction
    except ImportError:
        # Expected in A2A server containers — module not deployed there
        logger.debug("EvaluationPipeline: experience_extraction_pipeline not available (expected in A2A containers)")
        return

    try:
        extracted = run_extraction(redis_client)

        if redis_client:
            try:
                redis_client.setex(
                    "metrics:experiences_extracted_last_cycle", 86400, str(extracted)
                )
            except Exception as r_exc:
                logger.debug("EvaluationPipeline: Redis write failed: %s", r_exc)

        logger.info("EvaluationPipeline: experience extraction completed, extracted=%d", extracted)

    except Exception as exc:
        logger.error("EvaluationPipeline._run_experience_extraction failed: %s", exc)


def _run_pattern_analysis(redis_client=None) -> None:
    """
    Run feedback pattern analysis once per day.

    Uses a Redis key with 24h TTL as a throttle — if the key exists,
    the analysis was already run today and this call is a no-op.

    Non-fatal — any failure is caught and logged without affecting
    the other metrics in the evaluation cycle.

    In A2A server containers, the feedback_pattern_analyzer module
    is not present (intentionally — analysis runs only in the main API).
    The ImportError is caught at debug level to avoid log spam.
    """
    try:
        from observability.feedback_pattern_analyzer import run_analysis
    except ImportError:
        logger.debug("EvaluationPipeline: feedback_pattern_analyzer not available (expected in A2A containers)")
        return

    # Daily throttle — skip if already run within 24h
    _THROTTLE_KEY = "eval:pattern_analysis:last_run"
    try:
        if redis_client and redis_client.exists(_THROTTLE_KEY):
            return
    except Exception:
        pass  # Redis unavailable — run anyway

    try:
        reports = run_analysis(window_days=30, min_cluster_size=3)

        if redis_client:
            try:
                redis_client.setex(_THROTTLE_KEY, 86400, "1")  # 24h TTL
            except Exception as r_exc:
                logger.debug("EvaluationPipeline: Redis throttle write failed: %s", r_exc)

        logger.info("EvaluationPipeline: pattern analysis produced %d reports", len(reports))

    except Exception as exc:
        logger.error("EvaluationPipeline._run_pattern_analysis failed: %s", exc)


def _persist_metrics_to_mysql(redis_client=None) -> None:
    """
    Persist aggregated session, agent, and tool metrics to MySQL tables.

    Runs once per hour (throttled via Redis key with 1h TTL).  Queries
    the Context Graph for execution data from the current hour and upserts
    into ``session_metrics``, ``agent_metrics``, and ``tool_metrics``.

    These MySQL tables provide a permanent historical record that survives
    Prometheus TSDB expiry (default 15 days) and container restarts.  The
    CG remains the source of truth; MySQL is the pre-aggregated summary
    layer for fast trend queries.

    Non-fatal — any failure is caught and logged without affecting
    the other steps in the evaluation cycle.
    """
    _THROTTLE_KEY = "eval:mysql_metrics_persist:last_run"
    _THROTTLE_TTL = 3600  # 1 hour

    try:
        if redis_client and redis_client.exists(_THROTTLE_KEY):
            return
    except Exception:
        pass  # Redis unavailable — run anyway

    try:
        from databases.connections import get_mysql
        from databases.context_graph_data_access import get_cg_data_access
        import hashlib
        from datetime import datetime, timezone

        mysql = get_mysql()
        cg = get_cg_data_access()
        now = datetime.now(timezone.utc)
        current_date = now.strftime("%Y-%m-%d")
        current_hour = now.hour

        # ── Session metrics ──────────────────────────────────────────
        try:
            session_result = cg.conn.execute_query("""
                MATCH (s:Session)
                WHERE s.startTime >= datetime($dayStart)
                  AND s.startTime < datetime($dayEnd)
                WITH s,
                     CASE WHEN s.status = 'completed' THEN 1 ELSE 0 END AS is_completed,
                     CASE WHEN s.status = 'active'    THEN 1 ELSE 0 END AS is_active,
                     CASE WHEN s.status = 'error'     THEN 1 ELSE 0 END AS is_error,
                     CASE WHEN s.status = 'abandoned'  THEN 1 ELSE 0 END AS is_abandoned
                RETURN
                    count(s)          AS total,
                    sum(is_completed) AS completed,
                    sum(is_active)    AS active,
                    sum(is_error)     AS errored,
                    sum(is_abandoned) AS abandoned,
                    avg(CASE WHEN s.endTime IS NOT NULL
                         THEN duration.inSeconds(s.startTime, s.endTime).seconds
                         ELSE null END) AS avg_dur
            """, {
                "dayStart": f"{current_date}T{current_hour:02d}:00:00Z",
                "dayEnd":   f"{current_date}T{(current_hour + 1) % 24:02d}:00:00Z",
            })

            if session_result and session_result[0]:
                row = session_result[0]
                # Deterministic metric_id for upsert
                metric_id = hashlib.sha256(
                    f"session:{current_date}:{current_hour}".encode()
                ).hexdigest()[:36]

                mysql.execute_update("""
                    INSERT INTO session_metrics
                        (metric_id, date, hour, total_sessions, active_sessions,
                         completed_sessions, abandoned_sessions, error_sessions,
                         avg_duration_seconds)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        total_sessions     = VALUES(total_sessions),
                        active_sessions    = VALUES(active_sessions),
                        completed_sessions = VALUES(completed_sessions),
                        abandoned_sessions = VALUES(abandoned_sessions),
                        error_sessions     = VALUES(error_sessions),
                        avg_duration_seconds = VALUES(avg_duration_seconds)
                """, (
                    metric_id, current_date, current_hour,
                    int(row.get("total", 0) or 0),
                    int(row.get("active", 0) or 0),
                    int(row.get("completed", 0) or 0),
                    int(row.get("abandoned", 0) or 0),
                    int(row.get("errored", 0) or 0),
                    round(float(row.get("avg_dur", 0) or 0), 2),
                ))
            logger.info("EvaluationPipeline: session_metrics persisted for %s hour %d", current_date, current_hour)
        except Exception as exc:
            logger.error("EvaluationPipeline: session_metrics persistence failed: %s", exc)

        # ── Agent metrics ────────────────────────────────────────────
        try:
            agent_result = cg.conn.execute_query("""
                MATCH (ae:AgentExecution)
                WHERE ae.startTime >= datetime($dayStart)
                  AND ae.startTime < datetime($dayEnd)
                RETURN
                    ae.agentName  AS agentName,
                    ae.agentType  AS agentType,
                    count(ae)     AS execCount,
                    sum(CASE WHEN ae.status IN ['completed', 'success'] THEN 1 ELSE 0 END) AS successCount,
                    sum(CASE WHEN ae.status = 'failed' THEN 1 ELSE 0 END) AS failCount,
                    avg(ae.duration) AS avgDur,
                    sum(ae.duration) AS totalDur
            """, {
                "dayStart": f"{current_date}T{current_hour:02d}:00:00Z",
                "dayEnd":   f"{current_date}T{(current_hour + 1) % 24:02d}:00:00Z",
            })

            for row in (agent_result or []):
                agent_name = row.get("agentName", "unknown")
                agent_type = row.get("agentType", "unknown")
                if not agent_name:
                    continue

                metric_id = hashlib.sha256(
                    f"agent:{agent_name}:{current_date}:{current_hour}".encode()
                ).hexdigest()[:36]

                # Count tool calls for this agent in this hour
                tool_count_result = cg.conn.execute_query("""
                    MATCH (ae:AgentExecution {agentName: $agentName})-[:CALLED_TOOL]->(te:ToolExecution)
                    WHERE ae.startTime >= datetime($dayStart)
                      AND ae.startTime < datetime($dayEnd)
                    RETURN count(te) AS toolCalls
                """, {
                    "agentName": agent_name,
                    "dayStart": f"{current_date}T{current_hour:02d}:00:00Z",
                    "dayEnd":   f"{current_date}T{(current_hour + 1) % 24:02d}:00:00Z",
                })
                tool_calls = int((tool_count_result[0].get("toolCalls", 0) or 0)) if tool_count_result else 0

                mysql.execute_update("""
                    INSERT INTO agent_metrics
                        (metric_id, agent_name, agent_type, execution_count,
                         success_count, failure_count, avg_execution_time_ms,
                         total_execution_time_ms, tool_call_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        execution_count       = VALUES(execution_count),
                        success_count         = VALUES(success_count),
                        failure_count         = VALUES(failure_count),
                        avg_execution_time_ms = VALUES(avg_execution_time_ms),
                        total_execution_time_ms = VALUES(total_execution_time_ms),
                        tool_call_count       = VALUES(tool_call_count)
                """, (
                    metric_id, agent_name, agent_type,
                    int(row.get("execCount", 0) or 0),
                    int(row.get("successCount", 0) or 0),
                    int(row.get("failCount", 0) or 0),
                    round(float(row.get("avgDur", 0) or 0), 2),
                    int(row.get("totalDur", 0) or 0),
                    tool_calls,
                ))
            logger.info("EvaluationPipeline: agent_metrics persisted for %s hour %d", current_date, current_hour)
        except Exception as exc:
            logger.error("EvaluationPipeline: agent_metrics persistence failed: %s", exc)

        # ── Tool metrics ─────────────────────────────────────────────
        try:
            # Map tool names to categories based on team ownership
            _TOOL_CATEGORIES = {
                "claim_lookup": "DATABASE", "claim_status": "DATABASE",
                "claim_payment_info": "DATABASE", "update_claim_status": "DATABASE",
                "member_lookup": "DATABASE", "check_eligibility": "DATABASE",
                "coverage_lookup": "DATABASE", "update_member_info": "DATABASE",
                "pa_lookup": "DATABASE", "pa_status": "DATABASE",
                "pa_requirements": "API", "approve_prior_auth": "DATABASE",
                "deny_prior_auth": "DATABASE",
                "provider_lookup": "DATABASE", "provider_search": "SEARCH",
                "network_check": "API",
                "search_policy_info": "SEARCH", "search_medical_codes": "SEARCH",
                "search_knowledge_base": "SEARCH",
            }

            tool_result = cg.conn.execute_query("""
                MATCH (te:ToolExecution)
                WHERE te.timestamp >= datetime($dayStart)
                  AND te.timestamp < datetime($dayEnd)
                RETURN
                    te.toolName   AS toolName,
                    count(te)     AS callCount,
                    sum(CASE WHEN te.status = 'success' THEN 1 ELSE 0 END) AS successCount,
                    sum(CASE WHEN te.status = 'error'   THEN 1 ELSE 0 END) AS failCount,
                    avg(te.executionTimeMs) AS avgTime,
                    sum(te.executionTimeMs) AS totalTime
            """, {
                "dayStart": f"{current_date}T{current_hour:02d}:00:00Z",
                "dayEnd":   f"{current_date}T{(current_hour + 1) % 24:02d}:00:00Z",
            })

            for row in (tool_result or []):
                tool_name = row.get("toolName", "unknown")
                if not tool_name:
                    continue

                metric_id = hashlib.sha256(
                    f"tool:{tool_name}:{current_date}:{current_hour}".encode()
                ).hexdigest()[:36]

                mysql.execute_update("""
                    INSERT INTO tool_metrics
                        (metric_id, tool_name, tool_category, call_count,
                         success_count, failure_count, avg_execution_time_ms,
                         total_execution_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        call_count              = VALUES(call_count),
                        success_count           = VALUES(success_count),
                        failure_count           = VALUES(failure_count),
                        avg_execution_time_ms   = VALUES(avg_execution_time_ms),
                        total_execution_time_ms = VALUES(total_execution_time_ms)
                """, (
                    metric_id, tool_name,
                    _TOOL_CATEGORIES.get(tool_name, "OTHER"),
                    int(row.get("callCount", 0) or 0),
                    int(row.get("successCount", 0) or 0),
                    int(row.get("failCount", 0) or 0),
                    round(float(row.get("avgTime", 0) or 0), 2),
                    int(row.get("totalTime", 0) or 0),
                ))
            logger.info("EvaluationPipeline: tool_metrics persisted for %s hour %d", current_date, current_hour)
        except Exception as exc:
            logger.error("EvaluationPipeline: tool_metrics persistence failed: %s", exc)

        # ── Set throttle ─────────────────────────────────────────────
        if redis_client:
            try:
                redis_client.setex(_THROTTLE_KEY, _THROTTLE_TTL, "1")
            except Exception as r_exc:
                logger.debug("EvaluationPipeline: Redis throttle write failed: %s", r_exc)

        logger.info("EvaluationPipeline: MySQL metrics persistence completed for %s hour %d", current_date, current_hour)

    except ImportError:
        logger.debug("EvaluationPipeline: MySQL or CG not available for metrics persistence (expected in A2A containers)")
    except Exception as exc:
        logger.error("EvaluationPipeline._persist_metrics_to_mysql failed: %s", exc)
