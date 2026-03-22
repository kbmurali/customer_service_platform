"""
Feedback Pattern Analyzer — surfaces actionable improvement patterns
from low-rated sessions.

Clusters ``incorrect`` and ``partial`` feedback records by team and
failure classification, producing structured reports that the Feedback
Learning Dashboard consumes.  Each report identifies recurring query
patterns, the affected team, the common failure mode, and a suggested
action (prompt refinement, retrieval update, or data fix).

Designed to run daily (configurable) or on-demand via a management
API endpoint.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics — imported from the central registry to avoid
# duplicate registration errors.
# ---------------------------------------------------------------------------
try:
    from observability.prometheus_metrics import (
        pattern_reports_generated_total as pattern_reports_generated,
    )
    _PROM = True
except ImportError:
    _PROM = False


# ---------------------------------------------------------------------------
# Classification types (kept in sync with MySQL ENUM)
# ---------------------------------------------------------------------------
VALID_CLASSIFICATIONS = {
    "planning", "routing", "tool", "synthesis",
    "security", "data_quality", "retrieval",
}


def run_analysis(window_days: int = 30, min_cluster_size: int = 3) -> List[Dict[str, Any]]:
    """
    Analyze low-rated sessions and produce failure pattern reports.

    Args:
        window_days:      Look-back window for feedback records.
        min_cluster_size: Minimum number of sessions in a group to
                          generate a report (filters noise).

    Returns:
        List of report dicts, each stored in ``feedback_pattern_reports``.
    """
    reports: List[Dict[str, Any]] = []

    try:
        from databases.feedback_data_access import get_feedback_data_access
        from databases.context_graph_data_access import get_cg_data_access

        feedback_da = get_feedback_data_access()
        cg = get_cg_data_access()

        # ── Step 1: Get classified low-rated sessions ─────────────────
        failures = feedback_da.get_classified_failures(
            days=window_days,
            min_count=1,
        )
        if not failures:
            logger.info("PatternAnalyzer: no classified failures in window")
            return reports

        # ── Step 2: Group by (team, classification_type) ──────────────
        groups: Dict[str, List[Dict]] = defaultdict(list)

        for row in failures:
            session_id = row.get("session_id", "")
            classification = row.get("classification_type", "unknown")

            # Determine team from CG plan
            team = "unknown"
            try:
                plan = cg.get_active_plan(session_id)
                if plan:
                    steps = plan.get("steps", [])
                    # Use the most common team assignment
                    team_counts: Dict[str, int] = defaultdict(int)
                    for step in steps:
                        agent = step.get("agent", "") or step.get("worker", "")
                        if agent:
                            team_counts[agent] += 1
                    if team_counts:
                        team = max(team_counts, key=team_counts.get)
            except Exception:
                pass

            key = f"{team}|{classification}"
            groups[key].append({
                "session_id": session_id,
                "query": row.get("query", ""),
                "notes": row.get("notes", ""),
                "classified_at": str(row.get("classified_at", "")),
            })

        # ── Step 3: Generate reports for groups above threshold ───────
        for key, sessions in groups.items():
            if len(sessions) < min_cluster_size:
                continue

            team, classification = key.split("|", 1)

            # Collect representative queries (first 5)
            representative = [s.get("query", s.get("session_id", ""))
                              for s in sessions[:5]]

            # Suggest action based on classification type
            action_map = {
                "planning": "Review and refine the Central Supervisor PLANNING_SYSTEM_PROMPT. "
                            "Consider adding an explicit rule for this query pattern.",
                "routing": "Check the team supervisor's routing prompt and Agent Card "
                           "skills description for the affected team.",
                "tool": "Inspect the MCP tool function and its Knowledge Graph query. "
                        "Verify the data exists and the query returns correct results.",
                "synthesis": "Review the worker's ReAct agent behavior. The tool returned "
                             "correct data but the LLM summary was inaccurate.",
                "security": "A security control blocked or modified the query/response "
                            "incorrectly. Review control configuration.",
                "data_quality": "The underlying data in the Knowledge Graph or MySQL is "
                                "incorrect or outdated. Fix the source data.",
                "retrieval": "The Chroma collection is missing relevant documents or "
                             "existing documents have poor embedding quality.",
            }
            suggested_action = action_map.get(classification, "Investigate manually.")

            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "team": team,
                "classification_type": classification,
                "cluster_count": len(sessions),
                "representative_queries": json.dumps(representative),
                "suggested_action": suggested_action,
                "status": "new",
            }

            # Store in MySQL
            try:
                feedback_da.store_pattern_report(report)
                reports.append(report)
            except Exception as store_exc:
                logger.error("PatternAnalyzer: failed to store report: %s", store_exc)

        if _PROM and reports:
            pattern_reports_generated.inc(len(reports))

        logger.info(
            "PatternAnalyzer: generated %d reports from %d groups",
            len(reports), len(groups),
        )

    except Exception as exc:
        logger.error("PatternAnalyzer.run_analysis failed: %s", exc)

    return reports
