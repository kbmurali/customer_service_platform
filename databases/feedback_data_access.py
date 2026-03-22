"""
Feedback Data Access — stores CSR quality feedback for agent evaluation.

Each feedback record links a CSIP session to a LangFuse trace and captures
a rating plus an optional correction.  These records are the seed material
for future DPO (Direct Preference Optimization) fine-tuning datasets and
for the routing-accuracy metrics emitted by evaluation_pipeline.py.

MySQL table DDL (run once during schema migration)::

    CREATE TABLE IF NOT EXISTS agent_feedback (
        feedback_id   VARCHAR(36)   NOT NULL PRIMARY KEY,
        session_id    VARCHAR(36)   NOT NULL,
        trace_id      VARCHAR(36)   NOT NULL DEFAULT '',
        rating        ENUM('correct','incorrect','partial') NOT NULL,
        correction    TEXT          DEFAULT NULL,
        user_id       VARCHAR(64)   NOT NULL DEFAULT '',
        created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_session   (session_id),
        INDEX idx_rating    (rating),
        INDEX idx_created   (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Valid rating values — kept in sync with the ENUM definition above
VALID_RATINGS = {"correct", "incorrect", "partial"}


class FeedbackDataAccess:
    """
    Thin MySQL wrapper for the agent_feedback table.

    The class accepts a connection factory callable so it can be used with
    the project's existing MySQLConnection singleton without creating a new
    dependency.
    """

    def __init__(self, get_connection_fn):
        """
        Args:
            get_connection_fn: Zero-argument callable returning a live
                               mysql.connector connection (or equivalent).
        """
        self._get_conn = get_connection_fn

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def store_feedback(
        self,
        session_id: str,
        rating: str,
        user_id: str,
        trace_id: str = "",
        correction: Optional[str] = None,
    ) -> Optional[str]:
        """
        Persist a CSR feedback record.

        Args:
            session_id:  CSIP session the feedback relates to.
            rating:      'correct', 'incorrect', or 'partial'.
            user_id:     CSR who submitted the feedback.
            trace_id:    LangFuse trace ID (empty if not available).
            correction:  Free-text correction supplied by the CSR.

        Returns:
            feedback_id (UUID string) on success, None on failure.
        """
        if rating not in VALID_RATINGS:
            logger.error(
                "FeedbackDataAccess.store_feedback: invalid rating '%s'. "
                "Must be one of %s",
                rating, sorted(VALID_RATINGS),
            )
            return None

        feedback_id = str(uuid.uuid4())
        sql = """
            INSERT INTO agent_feedback
                (feedback_id, session_id, trace_id, rating, correction,
                 user_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        try:
            self._get_conn().execute_update(sql, (
                feedback_id,
                session_id,
                trace_id or "",
                rating,
                correction,
                user_id,
                datetime.utcnow(),
            ))
            logger.info(
                "FeedbackDataAccess: stored feedback_id=%s session=%s rating=%s",
                feedback_id, session_id, rating,
            )
            return feedback_id
        except Exception as exc:
            logger.error("FeedbackDataAccess.store_feedback failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_feedback_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all feedback records for a given session_id."""
        sql = """
            SELECT feedback_id, session_id, trace_id, rating,
                   correction, user_id, created_at
            FROM   agent_feedback
            WHERE  session_id = %s
            ORDER  BY created_at ASC
        """
        try:
            return self._get_conn().execute_query(sql, (session_id,)) or []
        except Exception as exc:
            logger.error(
                "FeedbackDataAccess.get_feedback_for_session failed: %s", exc
            )
            return []

    def get_recent_feedback(
        self, limit: int = 100, rating: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Return the most recent feedback records, optionally filtered by rating.
        Used by the evaluation pipeline to build DPO training datasets.
        """
        if rating and rating not in VALID_RATINGS:
            return []

        if rating:
            sql = """
                SELECT feedback_id, session_id, trace_id, rating,
                       correction, user_id, created_at
                FROM   agent_feedback
                WHERE  rating = %s
                ORDER  BY created_at DESC
                LIMIT  %s
            """
            params = (rating, limit)
        else:
            sql = """
                SELECT feedback_id, session_id, trace_id, rating,
                       correction, user_id, created_at
                FROM   agent_feedback
                ORDER  BY created_at DESC
                LIMIT  %s
            """
            params = (limit,)

        try:
            return self._get_conn().execute_query(sql, params) or []
        except Exception as exc:
            logger.error("FeedbackDataAccess.get_recent_feedback failed: %s", exc)
            return []

    def get_positive_feedback_rate(self, window_days: int = 7) -> float:
        """
        Return fraction of 'correct' ratings in the last window_days days.
        Returns 0.0 if no records exist.
        """
        sql = """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN rating = 'correct' THEN 1 ELSE 0 END) AS correct_count
            FROM agent_feedback
            WHERE created_at >= NOW() - INTERVAL %s DAY
        """
        try:
            rows = self._get_conn().execute_query(sql, (window_days,))
            if not rows:
                return 0.0
            row = rows[0]
            if not row or not row.get("total"):
                return 0.0
            return row["correct_count"] / row["total"]
        except Exception as exc:
            logger.error(
                "FeedbackDataAccess.get_positive_feedback_rate failed: %s", exc
            )
            return 0.0


    # ------------------------------------------------------------------
    # Experience Extraction Support
    # ------------------------------------------------------------------

    def get_unprocessed_correct_sessions(
        self, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Return ``correct``-rated sessions that have NOT yet been processed
        into the experience store.
        """
        sql = """
            SELECT af.session_id, af.rating, af.created_at
            FROM   agent_feedback af
            LEFT JOIN experience_extraction_log eel
                   ON af.session_id = eel.session_id
            WHERE  af.rating = 'correct'
              AND  eel.session_id IS NULL
            ORDER BY af.created_at DESC
            LIMIT %s
        """
        try:
            return self._get_conn().execute_query(sql, (limit,)) or []
        except Exception as exc:
            logger.error(
                "FeedbackDataAccess.get_unprocessed_correct_sessions failed: %s", exc
            )
            return []

    def mark_session_extracted(
        self, session_id: str, collection_name: str
    ) -> bool:
        """Record that a session has been processed into the experience store."""
        sql = """
            INSERT IGNORE INTO experience_extraction_log
                (session_id, collection_name, status)
            VALUES (%s, %s, %s)
        """
        try:
            self._get_conn().execute_update(sql, (
                session_id, collection_name, "extracted",
            ))
            return True
        except Exception as exc:
            logger.error(
                "FeedbackDataAccess.mark_session_extracted failed: %s", exc
            )
            return False

    # ------------------------------------------------------------------
    # Feedback Classification Support
    # ------------------------------------------------------------------

    def store_classification(
        self,
        session_id: str,
        classified_by: str,
        classification_type: str,
        notes: str = "",
    ) -> Optional[str]:
        """
        Store a root-cause classification for a low-rated session.

        Args:
            session_id:          Session being classified.
            classified_by:       User ID of the classifier.
            classification_type: One of: planning, routing, tool, synthesis,
                                 security, data_quality, retrieval.
            notes:               Free-text explanation.

        Returns:
            Classification ID on success, None on failure.
        """
        classification_id = str(uuid.uuid4())
        sql = """
            INSERT INTO feedback_classifications
                (id, session_id, classified_by, classification_type, notes, classified_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            self._get_conn().execute_update(sql, (
                classification_id, session_id, classified_by,
                classification_type, notes, datetime.utcnow(),
            ))
            logger.info(
                "FeedbackDataAccess: stored classification %s for session %s type=%s",
                classification_id, session_id, classification_type,
            )
            return classification_id
        except Exception as exc:
            logger.error(
                "FeedbackDataAccess.store_classification failed: %s", exc
            )
            return None

    def get_classified_failures(
        self, days: int = 30, min_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Return classified low-rated sessions within the look-back window.
        Joins feedback with classifications for pattern analysis.
        """
        sql = """
            SELECT af.session_id, af.rating, af.correction,
                   fc.classification_type, fc.notes, fc.classified_at
            FROM   agent_feedback af
            INNER JOIN feedback_classifications fc
                    ON af.session_id = fc.session_id
            WHERE  af.rating IN ('incorrect', 'partial')
              AND  af.created_at >= NOW() - INTERVAL %s DAY
            ORDER BY af.created_at DESC
        """
        try:
            return self._get_conn().execute_query(sql, (days,)) or []
        except Exception as exc:
            logger.error(
                "FeedbackDataAccess.get_classified_failures failed: %s", exc
            )
            return []

    # ------------------------------------------------------------------
    # Pattern Reports
    # ------------------------------------------------------------------

    def store_pattern_report(self, report: Dict[str, Any]) -> bool:
        """Store a feedback pattern analysis report."""
        sql = """
            INSERT INTO feedback_pattern_reports
                (report_id, generated_at, team, classification_type,
                 cluster_count, representative_queries, suggested_action, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            self._get_conn().execute_update(sql, (
                report.get("report_id", str(uuid.uuid4())),
                report.get("generated_at", datetime.utcnow()),
                report.get("team", "unknown"),
                report.get("classification_type", "unknown"),
                report.get("cluster_count", 0),
                report.get("representative_queries", "[]"),
                report.get("suggested_action", ""),
                report.get("status", "new"),
            ))
            return True
        except Exception as exc:
            logger.error("FeedbackDataAccess.store_pattern_report failed: %s", exc)
            return False

    def get_latest_pattern_reports(
        self, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return the most recent pattern analysis reports."""
        sql = """
            SELECT report_id, generated_at, team, classification_type,
                   cluster_count, representative_queries, suggested_action, status
            FROM   feedback_pattern_reports
            ORDER BY generated_at DESC
            LIMIT %s
        """
        try:
            return self._get_conn().execute_query(sql, (limit,)) or []
        except Exception as exc:
            logger.error(
                "FeedbackDataAccess.get_latest_pattern_reports failed: %s", exc
            )
            return []

    # ------------------------------------------------------------------
    # Prompt Change Log
    # ------------------------------------------------------------------

    def log_prompt_change(self, change: Dict[str, Any]) -> bool:
        """Record a prompt or retrieval modification with baseline metrics."""
        sql = """
            INSERT INTO prompt_change_log
                (change_id, changed_by, change_type, component,
                 description, baseline_metrics, post_change_metrics, change_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            import json as _json
            self._get_conn().execute_update(sql, (
                change.get("change_id", str(uuid.uuid4())),
                change.get("changed_by", ""),
                change.get("change_type", "planning_rule"),
                change.get("component", ""),
                change.get("description", ""),
                _json.dumps(change.get("baseline_metrics", {})),
                _json.dumps(change.get("post_change_metrics", {})),
                change.get("change_timestamp", datetime.utcnow()),
            ))
            return True
        except Exception as exc:
            logger.error("FeedbackDataAccess.log_prompt_change failed: %s", exc)
            return False

    def get_prompt_changes(
        self, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return the most recent prompt/retrieval changes."""
        sql = """
            SELECT change_id, changed_by, change_type, component,
                   description, baseline_metrics, post_change_metrics,
                   change_timestamp
            FROM   prompt_change_log
            ORDER BY change_timestamp DESC
            LIMIT %s
        """
        try:
            return self._get_conn().execute_query(sql, (limit,)) or []
        except Exception as exc:
            logger.error("FeedbackDataAccess.get_prompt_changes failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_feedback_da: Optional[FeedbackDataAccess] = None


def get_feedback_data_access() -> FeedbackDataAccess:
    """Return (and lazily create) the module-level FeedbackDataAccess singleton."""
    global _feedback_da
    if _feedback_da is None:
        try:
            from databases.connections import get_mysql
            _feedback_da = FeedbackDataAccess(get_mysql)
        except Exception as exc:
            logger.error(
                "get_feedback_data_access: failed to initialise — %s", exc
            )
            raise
    return _feedback_da
