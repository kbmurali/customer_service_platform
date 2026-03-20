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
