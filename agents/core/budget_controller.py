"""
Token Budget Controller.

Enforces per-user and per-organization daily token spend limits before
the CentralSupervisor is invoked.  Sits as Control 3.5 in the
RequestProcessor pipeline — between rate limiting and input sanitization.

Architecture:
  - Limits stored in MySQL table token_budget_limits (keyed by user_role).
  - Rolling 24-hour usage counters stored in Redis DB 3 (shared with
    MetricsPersister so the REST API can read them).
  - Usage is updated by MetricsPersister.update_user_token_tallies() which
    runs every 30 seconds, meaning enforcement has a ~30s grace window.
  - Setting TOKEN_BUDGET_ENFORCEMENT_ENABLED=false (default) makes every
    check a no-op so existing deployments are unaffected.

MySQL table DDL (run once)::

    CREATE TABLE IF NOT EXISTS token_budget_limits (
        role_name       VARCHAR(64)  NOT NULL PRIMARY KEY,
        daily_limit     INT          NOT NULL DEFAULT 100000,
        updated_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    -- Seed defaults
    INSERT IGNORE INTO token_budget_limits (role_name, daily_limit) VALUES
        ('claims_supervisor',  200000),
        ('member_supervisor',  200000),
        ('pa_supervisor',      200000),
        ('provider_supervisor',200000),
        ('csr_tier1',          100000),
        ('csr_tier2',          150000),
        ('admin',              500000);
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Redis key templates
_USER_KEY  = "token_budget:user:{user_id}"
_ORG_KEY   = "token_budget:org:{org_id}"
_TTL       = 86400   # 24 hours


class BudgetExceededError(Exception):
    """Raised when a user or org has exceeded their daily token budget."""
    def __init__(self, scope: str, used: int, limit: int):
        super().__init__(
            f"Daily token budget exceeded for {scope}: "
            f"used={used:,} limit={limit:,}"
        )
        self.scope = scope
        self.used  = used
        self.limit = limit


class TokenBudgetController:
    """
    Checks and records token usage against per-role daily limits.

    All public methods are safe to call even when Redis or MySQL are
    unavailable — they log a warning and return without raising.
    """

    def __init__(self, redis_client, get_connection_fn, default_daily_limit: int = 100_000):
        self._redis       = redis_client
        self._get_conn    = get_connection_fn
        self._default_lim = default_daily_limit
        self._limit_cache: dict[str, int] = {}  # role -> limit (in-process cache)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_budget(self, user_id: str, user_role: str) -> None:
        """
        Check whether user_id has budget remaining.

        Raises BudgetExceededError if the user has consumed their daily limit.
        Does nothing (logs warning) if Redis is unavailable.
        """
        limit = self._get_limit(user_role)
        used  = self._get_used(user_id)
        if used >= limit:
            raise BudgetExceededError(
                scope=f"user={user_id} role={user_role}",
                used=used,
                limit=limit,
            )

    def record_usage(self, user_id: str, tokens: int) -> None:
        """
        Add tokens to the user's rolling 24-hour counter in Redis.

        Called by MetricsPersister; also callable directly for synchronous
        enforcement in the future.
        """
        if tokens <= 0:
            return
        key = _USER_KEY.format(user_id=user_id)
        try:
            pipe = self._redis.pipeline()
            pipe.incrby(key, tokens)
            pipe.expire(key, _TTL)
            pipe.execute()
        except Exception as exc:
            logger.warning("TokenBudgetController.record_usage failed: %s", exc)

    def get_usage(self, user_id: str) -> int:
        """Return current 24-hour token count for user_id."""
        return self._get_used(user_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_used(self, user_id: str) -> int:
        key = _USER_KEY.format(user_id=user_id)
        try:
            val = self._redis.get(key)
            return int(val) if val else 0
        except Exception as exc:
            logger.warning("TokenBudgetController._get_used failed: %s", exc)
            return 0

    def _get_limit(self, user_role: str) -> int:
        """Fetch limit from in-process cache, MySQL fallback, then default."""
        role = user_role.lower()
        if role in self._limit_cache:
            return self._limit_cache[role]

        try:
            conn   = self._get_conn()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT daily_limit FROM token_budget_limits WHERE role_name=%s",
                (role,),
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                limit = int(row["daily_limit"])
                self._limit_cache[role] = limit
                return limit
        except Exception as exc:
            logger.warning("TokenBudgetController._get_limit MySQL error: %s", exc)

        # Fall back to configured default
        return self._default_lim


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_controller: Optional[TokenBudgetController] = None


def get_budget_controller() -> Optional[TokenBudgetController]:
    """
    Return (and lazily create) the module-level TokenBudgetController.

    Returns None if TOKEN_BUDGET_ENFORCEMENT_ENABLED is False, so callers
    can short-circuit with a simple ``if get_budget_controller()`` check.
    """
    global _controller
    try:
        from config.settings import get_settings
        settings = get_settings()
        if not getattr(settings, "TOKEN_BUDGET_ENFORCEMENT_ENABLED", False):
            return None
    except Exception:
        return None

    if _controller is None:
        try:
            import redis as redis_lib
            from databases.connections import get_mysql_connection
            from config.settings import get_settings
            settings = get_settings()
            r = redis_lib.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=3,
                decode_responses=True,
            )
            _controller = TokenBudgetController(
                redis_client=r,
                get_connection_fn=get_mysql_connection,
                default_daily_limit=getattr(
                    settings, "TOKEN_BUDGET_DEFAULT_DAILY_LIMIT", 100_000
                ),
            )
        except Exception as exc:
            logger.error("get_budget_controller: init failed — %s", exc)
            return None

    return _controller
