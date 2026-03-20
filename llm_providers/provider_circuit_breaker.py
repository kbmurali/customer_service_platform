"""
Provider Circuit Breaker for LLM failover.

Wraps a BaseChatModel and tracks per-provider failure rates in Redis.
When failures cross a configurable threshold the circuit opens and
subsequent calls are routed to the next provider in the fallback chain
— transparently, with no changes required in supervisors or workers.

Usage is internal to LLMProviderFactory.  External callers continue to
use the BaseChatModel interface unchanged::

    from llm_providers.llm_provider_factory import get_factory
    llm = get_factory().get_llm_provider()   # may return a wrapped model
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator, List, Optional

import redis as redis_lib
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REDIS_KEY_PREFIX = "llm_cb"          # llm_cb:{provider}:failures
_WINDOW_SECONDS   = 60                 # rolling window for failure counts
_DEFAULT_THRESHOLD = 5                 # failures in window before opening
_OPEN_DURATION     = 120               # seconds to keep circuit open


class ProviderCircuitBreaker(BaseChatModel):
    """
    A BaseChatModel wrapper that implements a per-provider circuit breaker.

    When the primary provider's error rate crosses _DEFAULT_THRESHOLD within
    _WINDOW_SECONDS, the breaker opens and calls are forwarded to the next
    model in the fallback_chain list.  After _OPEN_DURATION seconds the
    breaker half-opens and tries the primary provider again.

    All LangChain callbacks, config, and streaming are forwarded transparently.
    """

    # Pydantic fields required by BaseChatModel
    primary: BaseChatModel
    fallback_chain: List[BaseChatModel] = []
    provider_name: str = "primary"
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 4          # dedicated DB so we don't pollute other namespaces
    failure_threshold: int = _DEFAULT_THRESHOLD
    open_duration_seconds: int = _OPEN_DURATION
    window_seconds: int = _WINDOW_SECONDS

    # Non-pydantic internal state (excluded from serialisation)
    _redis: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return f"circuit_breaker({self.provider_name})"

    def _get_redis(self):
        if self._redis is None:
            try:
                self._redis = redis_lib.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
            except Exception as exc:
                logger.warning("ProviderCircuitBreaker: Redis unavailable: %s", exc)
        return self._redis

    # ------------------------------------------------------------------
    # Circuit state helpers
    # ------------------------------------------------------------------

    def _failure_key(self, provider: str) -> str:
        return f"{_REDIS_KEY_PREFIX}:{provider}:failures"

    def _open_key(self, provider: str) -> str:
        return f"{_REDIS_KEY_PREFIX}:{provider}:open"

    def _is_open(self, provider: str) -> bool:
        r = self._get_redis()
        if r is None:
            return False
        try:
            return bool(r.exists(self._open_key(provider)))
        except Exception:
            return False

    def _record_failure(self, provider: str) -> None:
        r = self._get_redis()
        if r is None:
            return
        try:
            pipe = r.pipeline()
            fkey = self._failure_key(provider)
            pipe.incr(fkey)
            pipe.expire(fkey, self.window_seconds)
            pipe.execute()
            count = int(r.get(fkey) or 0)
            if count >= self.failure_threshold:
                logger.warning(
                    "ProviderCircuitBreaker: opening circuit for '%s' "
                    "(%d failures in %ds window)",
                    provider, count, self.window_seconds,
                )
                r.setex(self._open_key(provider), self.open_duration_seconds, "1")
        except Exception as exc:
            logger.debug("ProviderCircuitBreaker: Redis write failed: %s", exc)

    def _record_success(self, provider: str) -> None:
        r = self._get_redis()
        if r is None:
            return
        try:
            r.delete(self._failure_key(provider))
            r.delete(self._open_key(provider))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _active_model(self) -> tuple[str, BaseChatModel]:
        """Return (provider_label, model) for the first non-open provider."""
        if not self._is_open(self.provider_name):
            return self.provider_name, self.primary
        for i, fallback in enumerate(self.fallback_chain):
            label = getattr(fallback, "_llm_type", f"fallback_{i}")
            if not self._is_open(label):
                logger.info(
                    "ProviderCircuitBreaker: routing to fallback '%s' "
                    "(primary '%s' circuit open)",
                    label, self.provider_name,
                )
                return label, fallback
        # All circuits open — try primary anyway (fail loudly)
        logger.error(
            "ProviderCircuitBreaker: all provider circuits open, attempting primary"
        )
        return self.provider_name, self.primary

    # ------------------------------------------------------------------
    # BaseChatModel implementation
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        label, model = self._active_model()
        try:
            result = model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self._record_success(label)
            return result
        except Exception as exc:
            self._record_failure(label)
            raise

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        label, model = self._active_model()
        try:
            result = await model._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self._record_success(label)
            return result
        except Exception as exc:
            self._record_failure(label)
            raise
