"""
LLM Provider Factory for Health Insurance CSIP

Returns a LangChain-compliant chat model based on the configured LLM_PROVIDER.
Supported providers: openai, anthropic, bedrock (AWS Bedrock).

When LLM_PROVIDER_FALLBACK_CHAIN is set (e.g. "openai,anthropic"), the
factory wraps the primary model in a ProviderCircuitBreaker that routes
automatically to the next provider when the primary exceeds its failure
threshold — transparently, with no changes in supervisors or workers.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

ChatModel = BaseChatModel


class LLMProviderFactory:
    """Creates LangChain chat-model instances from application settings."""

    _SUPPORTED_PROVIDERS = ("openai", "anthropic", "bedrock")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm_provider(self) -> ChatModel:
        """
        Return a LangChain chat model for the configured provider.

        If LLM_PROVIDER_FALLBACK_CHAIN contains additional providers, the
        primary model is wrapped in a ProviderCircuitBreaker so that runtime
        failures automatically route to the next provider in the list.

        Raises:
            ValueError: If the primary provider is not supported or required
                        credentials are missing.
        """
        primary_label = self._settings.LLM_PROVIDER.lower().strip()
        primary_model = self._build_model(primary_label)

        # Build fallback chain if configured
        fallback_chain: List[ChatModel] = []
        chain_spec = getattr(self._settings, "LLM_PROVIDER_FALLBACK_CHAIN", "")
        if chain_spec:
            for label in [p.strip().lower() for p in chain_spec.split(",")]:
                if label == primary_label or not label:
                    continue
                try:
                    fallback_chain.append(self._build_model(label))
                    logger.info("LLMProviderFactory: added fallback provider=%s", label)
                except Exception as exc:
                    logger.warning(
                        "LLMProviderFactory: fallback provider=%s skipped (%s)",
                        label, exc,
                    )

        if not fallback_chain:
            return primary_model

        # Wrap in circuit breaker for automatic runtime failover
        try:
            from llm_providers.provider_circuit_breaker import ProviderCircuitBreaker
            from config.settings import get_settings as _gs
            s = _gs()
            return ProviderCircuitBreaker(
                primary=primary_model,
                fallback_chain=fallback_chain,
                provider_name=primary_label,
                redis_host=s.REDIS_HOST,
                redis_port=s.REDIS_PORT,
                failure_threshold=getattr(s, "LLM_CB_FAILURE_THRESHOLD", 5),
                open_duration_seconds=getattr(s, "LLM_CB_OPEN_DURATION_SECONDS", 120),
            )
        except Exception as exc:
            logger.warning(
                "LLMProviderFactory: ProviderCircuitBreaker unavailable (%s), "
                "using primary model without failover",
                exc,
            )
            return primary_model

    # ------------------------------------------------------------------
    # Builder dispatch
    # ------------------------------------------------------------------

    def _build_model(self, provider: str) -> ChatModel:
        """Build a single model instance for the given provider label."""
        if provider not in self._SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{provider}'. "
                f"Choose from: {', '.join(self._SUPPORTED_PROVIDERS)}"
            )
        builder = {
            "openai":    self._build_openai,
            "anthropic": self._build_anthropic,
            "bedrock":   self._build_bedrock,
        }[provider]
        model = builder()
        logger.info(
            "Initialised LLM provider=%s  model=%s  temperature=%s  max_tokens=%s",
            provider,
            self._settings.LLM_MODEL,
            self._settings.LLM_TEMPERATURE,
            self._settings.LLM_MAX_TOKENS,
        )
        return model

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_openai(self) -> ChatModel:
        if not self._settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self._settings.LLM_MODEL,
            temperature=self._settings.LLM_TEMPERATURE,
            max_tokens=self._settings.LLM_MAX_TOKENS,
            api_key=self._settings.OPENAI_API_KEY,
        )

    def _build_anthropic(self) -> ChatModel:
        if not self._settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic'.")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name=self._settings.LLM_MODEL,
            temperature=self._settings.LLM_TEMPERATURE,
            max_tokens=self._settings.LLM_MAX_TOKENS,
            api_key=self._settings.ANTHROPIC_API_KEY,
        )

    def _build_bedrock(self) -> ChatModel:
        if not self._settings.AWS_ACCESS_KEY_ID:
            raise ValueError("AWS_ACCESS_KEY_ID is required when LLM_PROVIDER is 'bedrock'.")
        if not self._settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError("AWS_SECRET_ACCESS_KEY is required when LLM_PROVIDER is 'bedrock'.")
        if not self._settings.AWS_REGION:
            raise ValueError("AWS_REGION is required when LLM_PROVIDER is 'bedrock'.")
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(
            model=self._settings.LLM_MODEL,
            temperature=self._settings.LLM_TEMPERATURE,
            max_tokens=self._settings.LLM_MAX_TOKENS,
            region_name=self._settings.AWS_REGION,
            credentials_profile_name=None,
            aws_access_key_id=self._settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self._settings.AWS_SECRET_ACCESS_KEY,
        )


# ---------------------------------------------------------------------------
# Module-level convenience accessor (kept lightweight — no @lru_cache so
# tests can reset between runs by simply calling get_factory() again)
# ---------------------------------------------------------------------------

def get_factory() -> LLMProviderFactory:
    """Return a LLMProviderFactory built from application settings."""
    return LLMProviderFactory(get_settings())
