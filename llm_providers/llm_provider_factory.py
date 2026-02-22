"""
LLM Provider Factory for Health Insurance CSIP

Returns a LangChain-compliant chat model based on the configured LLM_PROVIDER.
Supported providers: openai, anthropic, bedrock (AWS Bedrock).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Type alias for the chat models this factory can return
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
        """Return a LangChain chat model for the configured provider.

        Raises:
            ValueError: If the provider is not supported or required
                        credentials are missing.
        """
        provider = self._settings.LLM_PROVIDER.lower().strip()

        if provider not in self._SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{provider}'. "
                f"Choose from: {', '.join(self._SUPPORTED_PROVIDERS)}"
            )

        builder = {
            "openai": self._build_openai,
            "anthropic": self._build_anthropic,
            "bedrock": self._build_bedrock,
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
        """Build a ChatOpenAI instance."""
        if not self._settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'."
            )

        from langchain_openai import ChatOpenAI  # lazy import

        return ChatOpenAI(
            model=self._settings.LLM_MODEL,
            temperature=self._settings.LLM_TEMPERATURE,
            max_tokens=self._settings.LLM_MAX_TOKENS,
            api_key=self._settings.OPENAI_API_KEY,
        )

    def _build_anthropic(self) -> ChatModel:
        """Build a ChatAnthropic instance."""
        if not self._settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic'."
            )

        from langchain_anthropic import ChatAnthropic  # lazy import

        return ChatAnthropic(
            model_name=self._settings.LLM_MODEL,
            temperature=self._settings.LLM_TEMPERATURE,
            max_tokens=self._settings.LLM_MAX_TOKENS,
            timeout=None,
            stop=None,
            api_key=self._settings.ANTHROPIC_API_KEY,
        )

    def _build_bedrock(self) -> ChatModel:
        """Build a ChatBedrockConverse instance (AWS Bedrock)."""
        if not self._settings.AWS_ACCESS_KEY_ID:
            raise ValueError(
                "AWS_ACCESS_KEY_ID is required when LLM_PROVIDER is 'bedrock'."
            )
        if not self._settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError(
                "AWS_SECRET_ACCESS_KEY is required when LLM_PROVIDER is 'bedrock'."
            )

        from langchain_aws import ChatBedrockConverse  # lazy import

        return ChatBedrockConverse(
            model=self._settings.LLM_MODEL,
            temperature=self._settings.LLM_TEMPERATURE,
            max_tokens=self._settings.LLM_MAX_TOKENS,
            region_name=self._settings.AWS_REGION,
            credentials_profile_name=None,  # use explicit keys
            aws_access_key_id=self._settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self._settings.AWS_SECRET_ACCESS_KEY,
        )


# ----------------------------------------------------------------------
# Module-level convenience accessor
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_factory() -> LLMProviderFactory:
    """Return a cached singleton *LLMProviderFactory* instance.

    Usage::

        from llm_provider_factory import get_factory

        factory = get_factory()
        llm     = factory.get_llm_provider()
    """
    return LLMProviderFactory(get_settings())