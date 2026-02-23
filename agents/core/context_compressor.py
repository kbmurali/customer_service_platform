"""
Context Compressor — LLMLingua-based Token Optimization (v24)
==============================================================
Provides token-level compression for conversation history, semantic
context (Chroma policy/FAQ documents), and cross-agent delegation
payloads before they are injected into LLM prompts.

    - Uses the ``llmlingua`` library directly via ``PromptCompressor``
      instead of the LangChain wrapper, giving finer control over
      compression parameters and token budgets.
    - Both ``ConversationHistoryCompressor`` and ``SemanticContextCompressor``
      share the same ``LLMLinguaEngine`` singleton so the underlying
      small language model (GPT-2) is loaded only once.
    - Adds ``tiktoken``-based token counting for accurate budget tracking.
    - Pulls configuration from ``config.settings`` for consistency.

Compression Approaches:
    1. Conversation History Compression (hierarchical summarization):
       Older messages are compressed via LLMLingua token pruning; the
       most recent 1-2 turns are kept verbatim.

    2. Semantic Context Compression (document-level):
       Policy documents and FAQ entries from Chroma are compressed
       using LLMLingua before injection into the planning prompt.

    3. Cross-Agent Context Compression:
       Accumulated state (plan + conversation summary + prior results)
       is compressed before delegating to a remote A2A agent.

References:
    - LLMLingua: https://github.com/microsoft/LLMLingua
    - LLMLingua-2: https://github.com/microsoft/LLMLingua/tree/main/llmlingua2
"""

import logging
from typing import Any, Dict, List, Optional
from functools import lru_cache

from llmlingua import PromptCompressor
import tiktoken

from config.settings import get_settings, Settings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CompressionConfig:
    """
    Configuration for context compression behavior.

    Values can be overridden by ``config.settings`` at startup via
    ``get_compression_config()``.
    """

    # ── Conversation history ──
    MAX_VERBATIM_TURNS: int = 2
    MAX_HISTORY_MESSAGES: int = 10
    SUMMARY_MAX_TOKENS: int = 150

    # ── LLMLingua engine ──
    MODEL_NAME: str = "openai-community/gpt2"
    DEVICE_MAP: str = "cpu"
    USE_LLMLINGUA: bool = True

    # ── Compression ratios ──
    CONVERSATION_RATE: float = 0.5       # 50 % token reduction for history
    SEMANTIC_RATE: float = 0.3           # 70 % token reduction for Chroma docs
    CROSS_AGENT_RATE: float = 0.4        # 60 % token reduction for A2A context
    PLAN_COMPLETED_RATE: float = 0.3     # 70 % reduction for completed goals

    # ── Domain-specific tokens that must never be removed ──
    FORCE_TOKENS: List[str] = [
        "member", "claim", "eligibility", "coverage", "deductible",
        "copay", "coinsurance", "prior authorization", "PA",
        "provider", "network", "in-network", "out-of-network",
        "denied", "approved", "pending", "appeal",
        "ICD-10", "CPT", "diagnosis", "procedure",
    ]

    # ── Cross-agent budget ──
    CROSS_AGENT_MAX_TOKENS: int = 500


def get_compression_config() -> CompressionConfig:
    """
    Build a ``CompressionConfig`` from application settings.

    Falls back to class defaults if settings are not available.
    """
    config = CompressionConfig()
    
    try:
        settings: Settings = get_settings()
        
        config.USE_LLMLINGUA = settings.CONTEXT_COMPRESSION_ENABLED
    
        config.MAX_VERBATIM_TURNS = settings.CONTEXT_COMPRESSION_MAX_VERBATIM_TURNS
        
        config.CONVERSATION_RATE = settings.CONTEXT_COMPRESSION_RATE
        
        config.MODEL_NAME = settings.CONTEXT_COMPRESSION_MODEL
        
        config.DEVICE_MAP = settings.CONTEXT_COMPRESSION_DEVICE
        
        config.SEMANTIC_RATE = settings.CONTEXT_COMPRESSION_SEMANTIC_RATE

        config.CROSS_AGENT_RATE = settings.CONTEXT_COMPRESSION_CROSS_AGENT_RATE
    except Exception:
        pass
    return config


# ---------------------------------------------------------------------------
# Token Counting Utility
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in *text* using ``tiktoken``.

    Falls back to a whitespace-based estimate if ``tiktoken`` is not
    installed.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except ImportError:
        # Rough estimate: 1 token ≈ 0.75 words
        return int(len(text.split()) / 0.75)
    except Exception:
        return len(text.split())


# ---------------------------------------------------------------------------
# LLMLingua Engine (singleton)
# ---------------------------------------------------------------------------

class LLMLinguaEngine:
    """
    Thin wrapper around ``llmlingua.PromptCompressor``.

    The engine is lazily initialized on first use so that the GPT-2
    model is loaded only once and shared across all compressor classes.
    If the ``llmlingua`` package is not installed, all calls transparently
    fall back to extractive compression.
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self._compressor: Optional[PromptCompressor] = None          # llmlingua.PromptCompressor instance
        self._initialized: bool = False
        self._available: bool = False     # True if llmlingua loaded OK

    # ── Lazy init ──────────────────────────────────────────────────────

    def _initialize(self) -> bool:
        """
        Attempt to load the LLMLingua ``PromptCompressor``.

        Returns ``True`` if the engine is ready, ``False`` otherwise.
        """
        if self._initialized:
            return self._available

        self._initialized = True

        if not self.config.USE_LLMLINGUA:
            logger.info("LLMLingua disabled via configuration.")
            return False

        try:
            self._compressor = PromptCompressor(
                model_name=self.config.MODEL_NAME,
                device_map=self.config.DEVICE_MAP,
                use_llmlingua2=True,       # Use LLMLingua-2 for better quality
            )
            self._available = True
            logger.info(
                "LLMLingua engine initialized (model=%s, device=%s)",
                self.config.MODEL_NAME,
                self.config.DEVICE_MAP,
            )
            return True

        except ImportError:
            logger.warning(
                "llmlingua package not installed. "
                "Install with: pip install llmlingua. "
                "Falling back to extractive compression."
            )
            return False

        except Exception as exc:
            logger.warning("LLMLingua init failed: %s. Using fallback.", exc)
            return False

    # ── Core compression ───────────────────────────────────────────────

    def compress(
        self,
        text: str,
        rate: float = 0.5,
        force_tokens: Optional[List[str]] = None,
        target_token: int = -1,
    ) -> str:
        """
        Compress *text* using LLMLingua token-level pruning.

        Args:
            text:         The text to compress.
            rate:         Target compression ratio (0.5 = keep 50 %).
            force_tokens: Domain tokens that must be preserved.
            target_token: If > 0, override *rate* with an absolute token budget.

        Returns:
            Compressed text.  If LLMLingua is unavailable, returns the
            result of ``_extractive_fallback``.
        """
        if not text or count_tokens(text) < 30:
            return text  # Too short to compress meaningfully

        tokens_before = count_tokens(text)

        if self._initialize() and self._compressor is not None:
            compressed = self._llmlingua_compress(
                text, rate, force_tokens, target_token,
            )
        else:
            compressed = self._extractive_fallback(text, rate)

        tokens_after = count_tokens(compressed)
        reduction = (1 - tokens_after / max(tokens_before, 1)) * 100
        logger.debug(
            "Compression: %d → %d tokens (%.1f%% reduction)",
            tokens_before, tokens_after, reduction,
        )
        return compressed

    def _llmlingua_compress(
        self,
        text: str,
        rate: float,
        force_tokens: Optional[List[str]],
        target_token: int,
    ) -> str:
        """
        Invoke the real LLMLingua ``PromptCompressor``.

        Uses ``compress_prompt`` which accepts a list of demonstration
        strings and an instruction.  For general text we pass the full
        text as the *context* list.
        """
        try:
            kwargs: Dict[str, Any] = {
                "context": [text],
                "rate": rate,
                "force_tokens": force_tokens or self.config.FORCE_TOKENS,
                "force_reserve_digit": True,   # Preserve numbers
                "drop_consecutive": True,      # Remove repeated whitespace
            }
            if target_token > 0:
                kwargs["target_token"] = target_token

            result = self._compressor.compress_prompt(**kwargs)

            compressed_prompt = result.get("compressed_prompt", text)
            origin_tokens = result.get("origin_tokens", 0)
            compressed_tokens = result.get("compressed_tokens", 0)

            logger.debug(
                "LLMLingua: origin=%d compressed=%d ratio=%.2f",
                origin_tokens, compressed_tokens,
                result.get("ratio", 0.0),
            )
            return compressed_prompt

        except Exception as exc:
            logger.warning("LLMLingua compression failed: %s. Using fallback.", exc)
            return self._extractive_fallback(text, rate)

    # ── Fallback ───────────────────────────────────────────────────────

    @staticmethod
    def _extractive_fallback(text: str, rate: float) -> str:
        """
        Extractive fallback when LLMLingua is not available.

        Strategy: keep the first 60 % and last 20 % of sentences, which
        typically contain the most important information (topic intro
        and conclusions / key findings).
        """
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        if len(sentences) <= 3:
            return text

        target_count = max(3, int(len(sentences) * rate))
        first_n = int(target_count * 0.6)
        last_n = target_count - first_n

        kept = sentences[:first_n] + sentences[-last_n:]
        compressed = ". ".join(kept)
        if not compressed.endswith("."):
            compressed += "."

        logger.debug(
            "Extractive fallback: %d → %d sentences", len(sentences), len(kept),
        )
        return compressed


# ---------------------------------------------------------------------------
# Shared engine singleton
# ---------------------------------------------------------------------------

_engine: Optional[LLMLinguaEngine] = None


def _get_engine() -> LLMLinguaEngine:
    """Return the shared ``LLMLinguaEngine`` singleton."""
    global _engine
    if _engine is None:
        _engine = LLMLinguaEngine(get_compression_config())
    return _engine


# ---------------------------------------------------------------------------
# Conversation History Compressor
# ---------------------------------------------------------------------------

class ConversationHistoryCompressor:
    """
    Compresses conversation history using hierarchical summarization
    powered by LLMLingua.

    Strategy:
        1. Keep the most recent N turns (default 2) verbatim — these
           provide immediate context for the current routing decision.
        2. Compress older turns via LLMLingua token pruning into a
           compact ``SystemMessage``.
        3. Return a ``list[BaseMessage]`` ready for the routing prompt.

    This replaces the v22 approach of truncating each message to 100
    characters, which lost semantic content and produced incoherent
    context fragments.
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or get_compression_config()
        self._engine = _get_engine()

    def compress_history(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[BaseMessage]:
        """
        Compress conversation history into a list of LangChain messages.

        Args:
            messages: Raw conversation history from Neo4j Context Graph.
                      Each dict has keys: ``role``, ``content``, ``timestamp``.

        Returns:
            List of ``BaseMessage`` objects:
                - ``SystemMessage`` with compressed summary of older turns
                  (if any exist beyond the verbatim window)
                - Recent turns as their proper message types
        """
        if not messages:
            return []

        # Limit to max configured messages
        messages = messages[-self.config.MAX_HISTORY_MESSAGES:]

        # Split into older (to compress) and recent (to keep verbatim)
        split_point = max(0, len(messages) - self.config.MAX_VERBATIM_TURNS)
        older_messages = messages[:split_point]
        recent_messages = messages[split_point:]

        result: List[BaseMessage] = []

        # ── Compress older messages via LLMLingua ──
        if older_messages:
            summary = self._compress_older_messages(older_messages)
            if summary:
                result.append(SystemMessage(
                    content=(
                        f"[Compressed conversation history — "
                        f"{len(older_messages)} earlier messages]\n{summary}"
                    )
                ))

        # ── Keep recent messages verbatim as proper types ──
        role_map = {
            "user": HumanMessage,
            "human": HumanMessage,
            "assistant": AIMessage,
            "ai": AIMessage,
            "system": SystemMessage,
        }

        for msg in recent_messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            msg_class = role_map.get(role, HumanMessage)
            result.append(msg_class(content=content))

        return result

    def _compress_older_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Compress older conversation messages using LLMLingua.

        Concatenates older messages into a single text block, then
        applies LLMLingua token pruning to produce a compact summary
        that preserves the semantic intent of the conversation.
        """
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        full_text = "\n".join(lines)

        return self._engine.compress(
            full_text,
            rate=self.config.CONVERSATION_RATE,
            target_token=self.config.SUMMARY_MAX_TOKENS,
        )


# ---------------------------------------------------------------------------
# Semantic Context Compressor (for Chroma policy/FAQ documents)
# ---------------------------------------------------------------------------

class SemanticContextCompressor:
    """
    Compresses Chroma policy documents and FAQ entries before injection
    into the planning prompt (``create_plan_node``).

    In v22, raw JSON from Chroma (2 policy docs + 2 FAQ entries) was
    included verbatim in the planning prompt, consuming significant
    tokens.  This compressor applies LLMLingua token pruning to reduce
    the document content while preserving policy-relevant information
    (coverage rules, exclusions, requirements).
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or get_compression_config()
        self._engine = _get_engine()

    def compress_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Compress a list of Chroma documents.

        Each document dict is expected to have at least a ``content`` or
        ``page_content`` field.  The compressed version retains the same
        structure with reduced content.

        Args:
            documents: List of document dicts from Chroma search.
            query:     The user query (unused currently; reserved for
                       future relevance-aware compression).

        Returns:
            List of document dicts with compressed content.
        """
        compressed_docs = []
        total_before = 0
        total_after = 0

        for doc in documents:
            content = doc.get("content") or doc.get("page_content", "")
            if not content:
                compressed_docs.append(doc)
                continue

            tokens_before = count_tokens(content)
            total_before += tokens_before

            compressed_content = self._engine.compress(
                content,
                rate=self.config.SEMANTIC_RATE,
            )

            tokens_after = count_tokens(compressed_content)
            total_after += tokens_after

            compressed_doc = dict(doc)
            if "content" in compressed_doc:
                compressed_doc["content"] = compressed_content
            if "page_content" in compressed_doc:
                compressed_doc["page_content"] = compressed_content
            compressed_docs.append(compressed_doc)

        if total_before > 0:
            logger.info(
                "Semantic compression: %d → %d tokens across %d docs (%.1f%% reduction)",
                total_before, total_after, len(documents),
                (1 - total_after / total_before) * 100,
            )

        return compressed_docs

    def compress_text(self, text: str) -> str:
        """Compress a single text string (convenience method)."""
        return self._engine.compress(
            text,
            rate=self.config.SEMANTIC_RATE,
        )


# ---------------------------------------------------------------------------
# Cross-Agent Context Compressor
# ---------------------------------------------------------------------------

class CrossAgentCompressor:
    """
    Compresses accumulated state before delegating to a remote A2A agent.

    When the ``CentralSupervisor`` delegates a task to a remote team
    supervisor via ``A2AClientNode``, the accumulated context (plan +
    conversation summary + prior results from other teams) can be large.
    This compressor reduces that context to prevent token budget
    exhaustion in the downstream agent's LLM calls.
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or get_compression_config()
        self._engine = _get_engine()

    def compress_delegation_context(
        self,
        plan: Optional[Dict[str, Any]] = None,
        conversation_summary: str = "",
        prior_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compress the context that will be sent to a remote agent.

        Args:
            plan:                 The execution plan dict.
            conversation_summary: Summary of conversation history.
            prior_results:        Results from previously invoked teams.

        Returns:
            Dict with compressed versions of each context component.
        """
        compressed: Dict[str, Any] = {}

        # Compress plan: keep current goal full, summarize completed goals
        if plan:
            compressed["plan"] = self._compress_plan(plan)

        # Compress conversation summary
        if conversation_summary:
            compressed["conversation_summary"] = self._engine.compress(
                conversation_summary,
                rate=self.config.CROSS_AGENT_RATE,
                target_token=self.config.CROSS_AGENT_MAX_TOKENS,
            )

        # Compress prior results
        if prior_results:
            compressed["prior_results"] = self._compress_prior_results(
                prior_results,
            )

        return compressed

    def _compress_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress a plan by summarizing completed goals.

        Strategy: Replace completed goal descriptions with one-line
        summaries; only the current goal retains full detail.
        """
        compressed_plan = dict(plan)
        goals = compressed_plan.get("goals", [])
        current_idx = compressed_plan.get("current_goal_index", 0)

        compressed_goals = []
        for i, goal in enumerate(goals):
            if i < current_idx:
                # Completed goal — compress description
                description = goal.get("description", "")
                compressed_desc = self._engine.compress(
                    description,
                    rate=self.config.PLAN_COMPLETED_RATE,
                )
                compressed_goal = dict(goal)
                compressed_goal["description"] = compressed_desc
                compressed_goal["_compressed"] = True
                compressed_goals.append(compressed_goal)
            else:
                # Current or future goal — keep full
                compressed_goals.append(goal)

        compressed_plan["goals"] = compressed_goals
        return compressed_plan

    def _compress_prior_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compress prior team results to essential information."""
        compressed = {}
        for key, value in results.items():
            if isinstance(value, str) and count_tokens(value) > 50:
                compressed[key] = self._engine.compress(
                    value,
                    rate=self.config.CROSS_AGENT_RATE,
                )
            else:
                compressed[key] = value
        return compressed


# ---------------------------------------------------------------------------
# Singleton Factories
# ---------------------------------------------------------------------------
@lru_cache
def get_conversation_compressor() -> ConversationHistoryCompressor:
    """Get or create the singleton ``ConversationHistoryCompressor``."""
    _conversation_compressor = ConversationHistoryCompressor()
    return _conversation_compressor

@lru_cache
def get_semantic_compressor() -> SemanticContextCompressor:
    """Get or create the singleton ``SemanticContextCompressor``."""
    _semantic_compressor = SemanticContextCompressor()
    return _semantic_compressor

@lru_cache
def get_cross_agent_compressor() -> CrossAgentCompressor:
    """Get or create the singleton ``CrossAgentCompressor``."""
    _cross_agent_compressor = CrossAgentCompressor()
    return _cross_agent_compressor
