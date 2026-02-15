"""
NeMo Guardrails Integration for Health Insurance AI Platform

Control 1: Prompt Injection Defense — Input Validation Only

Defense architecture — NeMo declarative rails are the PRIMARY guard:
    1. NeMo self check input  — LLM judge: catches jailbreaks, off-topic, PII extraction
    2. NeMo dialog flows      — Colang semantic matching: topic enforcement, jailbreak patterns
    3. Structural safety net  — deterministic backstop for length/encoding anomalies only
                                (NOT content policy — that belongs exclusively in NeMo)
"""

import re
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple

import nest_asyncio
from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

from nemoguardrails import RailsConfig, LLMRails

logger = logging.getLogger(__name__)


# =============================================================================
# Layer 3 (structural safety net only)
# Regex here is NOT for content policy — that belongs in NeMo/Colang.
# These only catch inputs that are structurally malformed or use encoding
# tricks that could bypass the LLM judge.
# =============================================================================
_MAX_INPUT_LENGTH = 1000
_MAX_SPECIAL_CHAR_RATIO = 0.3
_BASE64_PATTERN = re.compile(r'[A-Za-z0-9+/]{50,}={0,2}')


def _sanity_check(message: str) -> Tuple[bool, str]:
    """
    Structural safety net — not content policy.
    Returns (is_suspicious, reason).
    """
    if len(message) > _MAX_INPUT_LENGTH:
        return True, "input exceeds maximum allowed length"
    if _BASE64_PATTERN.search(message):
        return True, "potentially encoded content detected"
    special_char_ratio = sum(
        1 for c in message if not c.isalnum() and not c.isspace()
    ) / max(len(message), 1)
    if special_char_ratio > _MAX_SPECIAL_CHAR_RATIO:
        return True, "unusual character pattern detected"
    return False, ""


# =============================================================================
# Domain whitelist check
# Determines whether a message is plausibly health-insurance-related BEFORE
# hitting the LLM. This is a domain classifier, not content policy — it uses
# a whitelist of health-insurance terms rather than trying to detect bad intent.
# If a message contains none of these signals it is definitively off-topic.
# =============================================================================
_HEALTH_INSURANCE_TERMS = [
    # Plan and coverage
    r"\b(plan|coverage|covered|benefit|benefits|policy|policies)\b",
    r"\b(deductible|copay|co-pay|coinsurance|premium|out.of.pocket)\b",
    r"\b(in.network|out.of.network|network|provider|specialist)\b",
    # Medical services
    r"\b(doctor|physician|hospital|clinic|urgent\s+care|emergency)\b",
    r"\b(prescription|medication|drug|pharmacy|formulary)\b",
    r"\b(claim|claims|reimbursement|prior\s+authorization|referral)\b",
    r"\b(dental|vision|mental\s+health|telehealth|therapy)\b",
    # Member identity
    r"\b(member\s+id|member\s+number|insurance\s+card|group\s+number)\b",
    r"\b(enroll|enrollment|open\s+enrollment|dependent|subscriber)\b",
    r"\b(hmo|ppo|epo|hdhp|hsa|fsa)\b",
    # Sensitive data terms — jailbreak attempts targeting this platform
    # will reference these; they must reach NeMo to be correctly classified
    r"\b(ssn|ssns|social\s+security|password|passwords|pii|credentials)\b",
]

_COMPILED_HEALTH_TERMS = [re.compile(p, re.IGNORECASE) for p in _HEALTH_INSURANCE_TERMS]


def _is_health_insurance_related(message: str) -> bool:
    """
    Returns True if the message contains at least one health-insurance-related term.
    A message with NO matches is definitively off-topic and can be rejected
    without an LLM call.
    """
    return any(p.search(message) for p in _COMPILED_HEALTH_TERMS)


# =============================================================================
# NeMo response interpretation
# =============================================================================

# Phrases NeMo returns when self check input or a Colang flow blocks a request
_NEMO_REFUSAL_PHRASES = [
    "i'm sorry, i can't respond to that",
    "i cannot comply",
    "i can't respond to that",
    "i only assist with health insurance",
]

# Used to classify the block reason for the caller
_JAILBREAK_SIGNALS = [
    "ignore", "previous instructions", "developer mode", "pretend",
    "disregard", "you are now", "act as", "bypass", "override",
    "forget your", "hypothetically", "roleplay", "fictional", "grandmother",
]


def _classify_reason(message: str) -> str:
    """
    Infer block reason for messages blocked by NeMo.
    By the time NeMo runs, off-topic messages are already caught by the
    domain whitelist, so anything NeMo blocks is a jailbreak or policy violation.
    """
    lowered = message.lower()
    if any(signal in lowered for signal in _JAILBREAK_SIGNALS):
        return "jailbreak attempt detected"
    return "policy violation"


# =============================================================================
# Main filter class
# =============================================================================
class NemoGuardrailsFilter:
    """
    Input validation using NeMo Guardrails as the primary enforcement mechanism.

    NeMo handles all content policy via self check input (LLM judge) and
    Colang dialog flows (semantic intent matching). Regex is a structural
    backstop only.

    Usage:
        nemo_filter = get_nemo_filter()

        result = nemo_filter.validate_input(user_message, context={"user_id": "..."})
        if not result["safe"]:
            return result["response"]

        # safe — pass user_message to multi-agent pipeline
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "security/config"
        self.rails = self._initialize_rails()
        logger.info("NeMo Guardrails initialized successfully")

    def _initialize_rails(self) -> LLMRails:
        config = RailsConfig.from_path(self.config_path)
        return LLMRails(config)

    def validate_input(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate user input. NeMo is the primary guard.

        Args:
            user_message: Raw input from the user.
            context: Optional dict (e.g. {"user_id": "..."}) for logging/audit.

        Returns:
            {
                "safe":            bool — False if blocked,
                "reason":          str  — why blocked, or "ok" if safe,
                "response":        str  — refusal message to show user, or None if safe,
                "sanitized_input": str  — original message if safe, or None if blocked,
            }
        """
        user_id = (context or {}).get("user_id", "unknown")

        # ------------------------------------------------------------------
        # Domain whitelist check — fast pre-filter, no LLM cost
        # If the message contains no health-insurance-related terms it is
        # definitively off-topic. No need to spend an LLM call on NeMo.
        # ------------------------------------------------------------------
        if not _is_health_insurance_related(user_message):
            logger.warning(f"[Domain] Off-topic blocked for user={user_id}")
            return {
                "safe": False,
                "reason": "off-topic request: only health insurance queries are supported",
                "response": "I can only assist with health insurance related questions.",
                "sanitized_input": None,
            }

        # ------------------------------------------------------------------
        # Layers 1 + 2: NeMo — self check input (LLM judge) + Colang flows
        # Primary content policy enforcement. Both run inside generate_async:
        #   - self check input: LLM judges raw text against prompts.yml policy
        #   - Colang dialog flows: semantic matching against policies.co examples
        # ------------------------------------------------------------------
        messages = [{"role": "user", "content": user_message}]
        try:
            response = asyncio.run(self.rails.generate_async(messages=messages))
        except RuntimeError:
            # Event loop already running (e.g. pytest-asyncio, Jupyter)
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(
                self.rails.generate_async(messages=messages)
            )

        response_text = (
            response.get("content", "") if isinstance(response, dict)
            else str(response)
        )

        is_blocked = any(phrase in response_text.lower() for phrase in _NEMO_REFUSAL_PHRASES)

        if is_blocked:
            reason = _classify_reason(user_message)
            logger.warning(f"[NeMo] Blocked for user={user_id}: {reason}")
            return {
                "safe": False,
                "reason": reason,
                "response": response_text,
                "sanitized_input": None,
            }

        # ------------------------------------------------------------------
        # Layer 3: Structural safety net
        # Only fires if NeMo passed something structurally anomalous through.
        # Covers encoding tricks and extreme length — not content decisions.
        # ------------------------------------------------------------------
        is_suspicious, sanity_reason = _sanity_check(user_message)
        if is_suspicious:
            logger.warning(f"[Sanity] Blocked for user={user_id}: {sanity_reason}")
            return {
                "safe": False,
                "reason": sanity_reason,
                "response": "I'm sorry, I can't respond to that.",
                "sanitized_input": None,
            }

        logger.info(f"Input passed all checks for user={user_id}")
        return {
            "safe": True,
            "reason": "ok",
            "response": None,
            "sanitized_input": user_message,
        }


# =============================================================================
# Singleton accessor
# =============================================================================
_nemo_instance: Optional[NemoGuardrailsFilter] = None


def get_nemo_filter() -> NemoGuardrailsFilter:
    """Get or create the singleton NeMo filter instance."""
    global _nemo_instance
    if _nemo_instance is None:
        _nemo_instance = NemoGuardrailsFilter()
    return _nemo_instance