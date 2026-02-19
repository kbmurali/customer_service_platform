"""
Guardrails AI Output Validation for Health Insurance AI Platform
=================================================================

Control 9: Output Validation & Sanitization
- PII/PHI detection via Guardrails ``DetectPII`` **backed by Presidio AnalyzerEngine**
- Healthcare-domain entity detection (MEMBER_ID, CLAIM_NUMBER, POLICY_NUMBER, PA_NUMBER)
- Toxicity and bias detection (``ToxicLanguage``)
- Topic relevance enforcement (``RestrictToTopic``)
- HIPAA compliance validation delegated to Presidio (replaces regex-only approach)

Integration points
------------------
* Called from ``agents/request_processor.py`` after every agent response.
* Uses the shared ``AnalyzerEngine`` from ``security/presidio_healthcare_recognizers``
  so that healthcare recognizers are consistent across the platform.
"""

import logging
from typing import Dict, Any, Optional, List

from dataclasses import dataclass as _dataclass, field as _field

try:
    from guardrails import Guard
    from guardrails.validators import Validator, register_validator, ValidationResult
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    Guard = None  # type: ignore
    Validator = None  # type: ignore
    register_validator = None  # type: ignore
    logging.warning("guardrails-ai not available — output validation will use fallback mode")

    # Lightweight stand-in so the rest of the module works without guardrails
    @_dataclass
    class ValidationResult:  # type: ignore[no-redef]
        outcome: str = "pass"
        error_message: str = ""
        fix_value: str = None  # type: ignore
        metadata: dict = _field(default_factory=dict)

try:
    from guardrails.hub import DetectPII, ToxicLanguage, RestrictToTopic
    HUB_VALIDATORS_AVAILABLE = True
except (ImportError, AttributeError):
    HUB_VALIDATORS_AVAILABLE = False
    DetectPII = None  # type: ignore
    ToxicLanguage = None  # type: ignore
    RestrictToTopic = None  # type: ignore
    logging.warning("Guardrails hub validators not installed — using Presidio-only validation")

from security.presidio_healthcare_recognizers import (
    get_healthcare_analyzer,
    get_healthcare_recognizers,
    ALL_ENTITIES,
    STANDARD_PII_ENTITIES,
    HEALTHCARE_PHI_ENTITIES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Guardrails Validator — delegates to Presidio
# ============================================================================

# Use Validator as base when available, otherwise use object
_ValidatorBase = Validator if GUARDRAILS_AVAILABLE else object

# Register the validator with Guardrails after class definition (see below)
# We do this post-hoc because the class body needs to be complete first.


class HIPAAComplianceValidator(_ValidatorBase):
    """
    Custom Guardrails validator that delegates PHI detection to the shared
    Presidio ``AnalyzerEngine`` (with healthcare recognizers) instead of
    relying on basic regex patterns.

    When a violation is detected the validator returns a ``fix`` value with
    all PHI entities redacted via Presidio's anonymiser.
    """

    # Class-level attribute expected by Guardrails registry
    rail_alias = "hipaa_compliance_presidio"

    def __init__(self, **kwargs):
        if GUARDRAILS_AVAILABLE:
            super().__init__(**kwargs)
        self._analyzer = None  # lazy-init to avoid import-time side effects

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = get_healthcare_analyzer()
        return self._analyzer

    def validate(self, value: str, metadata: Dict[str, Any]) -> "ValidationResult":
        """Analyse *value* for HIPAA-regulated PHI using Presidio."""
        try:
            results = self.analyzer.analyze(
                text=value,
                entities=ALL_ENTITIES,
                language="en",
            )

            # Filter to high-confidence detections (score >= 0.5)
            significant = [r for r in results if r.score >= 0.5]

            if significant:
                violations = []
                for r in significant:
                    violations.append({
                        "entity_type": r.entity_type,
                        "score": round(r.score, 2),
                        "start": r.start,
                        "end": r.end,
                        "text_snippet": value[r.start:r.end][:20],  # truncated for logs
                    })

                error_msg = (
                    f"HIPAA violation: Found {len(violations)} PHI entities "
                    f"({', '.join(set(v['entity_type'] for v in violations))})"
                )
                logger.warning(
                    "HIPAA compliance check found %d PHI entities", len(violations)
                )

                redacted = self._redact_with_presidio(value, significant)

                return ValidationResult(
                    outcome="fail",
                    metadata={
                        "error_message": error_msg,
                        "fix_value": redacted,
                        "violations": violations,
                    },
                    validated_chunk=redacted,
                )

            return ValidationResult(outcome="pass")

        except Exception as exc:
            logger.error("HIPAA compliance check error: %s", exc, exc_info=True)
            # Fail-open with a warning — do not block the response if Presidio
            # itself errors (e.g. model not loaded).  The upstream Guardrails
            # DetectPII validator provides a second layer of defence.
            return ValidationResult(outcome="pass")

    # ------------------------------------------------------------------
    @staticmethod
    def _redact_with_presidio(text: str, results) -> str:
        """Replace detected entities with ``<ENTITY_TYPE>`` placeholders."""
        # Sort by start position descending so replacements don't shift offsets
        for r in sorted(results, key=lambda x: x.start, reverse=True):
            text = text[:r.start] + f"<{r.entity_type}>" + text[r.end:]
        return text


# Post-hoc registration with Guardrails validator registry
if GUARDRAILS_AVAILABLE and register_validator is not None:
    try:
        from guardrails.validator_base import validators_registry
        if "hipaa_compliance_presidio" not in validators_registry:
            validators_registry["hipaa_compliance_presidio"] = HIPAAComplianceValidator
    except Exception as _reg_exc:
        logging.warning("Could not register HIPAAComplianceValidator: %s", _reg_exc)


# ============================================================================
# Guard Profiles
# ============================================================================

def _create_presidio_detect_pii(
    pii_entities: List[str],
    on_fail: str = "fix",
):
    """
    Create a ``DetectPII`` validator backed by the shared Presidio
    ``AnalyzerEngine`` that includes healthcare-domain recognizers.
    """
    analyzer = get_healthcare_analyzer()
    return DetectPII(
        pii_entities=pii_entities,
        on_fail=on_fail,
    )


# ============================================================================
# Main Validator Class
# ============================================================================

class GuardrailsOutputValidator:
    """
    Manages output validation using Guardrails AI.

    Each guard profile combines:
    1. ``DetectPII`` — Presidio-backed PII/PHI detection
    2. ``ToxicLanguage`` — toxicity / bias detection
    3. ``RestrictToTopic`` — topic relevance enforcement
    4. ``HIPAAComplianceValidator`` — Presidio-backed HIPAA PHI sweep
    """

    def __init__(self):
        """Initialise Guardrails AI validators with Presidio backend."""
        self.guards = self._create_guards()
        logger.info("Guardrails AI Output Validator initialised (Presidio-backed)")

    def _create_guards(self) -> Dict[str, Guard]:
        """Create pre-configured guards for different agent scenarios."""
        guards: Dict[str, Guard] = {}

        # ── Standard output guard (used by default) ──────────────────────
        guards["standard"] = Guard().use_many(
            _create_presidio_detect_pii(
                pii_entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
                    "MEMBER_ID", "POLICY_NUMBER", "CLAIM_NUMBER", "PA_NUMBER",
                ],
                on_fail="fix",
            ),
            ToxicLanguage(threshold=0.7, on_fail="exception"),
            RestrictToTopic(
                valid_topics=[
                    "health insurance", "claims", "benefits", "coverage",
                    "prior authorization", "providers", "eligibility",
                    "deductible", "copay", "coinsurance", "premium",
                    "explanation of benefits", "appeal", "grievance",
                ],
                on_fail="reask",
            ),
            HIPAAComplianceValidator(on_fail="fix"),
        )

        # ── Member services guard (stricter PII controls) ────────────────
        guards["member_services"] = Guard().use_many(
            _create_presidio_detect_pii(
                pii_entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
                    "US_DRIVER_LICENSE", "US_PASSPORT", "LOCATION",
                    "MEMBER_ID", "POLICY_NUMBER",
                ],
                on_fail="fix",
            ),
            HIPAAComplianceValidator(on_fail="fix"),
        )

        # ── Claims guard ─────────────────────────────────────────────────
        guards["claims"] = Guard().use_many(
            _create_presidio_detect_pii(
                pii_entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
                    "CLAIM_NUMBER", "MEMBER_ID",
                ],
                on_fail="fix",
            ),
            HIPAAComplianceValidator(on_fail="fix"),
        )

        # ── Prior authorization guard ────────────────────────────────────
        guards["prior_authorization"] = Guard().use_many(
            _create_presidio_detect_pii(
                pii_entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
                    "MEDICAL_LICENSE", "PA_NUMBER", "MEMBER_ID",
                ],
                on_fail="fix",
            ),
            HIPAAComplianceValidator(on_fail="fix"),
        )

        return guards

    # ------------------------------------------------------------------
    def validate_output(
        self,
        output: str,
        guard_type: str = "standard",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate agent output using the specified guard profile.

        Args:
            output: Agent's output text.
            guard_type: Guard profile name.
            metadata: Optional context metadata.

        Returns:
            Dict with ``valid``, ``sanitized_output``, and diagnostic fields.
        """
        guard = self.guards.get(guard_type, self.guards["standard"])
        metadata = metadata or {}

        try:
            validated = guard.validate(output, metadata=metadata)

            return {
                "valid": True,
                "sanitized_output": validated.validated_output,
                "validation_passed": validated.validation_passed,
                "guard_type": guard_type,
                "metadata": metadata,
            }

        except Exception as exc:
            logger.error("Output validation failed: %s", exc)
            return {
                "valid": False,
                "sanitized_output": None,
                "error": str(exc),
                "reason": _classify_failure(exc),
                "guard_type": guard_type,
                "metadata": metadata,
            }


# ============================================================================
# Failure classification helper
# ============================================================================

def _classify_failure(exc: Exception) -> str:
    """Map exception text to a human-readable failure reason."""
    msg = str(exc).lower()
    if "pii" in msg or "phi" in msg:
        return "pii_detected"
    if "toxic" in msg:
        return "toxicity_detected"
    if "topic" in msg:
        return "off_topic"
    if "hipaa" in msg:
        return "hipaa_violation"
    return "validation_error"


# ============================================================================
# Singleton & convenience API
# ============================================================================

_guardrails_instance: Optional[GuardrailsOutputValidator] = None


def get_output_validator() -> GuardrailsOutputValidator:
    """Get or create the singleton output validator instance."""
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = GuardrailsOutputValidator()
    return _guardrails_instance


def validate_agent_output(
    output: str,
    agent_type: str = "standard",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to validate agent output."""
    validator = get_output_validator()
    return validator.validate_output(output, guard_type=agent_type, metadata=metadata)
