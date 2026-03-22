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
import re
from typing import Dict, Any, Optional, List

from dataclasses import dataclass as _dataclass, field as _field

try:
    from guardrails import Guard
    from guardrails.validators import Validator, register_validator, ValidationResult
    # PassResult / FailResult are the correct return types for custom validators
    # in Guardrails >= 0.4.x.  Import with fallback for older versions.
    try:
        from guardrails.validator_base import PassResult, FailResult
    except ImportError:
        try:
            from guardrails.validators import PassResult, FailResult
        except ImportError:
            # Older Guardrails (< 0.4) — validators return ValidationResult directly
            PassResult = None  # type: ignore
            FailResult = None  # type: ignore
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    Guard = None  # type: ignore
    Validator = None  # type: ignore
    register_validator = None  # type: ignore
    PassResult = None  # type: ignore
    FailResult = None  # type: ignore
    logging.getLogger(__name__).info("guardrails-ai not available — output validation will use fallback mode")

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
    logging.getLogger(__name__).info("Guardrails hub validators not installed — using Presidio-only validation")

from security.presidio_healthcare_recognizers import (
    get_healthcare_analyzer,
    get_healthcare_recognizers,
    ALL_ENTITIES,
    STANDARD_PII_ENTITIES,
    HEALTHCARE_PHI_ENTITIES,
)

from security.presidio_memory_security import _filter_uuid_overlaps, _UUID_RE


def _restore_uuids(original: str, sanitized: str) -> str:
    """
    Restore UUIDs in sanitized text that were corrupted by Presidio validators.

    The Guardrails hub ``DetectPII`` validator is a black box that runs its
    own Presidio analysis. We cannot inject UUID filtering into it. Instead,
    after all validation runs, we compare the sanitized output against the
    original and restore any UUIDs that were corrupted.

    A corrupted UUID looks like: ``6fffc059-<US_DRIVER_LICENSE>-4c0f-af6c-...``
    (one or more hex segments replaced with ``<ENTITY_TYPE>`` placeholders).

    Args:
        original:  The text before validation
        sanitized: The text after validation (may have corrupted UUIDs)

    Returns:
        sanitized text with UUIDs restored from the original
    """
    original_uuids = _UUID_RE.findall(original)
    if not original_uuids:
        return sanitized

    for uuid in original_uuids:
        if uuid in sanitized:
            continue  # This UUID survived intact

        # Build a regex that matches this UUID with any segment(s) replaced
        # by <ENTITY_TYPE> placeholders. UUID format: 8-4-4-4-12 hex segments.
        parts = uuid.split("-")
        # Each part could be intact or replaced with <...>
        pattern_parts = []
        for part in parts:
            # Match either the original hex segment or an <ENTITY_TYPE> placeholder
            pattern_parts.append(f"(?:{re.escape(part)}|<[A-Z_]+>)")
        corrupted_pattern = re.compile("-".join(pattern_parts))

        sanitized = corrupted_pattern.sub(uuid, sanitized)

    return sanitized

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

    def validate(self, value: str, metadata: Dict[str, Any]):
        """
        Analyse *value* for HIPAA-regulated PHI using Presidio.

        Returns PassResult / FailResult (Guardrails >= 0.4.x) when those
        classes are available, falling back to ValidationResult for older
        versions.  Using the wrong return type is what causes the
        'Unexpected result type' error inside Guard internals.
        """
        try:
            results = self.analyzer.analyze(
                text=value,
                entities=ALL_ENTITIES,
                language="en",
            )
            # Filter out entities that overlap with UUID positions — prevents
            # UUID segments like "c149" being detected as US_DRIVER_LICENSE
            results = _filter_uuid_overlaps(results, value)

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
                        "text_snippet": value[r.start:r.end][:20],
                    })

                error_msg = (
                    f"HIPAA violation: Found {len(violations)} PHI entities "
                    f"({', '.join(set(v['entity_type'] for v in violations))})"
                )
                logger.warning(
                    "HIPAA compliance check found %d PHI entities", len(violations)
                )

                redacted = self._redact_with_presidio(value, significant)

                # Return FailResult (>= 0.4.x) or ValidationResult (< 0.4.x)
                if FailResult is not None:
                    return FailResult(
                        error_message=error_msg,
                        fix_value=redacted,
                        metadata={"violations": violations},
                    )
                return ValidationResult(
                    outcome="fail",
                    metadata={"error_message": error_msg, "fix_value": redacted,
                              "violations": violations},
                    validated_chunk=redacted,
                )

            # No PHI detected — pass
            if PassResult is not None:
                return PassResult()
            return ValidationResult(outcome="pass")

        except Exception as exc:
            logger.error("HIPAA compliance check error: %s", exc, exc_info=True)
            # Fail-open — do not block the response if Presidio itself errors.
            if PassResult is not None:
                return PassResult()
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
        logging.getLogger(__name__).warning("Could not register HIPAAComplianceValidator: %s", _reg_exc)


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

    Returns ``None`` when the Guardrails hub is not installed so callers
    can filter it out before passing to ``Guard().use_many()``.
    """
    if not HUB_VALIDATORS_AVAILABLE or DetectPII is None:
        return None
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
        """
        Create pre-configured guards for different agent scenarios.

        When the Guardrails hub is not installed (HUB_VALIDATORS_AVAILABLE=False),
        DetectPII, ToxicLanguage, and RestrictToTopic are all None.
        Each guard is built from only the validators that are actually available,
        falling back to HIPAAComplianceValidator (Presidio-only) at minimum.

        A single DetectPII instance is shared across all guards to avoid
        spawning multiple Presidio AnalyzerEngines (each loads en_core_web_lg).
        Custom entity types that have no registered Presidio recognizer
        (MEMBER_ID, POLICY_NUMBER, CLAIM_NUMBER, PA_NUMBER) are intentionally
        excluded from DetectPII — they are handled exclusively by
        HIPAAComplianceValidator which uses the shared healthcare analyzer
        singleton that includes the NPINumberRecognizer.
        """
        guards: Dict[str, Guard] = {}

        def _build_guard(*validators) -> Guard:
            """Build a Guard from the subset of validators that are not None."""
            available = [v for v in validators if v is not None]
            if not available:
                # Should never happen — HIPAAComplianceValidator has no hub dep
                logger.warning("No validators available — guard will pass all output")
                return Guard()
            if not GUARDRAILS_AVAILABLE or Guard is None:
                # guardrails-ai not installed at all — return a no-op sentinel
                return None  # type: ignore
            return Guard().use_many(*available)

        # ── Shared validators — instantiated once ────────────────────────
        # DetectPII is created once and reused across all four guard profiles
        # to avoid loading en_core_web_lg multiple times. Only entities with
        # registered Presidio recognizers are included here.
        _shared_detect_pii = _create_presidio_detect_pii(
            pii_entities=[
                "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
                "US_DRIVER_LICENSE", "US_PASSPORT", "MEDICAL_LICENSE",
            ],
            on_fail="fix",
        )
        _toxic    = ToxicLanguage(threshold=0.7, on_fail="exception") if HUB_VALIDATORS_AVAILABLE and ToxicLanguage else None
        _restrict = RestrictToTopic(
            valid_topics=[
                "health insurance", "claims", "benefits", "coverage",
                "prior authorization", "providers", "eligibility",
                "deductible", "copay", "coinsurance", "premium",
                "explanation of benefits", "appeal", "grievance",
            ],
            on_fail="reask",
        ) if HUB_VALIDATORS_AVAILABLE and RestrictToTopic else None

        # ── Standard output guard (used by default) ──────────────────────
        guards["standard"] = _build_guard(
            _shared_detect_pii,
            _toxic,
            _restrict,
            HIPAAComplianceValidator(on_fail="fix"),
        )

        # ── Member services guard (stricter PII controls) ────────────────
        guards["member_services"] = _build_guard(
            _shared_detect_pii,
            HIPAAComplianceValidator(on_fail="fix"),
        )

        # ── Claims guard ─────────────────────────────────────────────────
        guards["claims"] = _build_guard(
            _shared_detect_pii,
            HIPAAComplianceValidator(on_fail="fix"),
        )

        # ── Prior authorization guard ────────────────────────────────────
        guards["prior_authorization"] = _build_guard(
            _shared_detect_pii,
            HIPAAComplianceValidator(on_fail="fix"),
        )

        mode = "full (hub + Presidio)" if HUB_VALIDATORS_AVAILABLE else "Presidio-only (hub not installed)"
        logger.info("Guards created in %s mode", mode)
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

        # When guardrails-ai is not installed _build_guard returns None.
        # Fall back to passing the output through unchanged — Presidio scrubbing
        # upstream (Control 5) and DLP post-scan downstream still apply.
        if guard is None:
            logger.warning(
                "guardrails-ai not installed — skipping Guard validation for %s",
                guard_type,
            )
            return {
                "valid": True,
                "sanitized_output": output,
                "validation_passed": True,
                "guard_type": guard_type,
                "metadata": metadata,
            }

        try:
            # Guardrails AI changed its API across versions:
            #   < 0.4.x : guard.validate(value) → ValidationResult
            #   >= 0.4.x : guard.parse(value)   → CallResult / ValidationOutcome
            # We try parse() first (newer API), fall back to validate() (older API),
            # then extract sanitized output defensively regardless of result type
            # so that a version mismatch never silently blocks every response.
            if hasattr(guard, "parse"):
                result = guard.parse(output, metadata=metadata)
            else:
                result = guard.validate(output, metadata=metadata)

            # Extract sanitized output — attribute name varies by version
            sanitized = (
                getattr(result, "validated_output", None)
                or getattr(result, "value", None)
                or getattr(result, "fix_value", None)
                or output        # last resort: pass original through
            )

            # Restore any UUIDs that were corrupted by DetectPII or other
            # Presidio-backed validators that we cannot patch internally.
            if isinstance(sanitized, str):
                sanitized = _restore_uuids(output, sanitized)
            # Extract pass/fail — attribute name varies by version
            passed = bool(
                getattr(result, "validation_passed", None)
                if hasattr(result, "validation_passed")
                else getattr(result, "outcome", "pass") == "pass"
            )

            return {
                "valid":             passed,
                "sanitized_output":  sanitized,
                "validation_passed": passed,
                "guard_type":        guard_type,
                "metadata":          metadata,
            }

        except Exception as exc:
            # Hard failures (e.g. ToxicLanguage on_fail="exception") raise here.
            # Classify and surface as a blocked response.
            logger.error("Output validation failed: %s", exc)
            reason = _classify_failure(exc)
            return {
                "valid":            False,
                "sanitized_output": None,
                "error":            str(exc),
                "reason":           reason,
                "guard_type":       guard_type,
                "metadata":         metadata,
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
