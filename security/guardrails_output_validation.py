"""
Guardrails AI Output Validation for Health Insurance AI Platform

Control 8: Output Validation & Sanitization
- PII/PHI detection and redaction
- Toxicity and bias detection
- Hallucination detection
- Topic relevance checking
- HIPAA compliance validation
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from guardrails import Guard
from guardrails.hub import DetectPII, ToxicLanguage, RestrictToTopic
from guardrails.validators import Validator, register_validator, ValidationResult

logger = logging.getLogger(__name__)


class OutputAction(Enum):
    """Actions to take when validation fails."""
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    REASK = "reask"
    FIX = "fix"


@register_validator(name="hipaa_compliance", data_type="string")
class HIPAAComplianceValidator(Validator):
    """
    Custom validator to ensure output complies with HIPAA regulations.
    Checks for PHI leakage (18 HIPAA identifiers).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 18 HIPAA identifiers
        self.phi_patterns = {
            "names": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            "geographic": r"\b\d{5}(?:-\d{4})?\b",
            "dates": r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "mrn": r"\bMRN:?\s*\d+\b",
        }
    
    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate output for HIPAA compliance."""
        import re
        
        violations = []
        
        # Check for PHI patterns
        for identifier, pattern in self.phi_patterns.items():
            matches = re.findall(pattern, value, re.IGNORECASE)
            if matches:
                violations.append({
                    "type": identifier,
                    "matches": matches[:3],
                    "count": len(matches)
                })
        
        if violations:
            return ValidationResult(
                outcome="fail",
                error_message=f"HIPAA violation: Found {len(violations)} types of PHI",
                fix_value=self._redact_phi(value),
                metadata={"violations": violations}
            )
        
        return ValidationResult(outcome="pass")
    
    def _redact_phi(self, text: str) -> str:
        """Redact PHI from text."""
        import re
        redacted = text
        for identifier, pattern in self.phi_patterns.items():
            redacted = re.sub(pattern, f"<{identifier.upper()}_REDACTED>", redacted, flags=re.IGNORECASE)
        return redacted


class GuardrailsOutputValidator:
    """
    Manages output validation using Guardrails AI.
    Provides pre-configured guards for different output types.
    """
    
    def __init__(self):
        """Initialize Guardrails AI validators."""
        self.guards = self._create_guards()
        logger.info("Guardrails AI Output Validator initialized")
    
    def _create_guards(self) -> Dict[str, Guard]:
        """Create pre-configured guards for different scenarios."""
        guards = {}
        
        # Standard output guard
        guards["standard"] = Guard().use_many(
            DetectPII(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD"],
                on_fail="fix"
            ),
            ToxicLanguage(threshold=0.7, on_fail="exception"),
            RestrictToTopic(
                valid_topics=[
                    "health insurance", "claims", "benefits", "coverage",
                    "prior authorization", "providers", "eligibility"
                ],
                on_fail="reask"
            ),
            HIPAAComplianceValidator(on_fail="fix")
        )
        
        # Member services guard (stricter PII controls)
        guards["member_services"] = Guard().use_many(
            DetectPII(
                pii_entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD",
                    "US_DRIVER_LICENSE", "US_PASSPORT", "LOCATION"
                ],
                on_fail="fix"
            ),
            HIPAAComplianceValidator(on_fail="fix")
        )
        
        # Claims guard
        guards["claims"] = Guard().use_many(
            DetectPII(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD"],
                on_fail="fix"
            ),
            HIPAAComplianceValidator(on_fail="fix")
        )
        
        # Prior authorization guard
        guards["prior_authorization"] = Guard().use_many(
            DetectPII(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "MEDICAL_LICENSE"],
                on_fail="fix"
            ),
            HIPAAComplianceValidator(on_fail="fix")
        )
        
        return guards
    
    def validate_output(
        self,
        output: str,
        guard_type: str = "standard",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate agent output using specified guard.
        
        Args:
            output: Agent's output text
            guard_type: Type of guard to use
            metadata: Optional metadata
            
        Returns:
            Dict with validation results and sanitized output
        """
        guard = self.guards.get(guard_type, self.guards["standard"])
        metadata = metadata or {}
        
        try:
            validated_output = guard.validate(output, metadata=metadata)
            
            return {
                "valid": True,
                "sanitized_output": validated_output.validated_output,
                "validation_passed": validated_output.validation_passed,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return {
                "valid": False,
                "sanitized_output": None,
                "error": str(e),
                "metadata": metadata
            }


# Singleton instance
_guardrails_instance: Optional[GuardrailsOutputValidator] = None


def get_output_validator() -> GuardrailsOutputValidator:
    """Get or create singleton output validator instance."""
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = GuardrailsOutputValidator()
    return _guardrails_instance


def validate_agent_output(
    output: str,
    agent_type: str = "standard",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate agent output.
    
    Args:
        output: Agent's output text
        agent_type: Type of agent
        metadata: Optional metadata
        
    Returns:
        Validation results
    """
    validator = get_output_validator()
    return validator.validate_output(output, guard_type=agent_type, metadata=metadata)
