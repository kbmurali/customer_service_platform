"""
Shared Presidio Healthcare Recognizers
=======================================
Provides a pre-configured ``AnalyzerEngine`` with custom healthcare-domain
recognizers for use across the platform:

* **Control 4** (Memory & Context Security) — ``presidio_memory_security.py``
* **Control 9** (Output Validation) — ``guardrails_output_validation.py``
* **DLP Scanner** — ``dlp_scanner.py``

By centralising recognizer definitions here, every module that needs
PII/PHI detection shares the same entity catalogue and detection quality.
"""

import logging
from typing import List

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

logger = logging.getLogger(__name__)

# ============================================================================
# Custom Healthcare Recognizers
# ============================================================================

_HEALTHCARE_RECOGNIZERS: List[PatternRecognizer] = []


def _build_recognizers() -> List[PatternRecognizer]:
    """Build the list of healthcare-specific Presidio recognizers (cached)."""
    global _HEALTHCARE_RECOGNIZERS
    if _HEALTHCARE_RECOGNIZERS:
        return _HEALTHCARE_RECOGNIZERS

    _HEALTHCARE_RECOGNIZERS = [
        PatternRecognizer(
            supported_entity="MEMBER_ID",
            patterns=[Pattern(name="member_id_pattern", regex=r"\b[A-Z]{1,2}\d{6,8}\b", score=0.85)],
            name="MemberIDRecognizer",
            supported_language="en",
        ),
        PatternRecognizer(
            supported_entity="POLICY_NUMBER",
            patterns=[Pattern(name="policy_pattern", regex=r"\bPOL-\d{8,10}\b", score=0.85)],
            name="PolicyNumberRecognizer",
            supported_language="en",
        ),
        PatternRecognizer(
            supported_entity="CLAIM_NUMBER",
            patterns=[Pattern(name="claim_pattern", regex=r"\bCLM-\d{5,8}\b", score=0.85)],
            name="ClaimNumberRecognizer",
            supported_language="en",
        ),
        PatternRecognizer(
            supported_entity="PA_NUMBER",
            patterns=[Pattern(name="pa_pattern", regex=r"\bPA-\d{4}-\d{4,6}\b", score=0.85)],
            name="PANumberRecognizer",
            supported_language="en",
        ),
        PatternRecognizer(
            supported_entity="NPI_NUMBER",
            patterns=[Pattern(name="npi_pattern", regex=r"\b\d{10}\b", score=0.40)],
            name="NPINumberRecognizer",
            supported_language="en",
            context=["npi", "provider", "national provider"],
        ),
        PatternRecognizer(
            supported_entity="ICD_CODE",
            patterns=[Pattern(name="icd10_pattern", regex=r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b", score=0.45)],
            name="ICD10Recognizer",
            supported_language="en",
            context=["diagnosis", "icd", "code", "dx"],
        ),
    ]
    return _HEALTHCARE_RECOGNIZERS


def get_healthcare_recognizers() -> List[PatternRecognizer]:
    """Return the list of healthcare-specific Presidio recognizers."""
    return _build_recognizers()


# ============================================================================
# Full entity list (standard + healthcare)
# ============================================================================

#: Standard PII entities detected by Presidio's built-in recognizers.
STANDARD_PII_ENTITIES: List[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "LOCATION",
    "DATE_TIME",
    "MEDICAL_LICENSE",
    "US_BANK_NUMBER",
    "US_ITIN",
    "IP_ADDRESS",
]

#: Healthcare-specific entities detected by custom recognizers above.
HEALTHCARE_PHI_ENTITIES: List[str] = [
    "MEMBER_ID",
    "POLICY_NUMBER",
    "CLAIM_NUMBER",
    "PA_NUMBER",
    "NPI_NUMBER",
    "ICD_CODE",
]

#: Combined list of all entities.
ALL_ENTITIES: List[str] = STANDARD_PII_ENTITIES + HEALTHCARE_PHI_ENTITIES


# ============================================================================
# Shared AnalyzerEngine factory
# ============================================================================

_shared_analyzer: AnalyzerEngine | None = None


def get_healthcare_analyzer() -> AnalyzerEngine:
    """
    Return a shared ``AnalyzerEngine`` instance pre-loaded with all
    healthcare-specific recognizers.

    The engine is created once and reused across the process.
    """
    global _shared_analyzer
    if _shared_analyzer is not None:
        return _shared_analyzer

    _shared_analyzer = AnalyzerEngine()
    for recognizer in get_healthcare_recognizers():
        _shared_analyzer.registry.add_recognizer(recognizer)
        logger.info("Registered healthcare recognizer: %s", recognizer.name)

    logger.info(
        "Shared Presidio AnalyzerEngine initialised with %d healthcare recognizers",
        len(get_healthcare_recognizers()),
    )
    return _shared_analyzer
