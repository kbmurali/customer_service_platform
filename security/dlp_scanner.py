"""
Data Loss Prevention (DLP) Scanner
====================================

Implements output classification and audit logging:
1. PII/PHI detection using the shared Presidio ``AnalyzerEngine``
   (with healthcare-domain recognizers from ``presidio_healthcare_recognizers``)
2. Content sensitivity classification (PUBLIC → RESTRICTED)
3. Automatic redaction for high-sensitivity outputs
4. ClickHouse audit trail for every scan

Security Control #9: Output Validation & DLP
---------------------------------------------
The DLP scanner runs as a **post-validation audit layer** after Guardrails AI
has already validated/redacted the agent response.  Its job is to:

* Classify the sensitivity level of the final response.
* Create an immutable audit record in ClickHouse.
* Provide defence-in-depth: if Guardrails missed something, the DLP scanner
  logs a warning and can optionally escalate.

Integration point: ``agents/request_processor.py`` (after Guardrails block).
"""

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logging.warning("Presidio not available — DLP scanning will use basic patterns only")

from security.presidio_healthcare_recognizers import (
    get_healthcare_analyzer,
    ALL_ENTITIES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data models
# ============================================================================

class SensitivityLevel(Enum):
    """Content sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class EntityFound:
    """A single detected entity."""
    entity_type: str
    text: str
    score: float
    start: int
    end: int


@dataclass
class DLPResult:
    """Result of a DLP scan."""
    safe: bool
    sensitivity: SensitivityLevel
    entities: List[EntityFound]
    redacted_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Sensitivity classification
# ============================================================================

HIGH_RISK_ENTITIES = frozenset({
    "US_SSN", "SSN", "CREDIT_CARD", "MEDICAL_LICENSE", "US_PASSPORT",
    "MEMBER_ID", "POLICY_NUMBER",
})

MEDIUM_RISK_ENTITIES = frozenset({
    "PHONE_NUMBER", "EMAIL_ADDRESS", "US_DRIVER_LICENSE",
    "CLAIM_NUMBER", "PA_NUMBER", "NPI_NUMBER",
})


def _classify_sensitivity(entities: List[EntityFound]) -> SensitivityLevel:
    """Classify content sensitivity based on detected entities."""
    if not entities:
        return SensitivityLevel.PUBLIC

    for ent in entities:
        if ent.entity_type in HIGH_RISK_ENTITIES:
            return SensitivityLevel.RESTRICTED

    for ent in entities:
        if ent.entity_type in MEDIUM_RISK_ENTITIES:
            return SensitivityLevel.CONFIDENTIAL

    return SensitivityLevel.INTERNAL


# ============================================================================
# Main DLP Scanner
# ============================================================================

class DLPScanner:
    """
    Data Loss Prevention scanner for agent outputs.

    Uses the shared Presidio ``AnalyzerEngine`` (with healthcare recognizers)
    and optionally logs every scan to ClickHouse for compliance auditing.
    """

    def __init__(
        self,
        clickhouse_host: str = "localhost",
        clickhouse_port: int = 9000,
        enable_clickhouse: bool = True,
    ):
        """Initialise DLP scanner."""
        # Presidio engines — shared analyzer, local anonymizer
        self._analyzer = None  # lazy
        self._anonymizer = AnonymizerEngine() if PRESIDIO_AVAILABLE else None

        # ClickHouse audit logging (best-effort)
        self._clickhouse = None
        if enable_clickhouse:
            try:
                from clickhouse_driver import Client as ClickHouseClient
                self._clickhouse = ClickHouseClient(
                    host=clickhouse_host, port=clickhouse_port
                )
                self._init_clickhouse_table()
            except Exception as exc:
                logger.warning("ClickHouse unavailable for DLP audit: %s", exc)

        logger.info("DLP Scanner initialised (Presidio=%s, ClickHouse=%s)",
                     PRESIDIO_AVAILABLE, self._clickhouse is not None)

    # ------------------------------------------------------------------
    # Lazy Presidio access
    # ------------------------------------------------------------------

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = get_healthcare_analyzer()
        return self._analyzer

    # ------------------------------------------------------------------
    # ClickHouse setup
    # ------------------------------------------------------------------

    def _init_clickhouse_table(self):
        """Create the ClickHouse audit table if it does not exist."""
        if self._clickhouse is None:
            return
        try:
            self._clickhouse.execute("""
                CREATE TABLE IF NOT EXISTS dlp_scans (
                    timestamp DateTime,
                    agent_id String,
                    action String,
                    entities_found Array(String),
                    sensitivity String,
                    text_length UInt32,
                    redacted Boolean,
                    scan_duration_ms UInt32
                ) ENGINE = MergeTree()
                ORDER BY timestamp
            """)
        except Exception as exc:
            logger.warning("Could not create dlp_scans table: %s", exc)

    # ------------------------------------------------------------------
    # Core scan API
    # ------------------------------------------------------------------

    def scan_output(
        self,
        text: str,
        agent_id: str = "unknown",
        action: str = "output_scan",
    ) -> DLPResult:
        """
        Scan agent output for PII/PHI, classify sensitivity, and log.

        Args:
            text: Output text to scan.
            agent_id: Identifier of the agent that produced the text.
            action: Descriptive action label for the audit log.

        Returns:
            ``DLPResult`` with scan results.
        """
        start_time = datetime.now(timezone.utc)

        # --- Detect entities ---
        if PRESIDIO_AVAILABLE:
            raw_results = self.analyzer.analyze(
                text=text, entities=ALL_ENTITIES, language="en"
            )
        else:
            raw_results = self._basic_scan(text)

        entities = [
            EntityFound(
                entity_type=r.entity_type,
                text=text[r.start:r.end],
                score=r.score if hasattr(r, "score") else 0.9,
                start=r.start,
                end=r.end,
            )
            for r in raw_results
        ]

        # --- Classify ---
        sensitivity = _classify_sensitivity(entities)

        # --- Redact if high-risk ---
        redacted_text = None
        if sensitivity in (SensitivityLevel.RESTRICTED, SensitivityLevel.CONFIDENTIAL):
            redacted_text = self._redact_text(text, raw_results)

        # --- Warnings ---
        warnings: List[str] = []
        if entities and sensitivity == SensitivityLevel.RESTRICTED:
            warnings.append(
                f"DLP: {len(entities)} high-risk entities detected after Guardrails validation"
            )

        # --- Audit log ---
        scan_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        self._log_scan(agent_id, action, entities, sensitivity, len(text),
                       redacted_text is not None, scan_ms)

        return DLPResult(
            safe=sensitivity != SensitivityLevel.RESTRICTED,
            sensitivity=sensitivity,
            entities=entities,
            redacted_text=redacted_text,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Fallback regex scan
    # ------------------------------------------------------------------

    @staticmethod
    def _basic_scan(text: str) -> List:
        """Basic pattern-based scan when Presidio is unavailable."""
        patterns = {
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "CREDIT_CARD": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE_NUMBER": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        }
        results = []
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                results.append(type("obj", (object,), {
                    "entity_type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.9,
                })())
        return results

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    def _redact_text(self, text: str, raw_results) -> str:
        """Redact sensitive entities from text."""
        if PRESIDIO_AVAILABLE and self._anonymizer and raw_results:
            try:
                return self._anonymizer.anonymize(
                    text=text, analyzer_results=raw_results
                ).text
            except Exception as exc:
                logger.warning("Presidio anonymizer error: %s — using basic redaction", exc)

        # Fallback: manual replacement
        redacted = text
        for r in sorted(raw_results, key=lambda x: x.start, reverse=True):
            redacted = redacted[:r.start] + "[REDACTED]" + redacted[r.end:]
        return redacted

    # ------------------------------------------------------------------
    # ClickHouse audit
    # ------------------------------------------------------------------

    def _log_scan(
        self,
        agent_id: str,
        action: str,
        entities: List[EntityFound],
        sensitivity: SensitivityLevel,
        text_length: int,
        redacted: bool,
        scan_duration_ms: int,
    ):
        """Log scan result to ClickHouse (best-effort)."""
        if self._clickhouse is None:
            return
        try:
            self._clickhouse.execute(
                "INSERT INTO dlp_scans VALUES",
                [{
                    "timestamp": datetime.now(timezone.utc),
                    "agent_id": agent_id,
                    "action": action,
                    "entities_found": [e.entity_type for e in entities],
                    "sensitivity": sensitivity.value,
                    "text_length": text_length,
                    "redacted": redacted,
                    "scan_duration_ms": scan_duration_ms,
                }],
            )
        except Exception as exc:
            logger.error("Failed to log DLP scan to ClickHouse: %s", exc)


# ============================================================================
# Singleton
# ============================================================================

_dlp_scanner: Optional[DLPScanner] = None


def get_dlp_scanner(**kwargs) -> DLPScanner:
    """Get or create the global DLP scanner instance."""
    global _dlp_scanner
    if _dlp_scanner is None:
        _dlp_scanner = DLPScanner(**kwargs)
    return _dlp_scanner
