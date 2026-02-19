"""
Unit Tests for Control 9: Output Validation & DLP
===================================================

Tests the Guardrails AI + Presidio integration for output validation, the
HIPAAComplianceValidator (Presidio-backed), and the DLP Scanner post-validation
audit layer.

Modules under test:
- security/presidio_healthcare_recognizers.py
- security/guardrails_output_validation.py
- security/dlp_scanner.py
"""

from unittest.mock import MagicMock, patch
import importlib
import importlib.util

from security.dlp_scanner import DLPScanner, _classify_sensitivity, EntityFound
from security.guardrails_output_validation import _classify_failure, validate_agent_output, HIPAAComplianceValidator, GuardrailsOutputValidator
from security.presidio_healthcare_recognizers import get_healthcare_recognizers, get_healthcare_analyzer
from security.presidio_healthcare_recognizers import (
            ALL_ENTITIES, STANDARD_PII_ENTITIES, HEALTHCARE_PHI_ENTITIES,
        )

# ============================================================================
# Tests for presidio_healthcare_recognizers.py
# ============================================================================

class TestPresidioHealthcareRecognizers:
    """Tests for the shared Presidio healthcare recognizer module."""

    def test_get_healthcare_recognizers_returns_list(self):
        """get_healthcare_recognizers() returns a non-empty list."""
        
        recognizers = get_healthcare_recognizers()
        assert isinstance(recognizers, list)
        assert len(recognizers) >= 4  # at least MEMBER_ID, POLICY_NUMBER, CLAIM_NUMBER, PA_NUMBER

    def test_recognizer_entity_types(self):
        """Each recognizer covers the expected entity type."""
        
        entity_types = set()
        for r in get_healthcare_recognizers():
            entity_types.update(r.supported_entities)
        assert "MEMBER_ID" in entity_types
        assert "POLICY_NUMBER" in entity_types
        assert "CLAIM_NUMBER" in entity_types
        assert "PA_NUMBER" in entity_types

    def test_all_entities_includes_standard_and_healthcare(self):
        """ALL_ENTITIES contains both standard PII and healthcare PHI."""
        
        for ent in STANDARD_PII_ENTITIES:
            assert ent in ALL_ENTITIES
        for ent in HEALTHCARE_PHI_ENTITIES:
            assert ent in ALL_ENTITIES

    def test_get_healthcare_analyzer_returns_engine(self):
        """get_healthcare_analyzer() returns an AnalyzerEngine instance."""
        
        analyzer = get_healthcare_analyzer()
        assert analyzer is not None
        # Should have an analyze method
        assert hasattr(analyzer, "analyze")

    def test_get_healthcare_analyzer_is_singleton(self):
        """Repeated calls return the same AnalyzerEngine instance."""
        
        a1 = get_healthcare_analyzer()
        a2 = get_healthcare_analyzer()
        assert a1 is a2

    def test_analyzer_detects_member_id(self):
        """The shared analyzer detects MEMBER_ID patterns."""
        
        analyzer = get_healthcare_analyzer()
        results = analyzer.analyze(
            text="Member ID is AB123456 for this account.",
            entities=["MEMBER_ID"],
            language="en",
        )
        member_hits = [r for r in results if r.entity_type == "MEMBER_ID"]
        assert len(member_hits) >= 1

    def test_analyzer_detects_policy_number(self):
        """The shared analyzer detects POLICY_NUMBER patterns."""
        
        analyzer = get_healthcare_analyzer()
        results = analyzer.analyze(
            text="Your policy is POL-12345678.",
            entities=["POLICY_NUMBER"],
            language="en",
        )
        policy_hits = [r for r in results if r.entity_type == "POLICY_NUMBER"]
        assert len(policy_hits) >= 1

    def test_analyzer_detects_claim_number(self):
        """The shared analyzer detects CLAIM_NUMBER patterns."""
        
        analyzer = get_healthcare_analyzer()
        results = analyzer.analyze(
            text="Claim CLM-12345 has been processed.",
            entities=["CLAIM_NUMBER"],
            language="en",
        )
        claim_hits = [r for r in results if r.entity_type == "CLAIM_NUMBER"]
        assert len(claim_hits) >= 1

    def test_analyzer_detects_pa_number(self):
        """The shared analyzer detects PA_NUMBER patterns."""
        
        analyzer = get_healthcare_analyzer()
        results = analyzer.analyze(
            text="Prior auth PA-1234-5678 was approved.",
            entities=["PA_NUMBER"],
            language="en",
        )
        pa_hits = [r for r in results if r.entity_type == "PA_NUMBER"]
        assert len(pa_hits) >= 1


# ============================================================================
# Tests for guardrails_output_validation.py
# ============================================================================

class TestHIPAAComplianceValidator:
    """Tests for the Presidio-backed HIPAAComplianceValidator."""

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    def test_validate_passes_clean_text(self, mock_get_analyzer):
        """Clean text with no PHI should pass validation."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        mock_get_analyzer.return_value = mock_analyzer

        
        validator = HIPAAComplianceValidator()
        validator._analyzer = mock_analyzer

        result = validator.validate("Your claim has been approved.", metadata={})
        assert result.outcome == "pass"

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    def test_validate_fails_on_phi(self, mock_get_analyzer):
        """Text containing PHI should fail validation with a fix value."""
        # Create mock Presidio result
        mock_result = MagicMock()
        mock_result.entity_type = "US_SSN"
        mock_result.score = 0.95
        mock_result.start = 15
        mock_result.end = 26

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [mock_result]
        mock_get_analyzer.return_value = mock_analyzer

        
        validator = HIPAAComplianceValidator()
        validator._analyzer = mock_analyzer

        text = "The SSN is 123-45-6789 for this member."
        result = validator.validate(text, metadata={})
        assert result.outcome == "fail"
        assert "HIPAA violation" in result.metadata["error_message"]
        assert result.metadata["fix_value"] is not None
        assert "<US_SSN>" in result.metadata["fix_value"]
        assert result.validated_chunk is not None

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    def test_validate_ignores_low_confidence(self, mock_get_analyzer):
        """Detections below score 0.5 should be ignored."""
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.score = 0.3  # below threshold
        mock_result.start = 0
        mock_result.end = 4

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [mock_result]
        mock_get_analyzer.return_value = mock_analyzer

        
        validator = HIPAAComplianceValidator()
        validator._analyzer = mock_analyzer

        result = validator.validate("John has a claim.", metadata={})
        assert result.outcome == "pass"

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    def test_validate_handles_analyzer_error_gracefully(self, mock_get_analyzer):
        """If Presidio errors, the validator should pass (fail-open)."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("Model not loaded")
        mock_get_analyzer.return_value = mock_analyzer

        
        validator = HIPAAComplianceValidator()
        validator._analyzer = mock_analyzer

        result = validator.validate("Some text.", metadata={})
        assert result.outcome == "pass"


class TestGuardrailsOutputValidator:
    """Tests for the GuardrailsOutputValidator class."""

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    @patch("security.guardrails_output_validation.Guard")
    def test_init_creates_four_guard_profiles(self, mock_guard_cls, mock_analyzer):
        """Initialisation should create standard, member_services, claims, and prior_authorization guards."""
        mock_guard_instance = MagicMock()
        mock_guard_instance.use_many.return_value = mock_guard_instance
        mock_guard_cls.return_value = mock_guard_instance
        mock_analyzer.return_value = MagicMock()

        
        validator = GuardrailsOutputValidator()

        assert "standard" in validator.guards
        assert "member_services" in validator.guards
        assert "claims" in validator.guards
        assert "prior_authorization" in validator.guards

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    @patch("security.guardrails_output_validation.Guard")
    def test_validate_output_success(self, mock_guard_cls, mock_analyzer):
        """Successful validation returns valid=True with sanitized output."""
        mock_validated = MagicMock()
        mock_validated.validated_output = "Clean response"
        mock_validated.validation_passed = True

        mock_guard = MagicMock()
        mock_guard.use_many.return_value = mock_guard
        mock_guard.validate.return_value = mock_validated
        mock_guard_cls.return_value = mock_guard
        mock_analyzer.return_value = MagicMock()

        
        validator = GuardrailsOutputValidator()

        result = validator.validate_output("Clean response", guard_type="standard")
        assert result["valid"] is True
        assert result["sanitized_output"] == "Clean response"
        assert result["guard_type"] == "standard"

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    @patch("security.guardrails_output_validation.Guard")
    def test_validate_output_failure(self, mock_guard_cls, mock_analyzer):
        """Failed validation returns valid=False with error details."""
        mock_guard = MagicMock()
        mock_guard.use_many.return_value = mock_guard
        mock_guard.validate.side_effect = Exception("PII detected in output")
        mock_guard_cls.return_value = mock_guard
        mock_analyzer.return_value = MagicMock()

        
        validator = GuardrailsOutputValidator()

        result = validator.validate_output("SSN: 123-45-6789", guard_type="standard")
        assert result["valid"] is False
        assert result["reason"] == "pii_detected"

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    @patch("security.guardrails_output_validation.Guard")
    def test_validate_output_falls_back_to_standard_guard(self, mock_guard_cls, mock_analyzer):
        """Unknown guard_type falls back to the standard guard."""
        mock_validated = MagicMock()
        mock_validated.validated_output = "Fallback response"
        mock_validated.validation_passed = True

        mock_guard = MagicMock()
        mock_guard.use_many.return_value = mock_guard
        mock_guard.validate.return_value = mock_validated
        mock_guard_cls.return_value = mock_guard
        mock_analyzer.return_value = MagicMock()

        
        validator = GuardrailsOutputValidator()

        result = validator.validate_output("Test", guard_type="nonexistent_guard")
        assert result["valid"] is True

    @patch("security.guardrails_output_validation.get_healthcare_analyzer")
    @patch("security.guardrails_output_validation.Guard")
    def test_validate_output_toxicity_failure(self, mock_guard_cls, mock_analyzer):
        """Toxicity detection failure is classified correctly."""
        mock_guard = MagicMock()
        mock_guard.use_many.return_value = mock_guard
        mock_guard.validate.side_effect = Exception("Toxic language detected")
        mock_guard_cls.return_value = mock_guard
        mock_analyzer.return_value = MagicMock()

        
        validator = GuardrailsOutputValidator()

        result = validator.validate_output("Bad content", guard_type="standard")
        assert result["valid"] is False
        assert result["reason"] == "toxicity_detected"


class TestClassifyFailure:
    """Tests for the _classify_failure helper."""

    def test_pii_classification(self):
        assert _classify_failure(Exception("PII detected")) == "pii_detected"

    def test_phi_classification(self):
        assert _classify_failure(Exception("PHI leakage")) == "pii_detected"

    def test_toxic_classification(self):
        assert _classify_failure(Exception("Toxic language")) == "toxicity_detected"

    def test_topic_classification(self):
        assert _classify_failure(Exception("Off topic response")) == "off_topic"

    def test_hipaa_classification(self):
        assert _classify_failure(Exception("HIPAA violation found")) == "hipaa_violation"

    def test_unknown_classification(self):
        assert _classify_failure(Exception("Something else")) == "validation_error"


class TestSingletonAPI:
    """Tests for the singleton and convenience API."""

    @patch("security.guardrails_output_validation.GuardrailsOutputValidator")
    def test_get_output_validator_creates_singleton(self, mock_cls):
        """get_output_validator() creates and caches a singleton."""
        import security.guardrails_output_validation as mod
        mod._guardrails_instance = None  # reset
        mock_cls.return_value = MagicMock()

        v1 = mod.get_output_validator()
        v2 = mod.get_output_validator()
        assert v1 is v2
        mock_cls.assert_called_once()

        mod._guardrails_instance = None  # cleanup

    @patch("security.guardrails_output_validation.get_output_validator")
    def test_validate_agent_output_convenience(self, mock_get):
        """validate_agent_output() delegates to the singleton."""
        mock_validator = MagicMock()
        mock_validator.validate_output.return_value = {"valid": True, "sanitized_output": "ok"}
        mock_get.return_value = mock_validator

        result = validate_agent_output("test", agent_type="claims")
        mock_validator.validate_output.assert_called_once_with(
            "test", guard_type="claims", metadata=None
        )


# ============================================================================
# Tests for dlp_scanner.py
# ============================================================================

class TestDLPScanner:
    """Tests for the DLP Scanner post-validation audit layer."""

    @patch("security.dlp_scanner.get_healthcare_analyzer")
    def test_scan_clean_output(self, mock_get_analyzer):
        """Clean output returns safe=True with PUBLIC sensitivity."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        mock_get_analyzer.return_value = mock_analyzer

        scanner = DLPScanner(enable_clickhouse=False)
        scanner._analyzer = mock_analyzer

        result = scanner.scan_output("Your claim has been approved.", agent_id="test")
        assert result.safe is True
        assert result.sensitivity.value == "public"
        assert len(result.entities) == 0

    @patch("security.dlp_scanner.get_healthcare_analyzer")
    def test_scan_detects_restricted_content(self, mock_get_analyzer):
        """Output with SSN is classified as RESTRICTED and not safe."""
        mock_result = MagicMock()
        mock_result.entity_type = "US_SSN"
        mock_result.score = 0.95
        mock_result.start = 8
        mock_result.end = 19

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [mock_result]
        mock_get_analyzer.return_value = mock_analyzer

        scanner = DLPScanner(enable_clickhouse=False)
        scanner._analyzer = mock_analyzer

        result = scanner.scan_output(
            "The SSN 123-45-6789 was found.", agent_id="test"
        )
        assert result.safe is False
        assert result.sensitivity.value == "restricted"
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "US_SSN"
        assert result.redacted_text is not None

    @patch("security.dlp_scanner.get_healthcare_analyzer")
    def test_scan_confidential_content(self, mock_get_analyzer):
        """Output with email is classified as CONFIDENTIAL but still safe."""
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.score = 0.9
        mock_result.start = 10
        mock_result.end = 30

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [mock_result]
        mock_get_analyzer.return_value = mock_analyzer

        scanner = DLPScanner(enable_clickhouse=False)
        scanner._analyzer = mock_analyzer

        result = scanner.scan_output(
            "Email is user@example.com here.", agent_id="test"
        )
        assert result.safe is True  # CONFIDENTIAL is safe, only RESTRICTED is not
        assert result.sensitivity.value == "confidential"

    @patch("security.dlp_scanner.get_healthcare_analyzer")
    def test_scan_generates_warnings_for_restricted(self, mock_get_analyzer):
        """RESTRICTED content generates DLP warnings."""
        mock_result = MagicMock()
        mock_result.entity_type = "CREDIT_CARD"
        mock_result.score = 0.95
        mock_result.start = 0
        mock_result.end = 19

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [mock_result]
        mock_get_analyzer.return_value = mock_analyzer

        scanner = DLPScanner(enable_clickhouse=False)
        scanner._analyzer = mock_analyzer

        result = scanner.scan_output("4111111111111111 is a card.", agent_id="test")
        assert len(result.warnings) > 0
        assert "high-risk" in result.warnings[0].lower()


class TestDLPSensitivityClassification:
    """Tests for the _classify_sensitivity helper."""

    def test_no_entities_is_public(self):
        assert _classify_sensitivity([]).value == "public"

    def test_high_risk_entity_is_restricted(self):
        entities = [EntityFound("US_SSN", "123-45-6789", 0.95, 0, 11)]
        assert _classify_sensitivity(entities).value == "restricted"

    def test_medium_risk_entity_is_confidential(self):
        entities = [EntityFound("EMAIL_ADDRESS", "a@b.com", 0.9, 0, 7)]
        assert _classify_sensitivity(entities).value == "confidential"

    def test_member_id_is_restricted(self):
        entities = [EntityFound("MEMBER_ID", "AB123456", 0.85, 0, 8)]
        assert _classify_sensitivity(entities).value == "restricted"

    def test_claim_number_is_confidential(self):
        entities = [EntityFound("CLAIM_NUMBER", "CLM-12345", 0.85, 0, 9)]
        assert _classify_sensitivity(entities).value == "confidential"

    def test_unknown_entity_is_internal(self):
        entities = [EntityFound("UNKNOWN_TYPE", "xyz", 0.5, 0, 3)]
        assert _classify_sensitivity(entities).value == "internal"


class TestDLPScannerSingleton:
    """Tests for the DLP scanner singleton."""

    @patch("security.dlp_scanner.DLPScanner")
    def test_get_dlp_scanner_creates_singleton(self, mock_cls):
        """get_dlp_scanner() creates and caches a singleton."""
        import security.dlp_scanner as mod
        mod._dlp_scanner = None  # reset
        mock_cls.return_value = MagicMock()

        s1 = mod.get_dlp_scanner(enable_clickhouse=False)
        s2 = mod.get_dlp_scanner()
        assert s1 is s2
        mock_cls.assert_called_once()

        mod._dlp_scanner = None  # cleanup


# ============================================================================
# Integration-style tests: request_processor DLP integration
# ============================================================================

class TestRequestProcessorDLPIntegration:
    """Verify that request_processor.py imports and calls the DLP scanner."""

    def test_dlp_scanner_import_exists(self):
        """request_processor.py imports get_dlp_scanner."""
        
        spec = importlib.util.find_spec("agents.request_processor")
        if spec and spec.origin:
            with open(spec.origin) as f:
                source = f.read()
            assert "from security.dlp_scanner import get_dlp_scanner" in source

    def test_dlp_scanner_called_in_pipeline(self):
        """request_processor.py calls dlp_scanner.scan_output."""
        spec = importlib.util.find_spec("agents.request_processor")
        if spec and spec.origin:
            with open(spec.origin) as f:
                source = f.read()
            assert "dlp_scanner.scan_output" in source
            assert "CONTROL 9b: DLP POST-VALIDATION AUDIT" in source
