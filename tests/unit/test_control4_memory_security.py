"""
Unit Tests for Presidio Memory Security (Control 4: Memory & Context Security)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from security.presidio_memory_security import (
    PresidioMemorySecurity,
    HealthcareRecognizers,
    get_presidio_security
)


class TestHealthcareRecognizers:
    """Test custom healthcare recognizers."""
    
    def test_member_id_recognizer(self):
        """Test member ID recognizer."""
        recognizer = HealthcareRecognizers.get_member_id_recognizer()
        
        assert recognizer is not None
        assert "MEMBER_ID" in recognizer.supported_entities
        assert recognizer.name == "MemberIDRecognizer"
    
    def test_policy_number_recognizer(self):
        """Test policy number recognizer."""
        recognizer = HealthcareRecognizers.get_policy_number_recognizer()
        
        assert recognizer is not None
        assert "POLICY_NUMBER" in recognizer.supported_entities
    
    def test_claim_number_recognizer(self):
        """Test claim number recognizer."""
        recognizer = HealthcareRecognizers.get_claim_number_recognizer()
        
        assert recognizer is not None
        assert "CLAIM_NUMBER" in recognizer.supported_entities
    
    def test_pa_number_recognizer(self):
        """Test PA number recognizer."""
        recognizer = HealthcareRecognizers.get_pa_number_recognizer()
        
        assert recognizer is not None
        assert "PA_NUMBER" in recognizer.supported_entities


class TestPresidioMemorySecurity:
    """Test PresidioMemorySecurity class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = Mock()
        redis_mock.setex = Mock()
        redis_mock.get = Mock(return_value=None)
        redis_mock.keys = Mock(return_value=[])
        redis_mock.ttl = Mock(return_value=3600)
        redis_mock.delete = Mock()
        redis_mock.connection_pool.connection_kwargs = {"db": 2}
        return redis_mock
    
    @pytest.fixture
    def presidio_security(self, mock_redis):
        """Create PresidioMemorySecurity instance with mocked Redis."""
        with patch('security.presidio_memory_security.redis.Redis', return_value=mock_redis):
            security = PresidioMemorySecurity(redis_client=mock_redis)
            return security
    
    def test_initialization(self, presidio_security):
        """Test Presidio security initializes correctly."""
        assert presidio_security is not None
        assert presidio_security.analyzer is not None
        assert presidio_security.anonymizer is not None
        assert len(presidio_security.pii_entities) > 0
    
    def test_scrub_before_storage_with_pii(self, presidio_security):
        """Test scrubbing text with PII before storage."""
        text = "My email is john.doe@example.com and phone is 555-123-4567"
        
        anonymized_text, vault_id, entities_found = presidio_security.scrub_before_storage(
            text,
            namespace="test_session",
            ttl_hours=24
        )
        
        assert anonymized_text != text  # Text should be modified
        assert vault_id is not None  # Vault ID should be generated
        assert len(entities_found) > 0  # Entities should be detected
    
    def test_scrub_before_storage_no_pii(self, presidio_security):
        """Test scrubbing text without PII."""
        text = "This is a normal message without sensitive data"
        
        anonymized_text, vault_id, entities_found = presidio_security.scrub_before_storage(
            text,
            namespace="test_session",
            ttl_hours=24
        )
        
        assert anonymized_text == text  # Text should be unchanged
        assert vault_id is None  # No vault needed
        assert len(entities_found) == 0  # No entities detected
    
    def test_scrub_before_storage_with_member_id(self, presidio_security):
        """Test scrubbing text with member ID."""
        text = "Member M123456 needs assistance"
        
        anonymized_text, vault_id, entities_found = presidio_security.scrub_before_storage(
            text,
            namespace="test_session",
            ttl_hours=24
        )
        
        # Member ID should be detected and anonymized
        assert "MEMBER_ID" in entities_found or "M123456" not in anonymized_text
    
    def test_scrub_before_storage_with_claim_number(self, presidio_security):
        """Test scrubbing text with claim number."""
        text = "Claim CLM-12345678 has been processed"
        
        anonymized_text, vault_id, entities_found = presidio_security.scrub_before_storage(
            text,
            namespace="test_session",
            ttl_hours=24
        )
        
        # Claim number should be detected
        assert "CLAIM_NUMBER" in entities_found or "CLM-12345678" not in anonymized_text
    
    def test_analyze_text(self, presidio_security):
        """Test analyzing text for PII/PHI."""
        text = "Contact John Doe at john.doe@example.com"
        
        entities = presidio_security.analyze_text(text)
        
        assert isinstance(entities, list)
        assert len(entities) > 0
        
        # Check entity structure
        for entity in entities:
            assert "entity_type" in entity
            assert "start" in entity
            assert "end" in entity
            assert "score" in entity
            assert "text" in entity
    
    def test_analyze_text_no_pii(self, presidio_security):
        """Test analyzing text without PII."""
        text = "This is a normal message"
        
        entities = presidio_security.analyze_text(text)
        
        assert isinstance(entities, list)
        assert len(entities) == 0
    
    def test_scrub_pii_simple(self, presidio_security):
        """Test simple PII scrubbing without vault."""
        text = "My SSN is 123-45-6789"
        
        scrubbed = presidio_security.scrub_pii(text)
        
        assert "123-45-6789" not in scrubbed
        assert scrubbed != text
    
    def test_retrieve_from_vault_not_found(self, presidio_security, mock_redis):
        """Test retrieving from vault when entry not found."""
        mock_redis.get.return_value = None
        
        vault_entry = presidio_security.retrieve_from_vault("nonexistent_vault")
        
        assert vault_entry is None
    
    def test_get_vault_stats(self, presidio_security, mock_redis):
        """Test getting vault statistics."""
        mock_redis.keys.return_value = [b"vault:1", b"vault:2", b"vault:3"]
        
        stats = presidio_security.get_vault_stats()
        
        assert "total_vaults" in stats
        assert stats["total_vaults"] == 3
    
    def test_clear_expired_vaults(self, presidio_security, mock_redis):
        """Test clearing expired vaults."""
        mock_redis.keys.return_value = [b"vault:1", b"vault:2"]
        mock_redis.ttl.side_effect = [-1, 3600]  # First has no TTL, second has TTL
        
        cleared = presidio_security.clear_expired_vaults()
        
        assert cleared == 1  # One vault without TTL should be cleared
        mock_redis.delete.assert_called_once()


class TestPresidioIntegration:
    """Test Presidio integration scenarios."""
    
    @pytest.fixture(autouse=True)
    def mock_redis_for_integration(self):
        """Mock Redis for all integration tests."""
        mock_redis = Mock()
        mock_redis.setex = Mock()
        mock_redis.get = Mock(return_value=None)
        mock_redis.keys = Mock(return_value=[])
        mock_redis.ttl = Mock(return_value=3600)
        mock_redis.delete = Mock()
        mock_redis.connection_pool.connection_kwargs = {"db": 2}
        
        with patch('security.presidio_memory_security.redis.Redis', return_value=mock_redis):
            # Reset the singleton so it gets recreated with mocked Redis
            import security.presidio_memory_security as mod
            mod._presidio_instance = None
            yield
            mod._presidio_instance = None
    
    def test_detect_email_addresses(self):
        """Test detecting email addresses."""
        presidio = get_presidio_security()
        
        text = "Contact us at support@example.com or sales@example.com"
        entities = presidio.analyze_text(text)
        
        email_entities = [e for e in entities if e["entity_type"] == "EMAIL_ADDRESS"]
        assert len(email_entities) >= 1
    
    def test_detect_phone_numbers(self):
        """Test detecting phone numbers."""
        presidio = get_presidio_security()
        
        text = "Call us at 555-123-4567 or (555) 987-6543"
        entities = presidio.analyze_text(text)
        
        phone_entities = [e for e in entities if e["entity_type"] == "PHONE_NUMBER"]
        assert len(phone_entities) >= 1
    
    def test_detect_person_names(self):
        """Test detecting person names."""
        presidio = get_presidio_security()
        
        text = "Patient John Smith visited Dr. Jane Doe"
        entities = presidio.analyze_text(text)
        
        person_entities = [e for e in entities if e["entity_type"] == "PERSON"]
        assert len(person_entities) >= 1
    
    def test_detect_multiple_pii_types(self):
        """Test detecting multiple PII types in one text."""
        presidio = get_presidio_security()
        
        text = "John Doe's email is john@example.com and phone is 555-1234"
        entities = presidio.analyze_text(text)
        
        entity_types = {e["entity_type"] for e in entities}
        assert len(entity_types) >= 2  # Should detect multiple types


class TestPresidioSingleton:
    """Test Presidio singleton pattern."""
    
    @pytest.fixture(autouse=True)
    def mock_redis_for_singleton(self):
        """Mock Redis for singleton tests."""
        mock_redis = Mock()
        mock_redis.setex = Mock()
        mock_redis.get = Mock(return_value=None)
        mock_redis.keys = Mock(return_value=[])
        mock_redis.connection_pool.connection_kwargs = {"db": 2}
        
        with patch('security.presidio_memory_security.redis.Redis', return_value=mock_redis):
            import security.presidio_memory_security as mod
            mod._presidio_instance = None
            yield
            mod._presidio_instance = None
    
    def test_get_presidio_security_singleton(self):
        """Test get_presidio_security returns singleton."""
        presidio1 = get_presidio_security()
        presidio2 = get_presidio_security()
        
        assert presidio1 is presidio2


class TestPresidioErrorHandling:
    """Test Presidio error handling."""
    
    @pytest.fixture(autouse=True)
    def mock_redis_for_errors(self):
        """Mock Redis for error handling tests."""
        mock_redis = Mock()
        mock_redis.setex = Mock()
        mock_redis.get = Mock(return_value=None)
        mock_redis.keys = Mock(return_value=[])
        mock_redis.connection_pool.connection_kwargs = {"db": 2}
        
        with patch('security.presidio_memory_security.redis.Redis', return_value=mock_redis):
            import security.presidio_memory_security as mod
            mod._presidio_instance = None
            yield
            mod._presidio_instance = None
    
    def test_scrub_with_invalid_text_type(self):
        """Test scrubbing with invalid text type."""
        presidio = get_presidio_security()
        
        # Should handle gracefully - None input returns empty string
        result = presidio.scrub_pii(None)
        assert result == ""
    
    def test_analyze_with_empty_text(self):
        """Test analyzing empty text."""
        presidio = get_presidio_security()
        
        entities = presidio.analyze_text("")
        
        assert isinstance(entities, list)
        assert len(entities) == 0


class TestPresidioPerformance:
    """Test Presidio performance characteristics."""
    
    @pytest.fixture(autouse=True)
    def mock_redis_for_perf(self):
        """Mock Redis for performance tests."""
        mock_redis = Mock()
        mock_redis.setex = Mock()
        mock_redis.get = Mock(return_value=None)
        mock_redis.keys = Mock(return_value=[])
        mock_redis.connection_pool.connection_kwargs = {"db": 2}
        
        with patch('security.presidio_memory_security.redis.Redis', return_value=mock_redis):
            import security.presidio_memory_security as mod
            mod._presidio_instance = None
            yield
            mod._presidio_instance = None
    
    def test_scrub_large_text(self):
        """Test scrubbing large text."""
        presidio = get_presidio_security()
        
        # Create large text with PII
        large_text = "Contact john@example.com. " * 1000
        
        anonymized_text, vault_id, entities_found = presidio.scrub_before_storage(
            large_text,
            namespace="perf_test",
            ttl_hours=1
        )
        
        assert anonymized_text is not None
        assert len(entities_found) > 0
    
    def test_analyze_performance(self):
        """Test analysis performance."""
        presidio = get_presidio_security()
        
        text = "This is a test message with john@example.com and 555-1234"
        
        # Should complete quickly
        import time
        start = time.time()
        entities = presidio.analyze_text(text)
        duration = time.time() - start
        
        assert duration < 1.0  # Should complete in less than 1 second
        assert len(entities) > 0