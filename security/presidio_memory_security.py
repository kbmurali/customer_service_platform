"""
Presidio Memory Security for Health Insurance AI Platform

This module uses Microsoft Presidio Analyzer and Anonymizer to scrub PII/PHI
from content before it enters vector stores, RAG indexes, or long-term memory.

Control 4: Memory & Context Security
- PII/PHI detection before storage
- Reversible anonymization with vault pattern
- Custom recognizers for healthcare domain
- Integration with Chroma vector DB and Neo4j Context Graph
"""

import os
import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import redis

logger = logging.getLogger(__name__)


class HealthcareRecognizers:
    """Custom Presidio recognizers for healthcare domain."""
    
    @staticmethod
    def get_member_id_recognizer() -> PatternRecognizer:
        """Recognizer for health insurance member IDs."""
        patterns = [
            Pattern(name="member_id_pattern", regex=r"\b[A-Z]{1,2}\d{6,8}\b", score=0.85),
        ]
        return PatternRecognizer(
            supported_entity="MEMBER_ID",
            patterns=patterns,
            name="MemberIDRecognizer",
            supported_language="en"
        )
    
    @staticmethod
    def get_policy_number_recognizer() -> PatternRecognizer:
        """Recognizer for insurance policy numbers."""
        patterns = [
            Pattern(name="policy_pattern", regex=r"\bPOL-\d{8,10}\b", score=0.85),
        ]
        return PatternRecognizer(
            supported_entity="POLICY_NUMBER",
            patterns=patterns,
            name="PolicyNumberRecognizer",
            supported_language="en"
        )
    
    @staticmethod
    def get_claim_number_recognizer() -> PatternRecognizer:
        """Recognizer for claim numbers."""
        patterns = [
            Pattern(name="claim_pattern", regex=r"\bCLM-\d{5,8}\b", score=0.85),
        ]
        return PatternRecognizer(
            supported_entity="CLAIM_NUMBER",
            patterns=patterns,
            name="ClaimNumberRecognizer",
            supported_language="en"
        )
    
    @staticmethod
    def get_pa_number_recognizer() -> PatternRecognizer:
        """Recognizer for prior authorization numbers."""
        patterns = [
            Pattern(name="pa_pattern", regex=r"\bPA-\d{4}-\d{4,6}\b", score=0.85),
        ]
        return PatternRecognizer(
            supported_entity="PA_NUMBER",
            patterns=patterns,
            name="PANumberRecognizer",
            supported_language="en"
        )


class PresidioMemorySecurity:
    """
    Manages PII/PHI scrubbing for memory and context security using Presidio.
    Implements reversible anonymization with vault pattern for authorized access.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize Presidio engines and custom recognizers."""
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Add custom healthcare recognizers
        self._add_custom_recognizers()
        
        # Initialize vault for reversible anonymization
        self.redis_client = redis_client or self._get_redis_client()
        self.encryption_key = self._get_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Define entities to detect
        self.pii_entities = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD",
            "US_DRIVER_LICENSE", "US_PASSPORT", "LOCATION", "DATE_TIME",
            "MEDICAL_LICENSE", "MEMBER_ID", "POLICY_NUMBER", "CLAIM_NUMBER", "PA_NUMBER"
        ]
        
        logger.info("Presidio Memory Security initialized")
    
    def _add_custom_recognizers(self):
        """Add healthcare-specific recognizers to analyzer."""
        recognizers = [
            HealthcareRecognizers.get_member_id_recognizer(),
            HealthcareRecognizers.get_policy_number_recognizer(),
            HealthcareRecognizers.get_claim_number_recognizer(),
            HealthcareRecognizers.get_pa_number_recognizer()
        ]
        
        for recognizer in recognizers:
            self.analyzer.registry.add_recognizer(recognizer)
            logger.info(f"Added custom recognizer: {recognizer.name}")
    
    def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for vault storage."""
        return redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_VAULT_DB", 2)),
            decode_responses=False
        )
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for vault."""
        key_env = os.getenv("VAULT_ENCRYPTION_KEY")
        if key_env:
            return key_env.encode()
        
        key = Fernet.generate_key()
        logger.warning("Generated new vault encryption key - store this securely!")
        return key
    
    def scrub_before_storage(
        self,
        text: str,
        namespace: str,
        ttl_hours: int = 24
    ) -> Tuple[str, str, Dict[str, int]]:
        """
        Scrub PII/PHI from text before storing in memory/vector DB.
        
        Args:
            text: Text to scrub
            namespace: Namespace for vault (e.g., "session:abc123")
            ttl_hours: How long to keep vault entry
            
        Returns:
            Tuple of (anonymized_text, vault_id, entities_found)
            - anonymized_text: Text with PII/PHI replaced
            - vault_id: Unique identifier for vault entry
            - entities_found: Dict mapping entity_type to count
        """
        # Analyze for PII/PHI
        results = self.analyzer.analyze(
            text=text,
            language='en',
            entities=self.pii_entities
        )
        
        # Count entities by type
        entities_found = {}
        for result in results:
            entity_type = result.entity_type
            entities_found[entity_type] = entities_found.get(entity_type, 0) + 1
        
        if not results:
            return text, None, {}
        
        # Create vault ID
        vault_id = self._generate_vault_id(namespace)
        
        # Define anonymization operators
        operators = self._get_anonymization_operators(namespace, vault_id)
        
        # Anonymize text
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )
        
        # Store vault for de-anonymization
        self._store_vault(vault_id, results, text, ttl_hours)
        
        logger.info(
            f"Scrubbed {len(results)} PII/PHI entities from text (vault: {vault_id}): "
            f"{entities_found}"
        )
        
        return anonymized_result.text, vault_id, entities_found
    
    def _get_anonymization_operators(self, namespace: str, vault_id: str) -> Dict[str, OperatorConfig]:
        """Define how each entity type should be anonymized."""
        return {
            "PERSON": OperatorConfig("hash", {"hash_type": "sha256"}),
            "SSN": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 11, "from_end": False}),
            "MEMBER_ID": OperatorConfig("encrypt", {"key": vault_id[:32]}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 7, "from_end": True}),
            "POLICY_NUMBER": OperatorConfig("encrypt", {"key": vault_id[:32]}),
            "CLAIM_NUMBER": OperatorConfig("encrypt", {"key": vault_id[:32]}),
            "PA_NUMBER": OperatorConfig("encrypt", {"key": vault_id[:32]}),
            "CREDIT_CARD": OperatorConfig("redact", {}),
            "US_DRIVER_LICENSE": OperatorConfig("redact", {}),
            "MEDICAL_LICENSE": OperatorConfig("redact", {}),
        }
    
    def _generate_vault_id(self, namespace: str) -> str:
        """Generate unique vault ID."""
        timestamp = datetime.utcnow().isoformat()
        data = f"{namespace}:{timestamp}".encode()
        return hashlib.sha256(data).hexdigest()
    
    def _store_vault(
        self,
        vault_id: str,
        analyzer_results: List,
        original_text: str,
        ttl_hours: int
    ):
        """Store vault entry for de-anonymization."""
        vault_entry = {
            "vault_id": vault_id,
            "timestamp": datetime.utcnow().isoformat(),
            "entities": [
                {
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": original_text[result.start:result.end]
                }
                for result in analyzer_results
            ]
        }
        
        # Encrypt vault entry
        encrypted_vault = self.cipher.encrypt(json.dumps(vault_entry).encode())
        
        # Store in Redis with TTL
        ttl_seconds = ttl_hours * 3600
        self.redis_client.setex(
            f"vault:{vault_id}",
            ttl_seconds,
            encrypted_vault
        )
        
        logger.debug(f"Stored vault {vault_id} with TTL {ttl_hours}h")


# Singleton instance
_presidio_instance: Optional[PresidioMemorySecurity] = None


def get_presidio_security() -> PresidioMemorySecurity:
    """Get or create singleton Presidio security instance."""
    global _presidio_instance
    if _presidio_instance is None:
        _presidio_instance = PresidioMemorySecurity()
    return _presidio_instance
