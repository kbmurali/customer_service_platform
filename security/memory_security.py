"""
Memory & Context Security Module

Implements security controls for agent memory and context:
1. TTL policies for automatic expiration
2. Namespace isolation per agent
3. Document validation before RAG ingestion
4. Integrity checks to detect tampering

Security Control #4: Memory & Context Security
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import redis
from neo4j import GraphDatabase
import chromadb

from security.prompt_injection_filter import get_prompt_filter

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory with different TTL policies"""
    SESSION = "session"  # 24 hours
    SHORT_TERM = "short_term"  # 7 days
    LONG_TERM = "long_term"  # 90 days
    RAG_DOCUMENT = "rag_document"  # 365 days


@dataclass
class ValidationResult:
    """Result of document validation"""
    valid: bool
    reason: str
    threats: List[str] = None


class MemorySecurity:
    """
    Memory and context security manager
    
    Features:
    - TTL policies for automatic expiration
    - Namespace isolation per agent
    - Document validation before ingestion
    - Integrity checks (checksums)
    """
    
    # TTL policies for different memory types
    TTL_POLICIES = {
        MemoryType.SESSION: timedelta(hours=24),
        MemoryType.SHORT_TERM: timedelta(days=7),
        MemoryType.LONG_TERM: timedelta(days=90),
        MemoryType.RAG_DOCUMENT: timedelta(days=365)
    }
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        neo4j_uri: str = "bolt://localhost:7688",  # Context Graph
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        chroma_host: str = "localhost",
        chroma_port: int = 8000
    ):
        """Initialize memory security manager"""
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )
        
        self.prompt_filter = get_prompt_filter()
        
        logger.info("Memory Security initialized")
    
    def store_memory(
        self,
        agent_id: str,
        memory_type: MemoryType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store memory with TTL and namespace isolation
        
        Args:
            agent_id: Unique agent identifier
            memory_type: Type of memory (determines TTL)
            data: Memory data to store
            metadata: Optional metadata
        
        Returns:
            memory_id: Unique identifier for stored memory
        """
        # 1. Create namespace for agent
        namespace = f"agent_{agent_id}_{memory_type.value}"
        
        # 2. Calculate TTL and expiration
        ttl = self.TTL_POLICIES[memory_type]
        expires_at = datetime.utcnow() + ttl
        
        # 3. Calculate checksum for integrity
        data_str = json.dumps(data, sort_keys=True)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()
        
        # 4. Create memory entry with metadata
        memory_id = f"{namespace}:{datetime.utcnow().timestamp()}"
        memory_entry = {
            "memory_id": memory_id,
            "namespace": namespace,
            "agent_id": agent_id,
            "memory_type": memory_type.value,
            "data": data,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "checksum": checksum
        }
        
        # 5. Store in appropriate backend
        if memory_type == MemoryType.SESSION or memory_type == MemoryType.SHORT_TERM:
            self._store_neo4j_cg(memory_entry)
        elif memory_type == MemoryType.RAG_DOCUMENT:
            self._store_chroma(memory_entry)
        
        # 6. Track TTL in Redis for fast expiration checks
        self.redis_client.setex(
            f"memory_ttl:{memory_id}",
            int(ttl.total_seconds()),
            expires_at.isoformat()
        )
        
        logger.info(f"Stored memory {memory_id} with TTL {ttl}")
        return memory_id
    
    def retrieve_memory(
        self,
        memory_id: str,
        requesting_agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve memory with validation
        
        Args:
            memory_id: Memory identifier
            requesting_agent_id: Agent requesting the memory
        
        Returns:
            Memory data if valid, None if expired/tampered/unauthorized
        """
        # 1. Check TTL (fast check via Redis)
        if not self._is_memory_valid(memory_id):
            logger.warning(f"Memory {memory_id} expired or invalid")
            return None
        
        # 2. Check namespace authorization
        if not self._is_authorized(memory_id, requesting_agent_id):
            logger.warning(f"Agent {requesting_agent_id} not authorized for {memory_id}")
            return None
        
        # 3. Retrieve from backend
        memory_entry = self._retrieve_from_backend(memory_id)
        if not memory_entry:
            return None
        
        # 4. Verify integrity (checksum)
        if not self._verify_integrity(memory_entry):
            logger.error(f"Memory {memory_id} integrity check failed - possible tampering")
            return None
        
        return memory_entry["data"]
    
    def validate_document(
        self,
        document: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate document before RAG ingestion
        
        Prevents memory poisoning by scanning documents for:
        - Prompt injection attempts
        - Malicious content
        - Suspicious patterns
        
        Args:
            document: Document text to validate
            source: Source of the document (for audit)
            metadata: Optional document metadata
        
        Returns:
            ValidationResult with validation status
        """
        threats = []
        
        # 1. Scan for prompt injection
        scan_result = self.prompt_filter.scan_input(document, user_id=f"document_validation:{source}")
        if not scan_result.safe:
            threats.append(f"Prompt injection detected: {scan_result.reason}")
            return ValidationResult(
                valid=False,
                reason="Prompt injection detected in document",
                threats=threats
            )
        
        # 2. Check for malicious patterns
        if self._contains_malicious_patterns(document):
            threats.append("Malicious patterns detected")
            return ValidationResult(
                valid=False,
                reason="Malicious content detected",
                threats=threats
            )
        
        # 3. Verify source authenticity (if metadata available)
        if metadata and "source_url" in metadata:
            if not self._verify_source(metadata["source_url"]):
                threats.append(f"Untrusted source: {metadata['source_url']}")
                return ValidationResult(
                    valid=False,
                    reason="Untrusted document source",
                    threats=threats
                )
        
        # 4. Check document size (prevent DoS)
        if len(document) > 1_000_000:  # 1MB limit
            threats.append("Document too large")
            return ValidationResult(
                valid=False,
                reason="Document exceeds size limit",
                threats=threats
            )
        
        return ValidationResult(valid=True, reason="Document validated successfully")
    
    def cleanup_expired_memories(self) -> int:
        """
        Cleanup expired memories from all backends
        
        Returns:
            Number of memories cleaned up
        """
        cleaned_count = 0
        
        # Get all memory TTL keys
        ttl_keys = self.redis_client.keys("memory_ttl:*")
        
        for ttl_key in ttl_keys:
            # Check if expired
            expires_at_str = self.redis_client.get(ttl_key)
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.utcnow() > expires_at:
                    memory_id = ttl_key.replace("memory_ttl:", "")
                    self._delete_memory(memory_id)
                    self.redis_client.delete(ttl_key)
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} expired memories")
        return cleaned_count
    
    def _is_memory_valid(self, memory_id: str) -> bool:
        """Check if memory is still valid (not expired)"""
        ttl_key = f"memory_ttl:{memory_id}"
        expires_at_str = self.redis_client.get(ttl_key)
        
        if not expires_at_str:
            return False
        
        expires_at = datetime.fromisoformat(expires_at_str)
        return datetime.utcnow() <= expires_at
    
    def _is_authorized(self, memory_id: str, requesting_agent_id: str) -> bool:
        """Check if agent is authorized to access memory"""
        # Extract namespace from memory_id
        namespace = memory_id.split(":")[0]
        
        # Check if agent_id matches namespace
        expected_prefix = f"agent_{requesting_agent_id}_"
        return namespace.startswith(expected_prefix)
    
    def _verify_integrity(self, memory_entry: Dict[str, Any]) -> bool:
        """Verify memory integrity using checksum"""
        stored_checksum = memory_entry.get("checksum")
        if not stored_checksum:
            return False
        
        # Recalculate checksum
        data_str = json.dumps(memory_entry["data"], sort_keys=True)
        calculated_checksum = hashlib.sha256(data_str.encode()).hexdigest()
        
        return stored_checksum == calculated_checksum
    
    def _contains_malicious_patterns(self, document: str) -> bool:
        """Check for known malicious patterns in document"""
        malicious_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",  # JavaScript protocol
            r"data:text/html",  # Data URL injection
            r"eval\s*\(",  # Code evaluation
            r"exec\s*\(",  # Code execution
        ]
        
        import re
        for pattern in malicious_patterns:
            if re.search(pattern, document, re.IGNORECASE | re.DOTALL):
                return True
        
        return False
    
    def _verify_source(self, source_url: str) -> bool:
        """Verify document source is trusted"""
        # Allowlist of trusted domains
        trusted_domains = [
            "healthcare.gov",
            "cms.gov",
            "irs.gov",
            "medicare.gov"
        ]
        
        from urllib.parse import urlparse
        parsed = urlparse(source_url)
        domain = parsed.netloc.lower()
        
        return any(domain.endswith(trusted) for trusted in trusted_domains)
    
    def _store_neo4j_cg(self, memory_entry: Dict[str, Any]):
        """Store memory in Neo4j Context Graph"""
        with self.neo4j_driver.session() as session:
            session.run("""
                CREATE (m:Memory {
                    memory_id: $memory_id,
                    namespace: $namespace,
                    agent_id: $agent_id,
                    memory_type: $memory_type,
                    data: $data,
                    metadata: $metadata,
                    created_at: $created_at,
                    expires_at: $expires_at,
                    checksum: $checksum
                })
            """, **{
                "memory_id": memory_entry["memory_id"],
                "namespace": memory_entry["namespace"],
                "agent_id": memory_entry["agent_id"],
                "memory_type": memory_entry["memory_type"],
                "data": json.dumps(memory_entry["data"]),
                "metadata": json.dumps(memory_entry["metadata"]),
                "created_at": memory_entry["created_at"],
                "expires_at": memory_entry["expires_at"],
                "checksum": memory_entry["checksum"]
            })
    
    def _store_chroma(self, memory_entry: Dict[str, Any]):
        """Store document in Chroma with namespace"""
        collection_name = memory_entry["namespace"]
        
        try:
            collection = self.chroma_client.get_or_create_collection(collection_name)
        except:
            collection = self.chroma_client.create_collection(collection_name)
        
        collection.add(
            ids=[memory_entry["memory_id"]],
            documents=[json.dumps(memory_entry["data"])],
            metadatas=[{
                "agent_id": memory_entry["agent_id"],
                "created_at": memory_entry["created_at"],
                "expires_at": memory_entry["expires_at"],
                "checksum": memory_entry["checksum"]
            }]
        )
    
    def _retrieve_from_backend(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory from appropriate backend"""
        # Determine backend from memory_id
        if "session" in memory_id or "short_term" in memory_id:
            return self._retrieve_neo4j_cg(memory_id)
        elif "rag_document" in memory_id:
            return self._retrieve_chroma(memory_id)
        return None
    
    def _retrieve_neo4j_cg(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve from Neo4j Context Graph"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Memory {memory_id: $memory_id})
                RETURN m
            """, memory_id=memory_id)
            
            record = result.single()
            if not record:
                return None
            
            node = record["m"]
            return {
                "memory_id": node["memory_id"],
                "namespace": node["namespace"],
                "agent_id": node["agent_id"],
                "memory_type": node["memory_type"],
                "data": json.loads(node["data"]),
                "metadata": json.loads(node["metadata"]),
                "created_at": node["created_at"],
                "expires_at": node["expires_at"],
                "checksum": node["checksum"]
            }
    
    def _retrieve_chroma(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve from Chroma"""
        namespace = memory_id.split(":")[0]
        
        try:
            collection = self.chroma_client.get_collection(namespace)
            result = collection.get(ids=[memory_id])
            
            if not result["ids"]:
                return None
            
            return {
                "memory_id": memory_id,
                "namespace": namespace,
                "agent_id": result["metadatas"][0]["agent_id"],
                "memory_type": "rag_document",
                "data": json.loads(result["documents"][0]),
                "metadata": result["metadatas"][0],
                "created_at": result["metadatas"][0]["created_at"],
                "expires_at": result["metadatas"][0]["expires_at"],
                "checksum": result["metadatas"][0]["checksum"]
            }
        except:
            return None
    
    def _delete_memory(self, memory_id: str):
        """Delete memory from backend"""
        if "session" in memory_id or "short_term" in memory_id:
            self._delete_neo4j_cg(memory_id)
        elif "rag_document" in memory_id:
            self._delete_chroma(memory_id)
    
    def _delete_neo4j_cg(self, memory_id: str):
        """Delete from Neo4j Context Graph"""
        with self.neo4j_driver.session() as session:
            session.run("""
                MATCH (m:Memory {memory_id: $memory_id})
                DELETE m
            """, memory_id=memory_id)
    
    def _delete_chroma(self, memory_id: str):
        """Delete from Chroma"""
        namespace = memory_id.split(":")[0]
        try:
            collection = self.chroma_client.get_collection(namespace)
            collection.delete(ids=[memory_id])
        except:
            pass


# Global instance
_memory_security: Optional[MemorySecurity] = None


def get_memory_security() -> MemorySecurity:
    """Get global memory security instance"""
    global _memory_security
    if _memory_security is None:
        _memory_security = MemorySecurity()
    return _memory_security
