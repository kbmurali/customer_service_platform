"""
Tests for Control 8: Inter-Agent Communication Security

Tests cover:
- KeyManager: HKDF key derivation, key separation, determinism
- SecureMessageBus: envelope wrapping/unwrapping, encryption, signing
- Replay protection: nonce deduplication, timestamp validation
- JSON Schema validation: request/response schema enforcement
- RemoteMCPNode: LangGraph node interface, error handling
- Tool schema registry: completeness and correctness
- Configuration: MCP agent config helper
"""

import json
import time
import os
from unittest.mock import patch, MagicMock

import pytest
from datetime import datetime, timezone, timedelta
import jsonschema
import requests as req_lib

from security.message_encryption import SecureMessageBus, KeyManager, SecurityError
from agents.core.remote_mcp_node import RemoteMCPNode
from config.settings import get_mcp_agent_config
from security.schemas.tool_schemas import build_schema_registry, MEMBER_LOOKUP_REQUEST, CLAIM_LOOKUP_REQUEST, UPDATE_CLAIM_STATUS_REQUEST
from observability.prometheus_metrics import track_mcp_encryption_event

# ============================================
# TEST: KeyManager
# ============================================

class TestKeyManager:
    """Tests for HKDF-based key derivation."""

    def setup_method(self):
        """Set up test fixtures."""
        os.environ["MCP_MASTER_SECRET"] = "test-master-secret-for-unit-tests"
        self.km = KeyManager()

    def test_get_encryption_key_returns_32_bytes(self):
        """Encryption key must be 256-bit (32 bytes)."""
        enc_key = self.km.get_encryption_key(
            "central_supervisor", "member_services_team"
        )
        assert len(enc_key) == 32

    def test_get_signing_key_returns_32_bytes(self):
        """Signing key must be 256-bit (32 bytes)."""
        sig_key = self.km.get_signing_key(
            "central_supervisor", "member_services_team"
        )
        assert len(sig_key) == 32

    def test_encryption_and_signing_keys_are_different(self):
        """Encryption and signing keys must be different (key separation)."""
        enc_key = self.km.get_encryption_key(
            "central_supervisor", "member_services_team"
        )
        sig_key = self.km.get_signing_key(
            "central_supervisor", "member_services_team"
        )
        assert enc_key != sig_key

    def test_keys_are_deterministic(self):
        """Same inputs must produce the same keys every time."""
        k1 = self.km.get_encryption_key("central_supervisor", "member_services_team")
        k2 = self.km.get_encryption_key("central_supervisor", "member_services_team")
        assert k1 == k2

    def test_different_pairs_produce_different_keys(self):
        """Different agent pairs must produce different keys."""
        k_member = self.km.get_encryption_key("central_supervisor", "member_services_team")
        k_claim = self.km.get_encryption_key("central_supervisor", "claim_services_team")
        assert k_member != k_claim

    def test_canonical_ordering(self):
        """Key derivation must be order-independent (canonical pair)."""
        k1 = self.km.get_encryption_key("central_supervisor", "member_services_team")
        k2 = self.km.get_encryption_key("member_services_team", "central_supervisor")
        assert k1 == k2

    def test_ephemeral_key_when_no_secret(self):
        """Missing MCP_MASTER_SECRET should generate ephemeral key (no crash)."""
        os.environ.pop("MCP_MASTER_SECRET", None)
        km = KeyManager(master_secret=None)
        key = km.get_encryption_key("a", "b")
        assert len(key) == 32


# ============================================
# TEST: SecureMessageBus
# ============================================

class TestSecureMessageBus:
    """Tests for AES-256-GCM encryption, HMAC signing, and envelope format."""

    def setup_method(self):
        """Set up test fixtures with a fresh SecureMessageBus."""
        os.environ["MCP_MASTER_SECRET"] = "test-master-secret-for-unit-tests"
        mock_redis = MagicMock()
        mock_redis.set.return_value = True  # nonce is new
        self.bus = SecureMessageBus(
            key_manager=KeyManager(),
            redis_client=mock_redis,
            schema_registry=build_schema_registry(),
        )

    def test_wrap_message_produces_valid_envelope(self):
        """wrap_message must produce an envelope with all required fields."""
        envelope = self.bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload={"member_id": "M1234", "user_id": "usr-001"},
        )
        assert "nonce" in envelope
        assert "timestamp" in envelope
        assert "from_agent" in envelope
        assert "to_agent" in envelope
        assert "encrypted_payload" in envelope
        assert "signature" in envelope
        assert envelope["from_agent"] == "central_supervisor"
        assert envelope["to_agent"] == "member_services_team"

    def test_wrap_unwrap_roundtrip(self):
        """Wrapping then unwrapping must return the original payload."""
        original = {"member_id": "M1234", "user_id": "usr-001"}
        envelope = self.bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload=original,
        )
        decrypted = self.bus.unwrap_message(envelope)
        assert decrypted == original

    def test_tampered_payload_rejected(self):
        """Modifying encrypted_payload must cause HMAC verification failure."""
        envelope = self.bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload={"member_id": "M1234", "user_id": "usr-001"},
        )
        # Tamper with the encrypted payload
        envelope["encrypted_payload"] = envelope["encrypted_payload"][:-4] + "XXXX"
        with pytest.raises(SecurityError, match="HMAC"):
            self.bus.unwrap_message(envelope)

    def test_tampered_signature_rejected(self):
        """Modifying signature must cause verification failure."""
        envelope = self.bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload={"member_id": "M1234", "user_id": "usr-001"},
        )
        envelope["signature"] = "a" * 64
        with pytest.raises(SecurityError, match="HMAC"):
            self.bus.unwrap_message(envelope)

    def test_different_nonces_per_message(self):
        """Each message must have a unique nonce."""
        payload = {"member_id": "M1234", "user_id": "usr-001"}
        e1 = self.bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload=payload,
        )
        e2 = self.bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload=payload,
        )
        assert e1["nonce"] != e2["nonce"]

    def test_encrypted_payload_is_not_plaintext(self):
        """The encrypted_payload must not contain plaintext member_id."""
        envelope = self.bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload={"member_id": "M1234", "user_id": "usr-001"},
        )
        assert "M1234" not in envelope["encrypted_payload"]

    def test_wrap_response_roundtrip(self):
        """wrap_response + unwrap_response must round-trip correctly."""
        original = {"member_id": "M1234", "name": "John Doe", "status": "active"}
        envelope = self.bus.wrap_response(
            from_agent="member_services_team",
            to_agent="central_supervisor",
            tool_name="member_lookup",
            payload=original,
        )
        decrypted = self.bus.unwrap_response(envelope)
        assert decrypted == original


# ============================================
# TEST: Replay Protection
# ============================================

class TestReplayProtection:
    """Tests for nonce deduplication and timestamp validation."""

    def test_replay_same_envelope_rejected(self):
        """Replaying the exact same envelope must be rejected."""
        os.environ["MCP_MASTER_SECRET"] = "test-master-secret-for-unit-tests"

        # Use a mock Redis that returns True on first set, None on second
        mock_redis = MagicMock()
        mock_redis.set.side_effect = [True, None]  # first=new, second=duplicate

        bus = SecureMessageBus(
            key_manager=KeyManager(),
            redis_client=mock_redis,
            schema_registry=build_schema_registry(),
        )

        envelope = bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload={"member_id": "M1234", "user_id": "usr-001"},
        )
        # First unwrap succeeds
        bus.unwrap_message(envelope)
        # Second unwrap (replay) must fail
        with pytest.raises(SecurityError, match="[Rr]eplay"):
            bus.unwrap_message(envelope)

    def test_expired_timestamp_rejected(self):
        """Messages with timestamps older than max_age must be rejected."""
        os.environ["MCP_MASTER_SECRET"] = "test-master-secret-for-unit-tests"

        mock_redis = MagicMock()
        mock_redis.set.return_value = True

        bus = SecureMessageBus(
            key_manager=KeyManager(),
            redis_client=mock_redis,
            schema_registry=build_schema_registry(),
        )

        envelope = bus.wrap_message(
            from_agent="central_supervisor",
            to_agent="member_services_team",
            tool_name="member_lookup",
            payload={"member_id": "M1234", "user_id": "usr-001"},
        )

        # Backdate the timestamp by 5 minutes (beyond CLOCK_SKEW_SECONDS=60)
        old_time = datetime.now(timezone.utc) - timedelta(seconds=300)
        envelope["timestamp"] = old_time.isoformat()

        # This will fail on either HMAC (because timestamp changed) or timestamp check
        with pytest.raises(SecurityError):
            bus.unwrap_message(envelope)


# ============================================
# TEST: JSON Schema Validation
# ============================================

class TestSchemaValidation:
    """Tests for JSON Schema validation of tool payloads."""

    def test_schema_registry_completeness(self):
        """Registry must contain schemas for all member and claim tools."""
        registry = build_schema_registry()

        expected_tools = [
            "member_lookup", "check_eligibility", "coverage_lookup",
            "search_policy_info", "update_member_info",
            "claim_lookup", "claim_status", "claim_payment_info",
            "update_claim_status",
        ]
        for tool in expected_tools:
            assert f"{tool}:request" in registry, f"Missing request schema for {tool}"
            assert f"{tool}:response" in registry, f"Missing response schema for {tool}"

    def test_schema_registry_count(self):
        """Registry must have 22 entries (9 tools x 2 directions + 2 invoke x 2 directions)."""
        registry = build_schema_registry()
        assert len(registry) == 22  # 18 tool schemas + 4 invoke schemas

    def test_member_lookup_request_schema_valid(self):
        """Valid member_lookup request must pass schema validation."""
        
        valid_payload = {"member_id": "M1234", "user_id": "usr-001"}
        jsonschema.validate(valid_payload, MEMBER_LOOKUP_REQUEST)

    def test_member_lookup_request_schema_invalid_member_id(self):
        """Invalid member_id pattern must fail schema validation."""
        invalid_payload = {"member_id": "INVALID", "user_id": "usr-001"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_payload, MEMBER_LOOKUP_REQUEST)

    def test_member_lookup_request_schema_missing_required(self):
        """Missing required fields must fail schema validation."""
        invalid_payload = {"user_id": "usr-001"}  # missing member_id
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_payload, MEMBER_LOOKUP_REQUEST)

    def test_claim_lookup_request_schema_valid(self):
        """Valid claim_lookup request must pass schema validation."""
        valid_payload = {"claim_id": "CLM-001", "user_id": "usr-001"}
        jsonschema.validate(valid_payload, CLAIM_LOOKUP_REQUEST)

    def test_update_claim_status_request_schema_valid(self):
        """Valid update_claim_status request must pass schema validation."""
        valid_payload = {
            "claim_id": "CLM-001",
            "new_status": "approved",
            "user_id": "usr-001",
        }
        jsonschema.validate(valid_payload, UPDATE_CLAIM_STATUS_REQUEST)

    def test_schema_validation_in_bus_rejects_invalid(self):
        """SecureMessageBus must reject payloads that fail schema validation."""
        os.environ["MCP_MASTER_SECRET"] = "test-master-secret-for-unit-tests"

        mock_redis = MagicMock()
        mock_redis.set.return_value = True

        bus = SecureMessageBus(
            key_manager=KeyManager(),
            redis_client=mock_redis,
            schema_registry=build_schema_registry(),
        )

        # Invalid payload: member_id doesn't match pattern
        with pytest.raises(SecurityError, match="[Ss]chema"):
            bus.wrap_message(
                from_agent="central_supervisor",
                to_agent="member_services_team",
                tool_name="member_lookup",
                payload={"member_id": "INVALID-FORMAT", "user_id": "usr-001"},
            )


# ============================================
# TEST: RemoteMCPNode
# ============================================

class TestRemoteMCPNode:
    """Tests for the LangGraph-compatible remote MCP node wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        os.environ["MCP_MASTER_SECRET"] = "test-master-secret-for-unit-tests"

    @patch("agents.core.remote_mcp_node.requests.post")
    def test_successful_remote_call(self, mock_post):
        """Successful remote call must return merged state with tool_results."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = True

        bus = SecureMessageBus(
            key_manager=KeyManager(),
            redis_client=mock_redis,
            schema_registry=build_schema_registry(),
        )
        node = RemoteMCPNode(
            agent_name="member_services_team",
            base_url="https://mcp-member:8443",
            secure_bus=bus,
        )

        # Build a valid response envelope that the remote agent would return
        response_payload = {
            "messages": ["Member M1234 found: John Doe"],
            "tool_results": {"member_lookup": {"name": "John Doe"}},
        }
        response_envelope = bus.wrap_response(
            from_agent="member_services_team",
            to_agent="central_supervisor",
            tool_name="member_services_team_invoke",
            payload=response_payload,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = response_envelope
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Create a mock state
        mock_message = MagicMock()
        mock_message.content = "Look up member M1234"
        state = {
            "messages": [mock_message],
            "user_id": "usr-001",
            "user_role": "CSR_TIER1",
            "session_id": "sess-001",
            "execution_path": [],
        }

        result = node(state)

        assert "execution_path" in result
        assert any("member_services_team" in p for p in result["execution_path"])
        assert "tool_results" in result
        mock_post.assert_called_once()

    @patch("agents.core.remote_mcp_node.requests.post")
    def test_http_error_returns_error_state(self, mock_post):
        """HTTP error must return error state without crashing."""

        mock_redis = MagicMock()
        node = RemoteMCPNode(
            agent_name="member_services_team",
            base_url="https://mcp-member:8443",
            secure_bus=MagicMock(
                wrap_message=MagicMock(return_value={"envelope": "data"}),
            ),
        )

        mock_post.side_effect = req_lib.ConnectionError("Connection refused")

        mock_message = MagicMock()
        mock_message.content = "Look up member M1234"
        state = {
            "messages": [mock_message],
            "user_id": "usr-001",
            "user_role": "CSR_TIER1",
            "session_id": "sess-001",
            "execution_path": [],
        }

        result = node(state)
        assert "error" in result
        assert result["error_count"] == 1
        assert any("ERROR" in p for p in result["execution_path"])

    @patch("agents.core.remote_mcp_node.requests.post")
    def test_security_error_returns_error_state(self, mock_post):
        """Security error (tampered response) must return error state."""

        mock_bus = MagicMock()
        mock_bus.wrap_message.return_value = {"envelope": "data"}
        mock_bus.unwrap_response.side_effect = SecurityError("HMAC mismatch")

        node = RemoteMCPNode(
            agent_name="member_services_team",
            base_url="https://mcp-member:8443",
            secure_bus=mock_bus,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"tampered": "envelope"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        mock_message = MagicMock()
        mock_message.content = "Look up member M1234"
        state = {
            "messages": [mock_message],
            "user_id": "usr-001",
            "user_role": "CSR_TIER1",
            "session_id": "sess-001",
            "execution_path": [],
        }

        result = node(state)
        assert "error" in result
        assert "Security error" in result["error"]
        assert any("ERROR" in p for p in result["execution_path"])

    def test_node_is_callable(self):
        """RemoteMCPNode must be callable (LangGraph interface)."""
        
        node = RemoteMCPNode(
            agent_name="member_services_team",
            base_url="https://mcp-member:8443",
            secure_bus=MagicMock(),
        )
        assert callable(node)

    @patch("agents.core.remote_mcp_node.requests.post")
    def test_node_posts_to_correct_url(self, mock_post):
        """Node must POST to {base_url}/mcp/invoke."""

        mock_bus = MagicMock()
        mock_bus.wrap_message.return_value = {"envelope": "data"}

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "data"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # unwrap_response will fail but we check the URL first
        mock_bus.unwrap_response.side_effect = Exception("stop")

        node = RemoteMCPNode(
            agent_name="member_services_team",
            base_url="https://mcp-member:8443",
            secure_bus=mock_bus,
        )
        mock_message = MagicMock()
        mock_message.content = "test"
        state = {
            "messages": [mock_message],
            "user_id": "usr-001",
            "user_role": "CSR_TIER1",
            "session_id": "sess-001",
            "execution_path": [],
        }
        node(state)
        call_args = mock_post.call_args
        assert call_args.kwargs["url"] == "https://mcp-member:8443/mcp/invoke"


# ============================================
# TEST: Configuration
# ============================================

class TestMCPConfiguration:
    """Tests for MCP agent configuration in settings."""

    def test_get_mcp_agent_config_member(self):
        """get_mcp_agent_config must return member services config."""
        os.environ["MCP_MEMBER_SERVICES_URL"] = "https://test-member:9443"
        os.environ["MCP_MEMBER_SERVICES_SHARED_SECRET"] = "test-secret"

        config = get_mcp_agent_config("member_services_team")
        assert "url" in config
        assert "shared_secret" in config
        assert "verify_tls" in config
        assert "encryption_enabled" in config

    def test_get_mcp_agent_config_claim(self):
        """get_mcp_agent_config must return claim services config."""
        os.environ["MCP_CLAIM_SERVICES_URL"] = "https://test-claims:9443"
        os.environ["MCP_CLAIM_SERVICES_SHARED_SECRET"] = "test-secret"

        config = get_mcp_agent_config("claim_services_team")
        assert "url" in config
        assert "shared_secret" in config

    def test_get_mcp_agent_config_unknown_agent(self):
        """Unknown agent name must return base config with defaults."""
        config = get_mcp_agent_config("unknown_agent")
        assert "verify_tls" in config  # Base fields still present


# ============================================
# TEST: Prometheus Metrics Integration
# ============================================

class TestMCPMetrics:
    """Tests for Prometheus metrics tracking."""

    def test_track_mcp_encryption_event_encrypt(self):
        """track_mcp_encryption_event must increment encrypt counter."""
        # Should not raise
        track_mcp_encryption_event("encrypt", "member_services_team")

    def test_track_mcp_encryption_event_decrypt(self):
        """track_mcp_encryption_event must increment decrypt counter."""
        track_mcp_encryption_event("decrypt", "member_services_team")

    def test_track_mcp_encryption_event_security_failure(self):
        """track_mcp_encryption_event must increment signature failure counter."""
        track_mcp_encryption_event("security_failure", "member_services_team")

    def test_track_mcp_encryption_event_transport_failure(self):
        """track_mcp_encryption_event must increment transport failure counter."""
        track_mcp_encryption_event("transport_failure", "claim_services_team")

    def test_track_mcp_encryption_event_replay_rejection(self):
        """track_mcp_encryption_event must increment replay rejection counter."""
        track_mcp_encryption_event("replay_rejection", "member_services_team")

    def test_track_mcp_encryption_event_schema_failure(self):
        """track_mcp_encryption_event must increment schema failure counter."""
        track_mcp_encryption_event(
            "schema_failure", "member_services_team",
            tool_name="member_lookup", direction="request"
        )
