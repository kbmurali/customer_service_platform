"""
Control 8: Inter-Agent Communication Security

Provides encrypted, signed, schema-validated, and replay-protected messaging
for remote MCP agent communication.

Components:
    - KeyManager: HKDF-based per-agent-pair key derivation
    - SecureMessageBus: AES-256-GCM encryption, HMAC-SHA256 signing,
      JSON Schema validation, and nonce/timestamp replay protection
"""

import hmac
import json
import logging
import os
import secrets
from base64 import b64decode, b64encode
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import jsonschema

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

import redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NONCE_BYTES = 12          # 96-bit nonce for AES-GCM (NIST recommended)
KEY_BYTES = 32            # 256-bit keys
HMAC_DIGEST = "sha256"
CLOCK_SKEW_SECONDS = 60   # Maximum allowed clock skew
NONCE_TTL_SECONDS = 300    # Redis nonce expiry (5 minutes)
DEFAULT_RATE_LIMIT = 120   # Max messages per minute per agent pair


# ---------------------------------------------------------------------------
# KeyManager - HKDF-based per-agent-pair key derivation
# ---------------------------------------------------------------------------
class KeyManager:
    """
    Derives per-agent-pair encryption and signing keys from a shared master
    secret using HKDF (RFC 5869).  Each agent pair gets two independent keys:
    one for AES-256-GCM encryption and one for HMAC-SHA256 signing.
    """

    def __init__(self, master_secret: Optional[str] = None):
        """
        Args:
            master_secret: Base64-encoded master secret.  Falls back to
                           the MCP_MASTER_SECRET environment variable.
        """
        raw = master_secret or os.getenv("MCP_MASTER_SECRET", "")
        if not raw:
            logger.warning(
                "MCP_MASTER_SECRET not set - generating ephemeral key. "
                "This is acceptable for testing but NOT for production."
            )
            raw = secrets.token_urlsafe(32)
        # Normalise to bytes
        self._master = raw.encode("utf-8") if isinstance(raw, str) else raw

    # ── public API ────────────────────────────────────────────────────────
    def get_encryption_key(self, agent_a: str, agent_b: str) -> bytes:
        """Return a 256-bit AES key unique to the ordered agent pair."""
        return self._derive(agent_a, agent_b, context=b"encrypt")

    def get_signing_key(self, agent_a: str, agent_b: str) -> bytes:
        """Return a 256-bit HMAC key unique to the ordered agent pair."""
        return self._derive(agent_a, agent_b, context=b"sign")

    # ── internal ──────────────────────────────────────────────────────────
    def _derive(self, agent_a: str, agent_b: str, context: bytes) -> bytes:
        """Derive a key using HKDF with agent pair + context label."""
        # Canonical ordering prevents key mismatch when A→B vs B→A
        pair = ":".join(sorted([agent_a, agent_b]))
        info = pair.encode("utf-8") + b":" + context
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=KEY_BYTES,
            salt=None,
            info=info,
        )
        return hkdf.derive(self._master)


# ---------------------------------------------------------------------------
# SecureMessageBus - encrypt / sign / validate / replay-protect
# ---------------------------------------------------------------------------
class SecureMessageBus:
    """
    Wraps inter-agent messages in a secure envelope:

    1. **Schema validation** - JSON Schema check before encryption (sender)
       and after decryption (receiver).
    2. **AES-256-GCM encryption** - Authenticated encryption of the payload.
    3. **HMAC-SHA256 signing** - Signature over the full envelope metadata +
       ciphertext, binding headers to payload.
    4. **Replay protection** - Nonce uniqueness via Redis SET + timestamp
       freshness check.
    """

    def __init__(
        self,
        key_manager: Optional[KeyManager] = None,
        redis_client: Optional[redis.Redis] = None,
        schema_registry: Optional[Dict[str, Any]] = None,
    ):
        self.key_manager = key_manager or KeyManager()
        self.redis_client = redis_client or self._default_redis()
        self.schema_registry = schema_registry or {}

    # ── public API ────────────────────────────────────────────────────────

    def wrap_message(
        self,
        from_agent: str,
        to_agent: str,
        tool_name: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a secure envelope around *payload*.

        Returns a dict ready for JSON serialisation and HTTP transport.
        """
        # 1. Schema validation (pre-encryption)
        self._validate_schema(tool_name, payload, direction="request")

        # 2. Derive keys
        enc_key = self.key_manager.get_encryption_key(from_agent, to_agent)
        sig_key = self.key_manager.get_signing_key(from_agent, to_agent)

        # 3. Generate nonce & timestamp
        nonce = secrets.token_bytes(NONCE_BYTES)
        timestamp = datetime.now(timezone.utc).isoformat()

        # 4. Encrypt payload with AES-256-GCM
        plaintext = json.dumps(payload, sort_keys=True).encode("utf-8")
        aesgcm = AESGCM(enc_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # 5. Build envelope (without signature)
        envelope: Dict[str, Any] = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "tool_name": tool_name,
            "timestamp": timestamp,
            "nonce": b64encode(nonce).decode("ascii"),
            "encrypted_payload": b64encode(ciphertext).decode("ascii"),
        }

        # 6. Sign envelope
        envelope["signature"] = self._sign(envelope, sig_key)

        logger.debug(
            "Wrapped secure message %s→%s tool=%s nonce=%s",
            from_agent, to_agent, tool_name, envelope["nonce"][:8],
        )
        return envelope

    def unwrap_message(
        self,
        envelope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verify and decrypt an incoming secure envelope.

        Returns the original plaintext payload dict.

        Raises:
            SecurityError: on signature mismatch, replay, or schema failure.
        """
        from_agent = envelope["from_agent"]
        to_agent = envelope["to_agent"]
        tool_name = envelope["tool_name"]

        # 1. Derive keys
        enc_key = self.key_manager.get_encryption_key(from_agent, to_agent)
        sig_key = self.key_manager.get_signing_key(from_agent, to_agent)

        # 2. Verify signature
        expected_sig = envelope.pop("signature", "")
        computed_sig = self._sign(envelope, sig_key)
        envelope["signature"] = expected_sig  # restore for caller

        if not hmac.compare_digest(expected_sig, computed_sig):
            raise SecurityError("HMAC signature verification failed")

        # 3. Replay protection - nonce + timestamp
        self._check_replay(envelope["nonce"], envelope["timestamp"])

        # 4. Decrypt payload
        nonce = b64decode(envelope["nonce"])
        ciphertext = b64decode(envelope["encrypted_payload"])
        aesgcm = AESGCM(enc_key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as exc:
            raise SecurityError(f"AES-GCM decryption failed: {exc}") from exc

        payload = json.loads(plaintext)

        # 5. Schema validation (post-decryption)
        self._validate_schema(tool_name, payload, direction="request")

        logger.debug(
            "Unwrapped secure message %s→%s tool=%s",
            from_agent, to_agent, tool_name,
        )
        return payload

    def wrap_response(
        self,
        from_agent: str,
        to_agent: str,
        tool_name: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Encrypt and sign a response envelope (agent → supervisor)."""
        self._validate_schema(tool_name, payload, direction="response")

        enc_key = self.key_manager.get_encryption_key(from_agent, to_agent)
        sig_key = self.key_manager.get_signing_key(from_agent, to_agent)

        nonce = secrets.token_bytes(NONCE_BYTES)
        timestamp = datetime.now(timezone.utc).isoformat()

        plaintext = json.dumps(payload, sort_keys=True).encode("utf-8")
        aesgcm = AESGCM(enc_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        envelope: Dict[str, Any] = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "tool_name": tool_name,
            "timestamp": timestamp,
            "nonce": b64encode(nonce).decode("ascii"),
            "encrypted_payload": b64encode(ciphertext).decode("ascii"),
            "is_response": True,
        }
        envelope["signature"] = self._sign(envelope, sig_key)
        return envelope

    def unwrap_response(
        self,
        envelope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify and decrypt a response envelope."""
        from_agent = envelope["from_agent"]
        to_agent = envelope["to_agent"]
        tool_name = envelope["tool_name"]

        enc_key = self.key_manager.get_encryption_key(from_agent, to_agent)
        sig_key = self.key_manager.get_signing_key(from_agent, to_agent)

        expected_sig = envelope.pop("signature", "")
        computed_sig = self._sign(envelope, sig_key)
        envelope["signature"] = expected_sig

        if not hmac.compare_digest(expected_sig, computed_sig):
            raise SecurityError("Response HMAC signature verification failed")

        self._check_replay(envelope["nonce"], envelope["timestamp"])

        nonce = b64decode(envelope["nonce"])
        ciphertext = b64decode(envelope["encrypted_payload"])
        aesgcm = AESGCM(enc_key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as exc:
            raise SecurityError(f"Response AES-GCM decryption failed: {exc}") from exc

        payload = json.loads(plaintext)
        self._validate_schema(tool_name, payload, direction="response")
        return payload

    # ── internal helpers ──────────────────────────────────────────────────

    def _sign(self, envelope: Dict[str, Any], key: bytes) -> str:
        """Compute HMAC-SHA256 over canonical JSON of the envelope."""
        # Exclude 'signature' field itself from the signed data
        data = {k: v for k, v in sorted(envelope.items()) if k != "signature"}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        digest = hmac.new(key, canonical.encode("utf-8"), HMAC_DIGEST).hexdigest()
        return digest

    def _check_replay(self, nonce_b64: str, timestamp_iso: str) -> None:
        """
        Reject replayed messages using nonce uniqueness + timestamp freshness.
        """
        # Timestamp freshness
        try:
            msg_time = datetime.fromisoformat(timestamp_iso)
            now = datetime.now(timezone.utc)
            age = abs((now - msg_time).total_seconds())
            if age > CLOCK_SKEW_SECONDS:
                raise SecurityError(
                    f"Message timestamp too old/future: {age:.1f}s "
                    f"(max {CLOCK_SKEW_SECONDS}s)"
                )
        except ValueError as exc:
            raise SecurityError(f"Invalid timestamp format: {exc}") from exc

        # Nonce uniqueness (Redis SET NX with TTL)
        nonce_key = f"mcp_nonce:{nonce_b64}"
        try:
            was_new = self.redis_client.set(
                nonce_key, "1", nx=True, ex=NONCE_TTL_SECONDS
            )
            if not was_new:
                raise SecurityError(f"Replay detected - nonce already seen: {nonce_b64[:16]}…")
        except redis.RedisError as exc:
            # Fail-open: log warning but allow message through
            logger.warning("Redis nonce check failed (fail-open): %s", exc)

    def _validate_schema(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        direction: str = "request",
    ) -> None:
        """
        Validate payload against the registered JSON Schema for the tool.
        Skips silently if no schema is registered (graceful degradation).
        """
        schema_key = f"{tool_name}:{direction}"
        schema = self.schema_registry.get(schema_key)
        if schema is None:
            return  # No schema registered - pass through

        try:
            jsonschema.validate(instance=payload, schema=schema)
        except ImportError:
            logger.warning("jsonschema not installed - skipping validation")
        except jsonschema.ValidationError as exc:
            raise SecurityError(
                f"Schema validation failed for {tool_name} ({direction}): "
                f"{exc.message}"
            ) from exc

    @staticmethod
    def _default_redis() -> redis.Redis:
        """Create a default Redis client from environment."""
        return redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0,
            decode_responses=True,
        )


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class SecurityError(Exception):
    """Raised when a message security check fails."""
    pass


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------
_secure_bus: Optional[SecureMessageBus] = None


def get_secure_message_bus(
    schema_registry: Optional[Dict[str, Any]] = None,
) -> SecureMessageBus:
    """Get or create the global SecureMessageBus singleton."""
    global _secure_bus
    if _secure_bus is None:
        _secure_bus = SecureMessageBus(schema_registry=schema_registry)
    return _secure_bus
