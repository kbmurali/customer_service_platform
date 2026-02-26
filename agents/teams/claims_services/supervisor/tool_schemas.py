"""
JSON Schema definitions for inter-agent tool communication.

Each tool has a request schema (arguments sent to the remote MCP agent)
and a response schema (result returned by the remote MCP agent).
These schemas are validated by SecureMessageBus before encryption
(sender side) and after decryption (receiver side).
"""


# ============================================================================
# CLAIM SERVICES TOOL SCHEMAS
# ============================================================================

CLAIM_LOOKUP_REQUEST = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["claim_id", "user_id"],
    "additionalProperties": False,
}

CLAIM_LOOKUP_RESPONSE = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "member_id": {"type": "string"},
        "provider": {"type": "string"},
        "service_date": {"type": "string"},
        "diagnosis_code": {"type": "string"},
        "procedure_code": {"type": "string"},
        "billed_amount": {"type": ["string", "number"]},
        "allowed_amount": {"type": ["string", "number"]},
        "status": {"type": "string"},
        "error": {"type": "string"},
    },
}

CLAIM_STATUS_REQUEST = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["claim_id", "user_id"],
    "additionalProperties": False,
}

CLAIM_STATUS_RESPONSE = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "status": {"type": "string"},
        "last_updated": {"type": "string"},
        "next_action": {"type": ["string", "null"]},
        "estimated_completion": {"type": ["string", "null"]},
        "error": {"type": "string"},
    },
}

CLAIM_PAYMENT_INFO_REQUEST = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["claim_id", "user_id"],
    "additionalProperties": False,
}

CLAIM_PAYMENT_INFO_RESPONSE = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "payment_status": {"type": "string"},
        "paid_amount": {"type": ["string", "number"]},
        "payment_date": {"type": ["string", "null"]},
        "check_number": {"type": ["string", "null"]},
        "eob_available": {"type": "boolean"},
        "error": {"type": "string"},
    },
}

UPDATE_CLAIM_STATUS_REQUEST = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "new_status": {"type": "string"},
        "reason": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["claim_id", "new_status", "user_id"],
    "additionalProperties": False,
}

UPDATE_CLAIM_STATUS_RESPONSE = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "old_status": {"type": ["string", "null"]},
        "new_status": {"type": "string"},
        "updated": {"type": "boolean"},
        "error": {"type": "string"},
    },
}

# ============================================================================
# SCHEMA REGISTRY – maps "tool_name:direction" → JSON Schema
# ============================================================================

def build_schema_registry() -> dict:
    """
    Build the complete schema registry for SecureMessageBus.

    Returns a dict keyed by ``"tool_name:request"`` or ``"tool_name:response"``.
    """
    return {
        # Claim services
        "claim_lookup:request": CLAIM_LOOKUP_REQUEST,
        "claim_lookup:response": CLAIM_LOOKUP_RESPONSE,
        "claim_status:request": CLAIM_STATUS_REQUEST,
        "claim_status:response": CLAIM_STATUS_RESPONSE,
        "claim_payment_info:request": CLAIM_PAYMENT_INFO_REQUEST,
        "claim_payment_info:response": CLAIM_PAYMENT_INFO_RESPONSE,
        "update_claim_status:request": UPDATE_CLAIM_STATUS_REQUEST,
        "update_claim_status:response": UPDATE_CLAIM_STATUS_RESPONSE,
    }