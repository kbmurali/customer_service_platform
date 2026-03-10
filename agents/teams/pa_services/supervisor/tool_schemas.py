"""
JSON Schema definitions for inter-agent tool communication.

Each tool has a request schema (arguments sent to the remote MCP agent)
and a response schema (result returned by the remote MCP agent).
These schemas are validated by SecureMessageBus before encryption
(sender side) and after decryption (receiver side).
"""


# ============================================================================
# PA SERVICES TOOL SCHEMAS
# ============================================================================

PA_LOOKUP_REQUEST = {
    "type": "object",
    "properties": {
        "pa_id": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["pa_id", "user_id"],
    "additionalProperties": False,
}

PA_LOOKUP_RESPONSE = {
    "type": "object",
    "properties": {
        "paId": {"type": "string"},
        "paNumber": {"type": "string"},
        "procedureCode": {"type": "string"},
        "procedureDescription": {"type": ["string", "null"]},
        "requestDate": {"type": ["string", "null"]},
        "status": {"type": "string"},
        "urgency": {"type": ["string", "null"]},
        "approvalDate": {"type": ["string", "null"]},
        "expirationDate": {"type": ["string", "null"]},
        "denialReason": {"type": ["string", "null"]},
        "error": {"type": "string"},
    },
}

PA_STATUS_REQUEST = {
    "type": "object",
    "properties": {
        "pa_id": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["pa_id", "user_id"],
    "additionalProperties": False,
}

PA_STATUS_RESPONSE = {
    "type": "object",
    "properties": {
        "paId": {"type": "string"},
        "paNumber": {"type": "string"},
        "status": {"type": "string"},
        "urgency": {"type": ["string", "null"]},
        "requestDate": {"type": ["string", "null"]},
        "approvalDate": {"type": ["string", "null"]},
        "expirationDate": {"type": ["string", "null"]},
        "denialReason": {"type": ["string", "null"]},
        "error": {"type": "string"},
    },
}

PA_REQUIREMENTS_REQUEST = {
    "type": "object",
    "properties": {
        "procedure_code": {"type": "string"},
        "policy_type": {
            "type": "string",
            "enum": ["HMO", "PPO", "EPO", "POS"],
        },
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["procedure_code", "policy_type", "user_id"],
    "additionalProperties": False,
}

PA_REQUIREMENTS_RESPONSE = {
    "type": "object",
    "properties": {
        "requires_pa": {"type": "boolean"},
        "procedureCode": {"type": "string"},
        "policyType": {"type": "string"},
        "reason": {"type": ["string", "null"]},
        "history": {
            "type": ["array", "null"],
            "items": {"type": "object"},
        },
        "error": {"type": "string"},
    },
}

# ============================================================================
# SCHEMA REGISTRY – maps "tool_name:direction" → JSON Schema
# ============================================================================

def build_pa_schema_registry() -> dict:
    """
    Build the complete schema registry for PA services SecureMessageBus.

    Returns a dict keyed by ``"tool_name:request"`` or ``"tool_name:response"``.
    """
    return {
        # PA services
        "pa_lookup:request": PA_LOOKUP_REQUEST,
        "pa_lookup:response": PA_LOOKUP_RESPONSE,
        "pa_status:request": PA_STATUS_REQUEST,
        "pa_status:response": PA_STATUS_RESPONSE,
        "pa_requirements:request": PA_REQUIREMENTS_REQUEST,
        "pa_requirements:response": PA_REQUIREMENTS_RESPONSE,
    }
