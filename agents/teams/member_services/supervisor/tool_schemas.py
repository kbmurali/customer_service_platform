"""
JSON Schema definitions for inter-agent tool communication.

Each tool has a request schema (arguments sent to the remote MCP agent)
and a response schema (result returned by the remote MCP agent).
These schemas are validated by SecureMessageBus before encryption
(sender side) and after decryption (receiver side).
"""

# ============================================================================
# MEMBER SERVICES TOOL SCHEMAS
# ============================================================================

MEMBER_LOOKUP_REQUEST = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string", "pattern": "^M\\d{4,10}$"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["member_id", "user_id"],
    "additionalProperties": False,
}

MEMBER_LOOKUP_RESPONSE = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string"},
        "name": {"type": "string"},
        "date_of_birth": {"type": "string"},
        "plan_id": {"type": "string"},
        "plan_name": {"type": "string"},
        "status": {"type": "string"},
        "effective_date": {"type": "string"},
        "termination_date": {"type": ["string", "null"]},
        "error": {"type": "string"},
    },
}

ELIGIBILITY_CHECK_REQUEST = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string", "pattern": "^M\\d{4,10}$"},
        "service_date": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["member_id", "user_id"],
    "additionalProperties": False,
}

ELIGIBILITY_CHECK_RESPONSE = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string"},
        "eligible": {"type": "boolean"},
        "plan_status": {"type": "string"},
        "coverage_start": {"type": "string"},
        "coverage_end": {"type": ["string", "null"]},
        "copay": {"type": ["string", "number"]},
        "deductible_remaining": {"type": ["string", "number"]},
        "error": {"type": "string"},
    },
}

COVERAGE_LOOKUP_REQUEST = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string", "pattern": "^M\\d{4,10}$"},
        "procedure_code": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["member_id", "procedure_code", "user_id"],
    "additionalProperties": False,
}

COVERAGE_LOOKUP_RESPONSE = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string"},
        "procedure_code": {"type": "string"},
        "covered": {"type": "boolean"},
        "coverage_percentage": {"type": ["number", "string"]},
        "prior_auth_required": {"type": "boolean"},
        "in_network_cost": {"type": ["string", "number"]},
        "out_of_network_cost": {"type": ["string", "number"]},
        "error": {"type": "string"},
    },
}

UPDATE_MEMBER_INFO_REQUEST = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string", "pattern": "^M\\d{4,10}$"},
        "field": {"type": "string", "minLength": 1},
        "new_value": {"type": "string"},
        "reason": {"type": "string"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["member_id", "field", "new_value", "user_id"],
    "additionalProperties": False,
}

UPDATE_MEMBER_INFO_RESPONSE = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string"},
        "field": {"type": "string"},
        "old_value": {"type": ["string", "null"]},
        "new_value": {"type": "string"},
        "updated": {"type": "boolean"},
        "error": {"type": "string"},
    },
}

MEMBER_POLICY_LOOKUP_REQUEST = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string", "pattern": "^M\\d{4,10}$"},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["member_id", "user_id"],
    "additionalProperties": False,
}

MEMBER_POLICY_LOOKUP_RESPONSE = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string"},
        "first_name": {"type": "string"},
        "last_name": {"type": "string"},
        "date_of_birth": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "status": {"type": "string"},
        "enrollment_date": {"type": "string"},
        "policy": {
            "type": ["object", "null"],
            "properties": {
                "policy_id": {"type": "string"},
                "policy_number": {"type": "string"},
                "policy_type": {"type": "string"},
                "plan_name": {"type": "string"},
                "plan_type": {"type": "string"},
                "effective_date": {"type": "string"},
                "expiration_date": {"type": ["string", "null"]},
                "status": {"type": "string"},
                "premium": {"type": ["number", "string", "null"]},
                "deductible": {"type": ["number", "string", "null"]},
                "out_of_pocket_max": {"type": ["number", "string", "null"]},
            },
        },
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
        # Member services
        "member_lookup:request": MEMBER_LOOKUP_REQUEST,
        "member_lookup:response": MEMBER_LOOKUP_RESPONSE,
        "check_eligibility:request": ELIGIBILITY_CHECK_REQUEST,
        "check_eligibility:response": ELIGIBILITY_CHECK_RESPONSE,
        "coverage_lookup:request": COVERAGE_LOOKUP_REQUEST,
        "coverage_lookup:response": COVERAGE_LOOKUP_RESPONSE,
        "update_member_info:request": UPDATE_MEMBER_INFO_REQUEST,
        "update_member_info:response": UPDATE_MEMBER_INFO_RESPONSE,
        "member_policy_lookup:request": MEMBER_POLICY_LOOKUP_REQUEST,
        "member_policy_lookup:response": MEMBER_POLICY_LOOKUP_RESPONSE,
    }
