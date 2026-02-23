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

CHECK_ELIGIBILITY_REQUEST = {
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

CHECK_ELIGIBILITY_RESPONSE = {
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

SEARCH_POLICY_INFO_REQUEST = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["query", "user_id"],
    "additionalProperties": False,
}

SEARCH_POLICY_INFO_RESPONSE = {
    "type": "object",
    "properties": {
        "results": {"type": "array"},
        "query": {"type": "string"},
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
# INVOKE-LEVEL SCHEMAS - full agent invocation envelope payloads
# Used by RemoteMCPNode ↔ remote agent server communication
# ============================================================================

MCP_INVOKE_REQUEST = {
    "type": "object",
    "required": ["query", "user_id", "user_role", "session_id"],
    "properties": {
        "query": {"type": "string"},
        "user_id": {"type": "string"},
        "user_role": {"type": "string"},
        "session_id": {"type": "string"},
        "plan": {"type": "object"},
    },
}

MCP_INVOKE_RESPONSE = {
    "type": "object",
    "required": ["messages"],
    "properties": {
        "messages": {"type": "array"},
        "tool_results": {"type": "object"},
        "execution_path": {"type": "array", "items": {"type": "string"}},
        "error": {"type": "string"},
        "error_count": {"type": "integer"},
        "error_history": {"type": "array"},
        "retry_count": {"type": "integer"},
    },
}

# ============================================================================
# SCHEMA REGISTRY - maps "tool_name:direction" → JSON Schema
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
        "check_eligibility:request": CHECK_ELIGIBILITY_REQUEST,
        "check_eligibility:response": CHECK_ELIGIBILITY_RESPONSE,
        "coverage_lookup:request": COVERAGE_LOOKUP_REQUEST,
        "coverage_lookup:response": COVERAGE_LOOKUP_RESPONSE,
        "search_policy_info:request": SEARCH_POLICY_INFO_REQUEST,
        "search_policy_info:response": SEARCH_POLICY_INFO_RESPONSE,
        "update_member_info:request": UPDATE_MEMBER_INFO_REQUEST,
        "update_member_info:response": UPDATE_MEMBER_INFO_RESPONSE,
        # Claim services
        "claim_lookup:request": CLAIM_LOOKUP_REQUEST,
        "claim_lookup:response": CLAIM_LOOKUP_RESPONSE,
        "claim_status:request": CLAIM_STATUS_REQUEST,
        "claim_status:response": CLAIM_STATUS_RESPONSE,
        "claim_payment_info:request": CLAIM_PAYMENT_INFO_REQUEST,
        "claim_payment_info:response": CLAIM_PAYMENT_INFO_RESPONSE,
        "update_claim_status:request": UPDATE_CLAIM_STATUS_REQUEST,
        "update_claim_status:response": UPDATE_CLAIM_STATUS_RESPONSE,
        # Remote agent invoke-level schemas
        "member_services_team_invoke:request": MCP_INVOKE_REQUEST,
        "member_services_team_invoke:response": MCP_INVOKE_RESPONSE,
        "claim_services_team_invoke:request": MCP_INVOKE_REQUEST,
        "claim_services_team_invoke:response": MCP_INVOKE_RESPONSE,
    }
