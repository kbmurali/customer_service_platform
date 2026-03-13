"""
JSON Schema definitions for inter-agent tool communication.

Each tool has a request schema (arguments sent to the remote MCP agent)
and a response schema (result returned by the remote MCP agent).
These schemas are validated by SecureMessageBus before encryption
(sender side) and after decryption (receiver side).
"""


# ============================================================================
# SEARCH SERVICES TOOL SCHEMAS
# ============================================================================

SEARCH_KNOWLEDGE_BASE_REQUEST = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "source": {
            "type": "string",
            "enum": ["faqs", "guidelines", "regulations", "all"],
        },
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["query", "source", "user_id"],
    "additionalProperties": False,
}

SEARCH_KNOWLEDGE_BASE_RESPONSE = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "faqs": {
            "type": ["array", "null"],
            "items": {"type": "object"},
        },
        "clinical_guidelines": {
            "type": ["array", "null"],
            "items": {"type": "object"},
        },
        "regulations": {
            "type": ["array", "null"],
            "items": {"type": "object"},
        },
        "error": {"type": "string"},
    },
}

SEARCH_MEDICAL_CODES_REQUEST = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "code_type": {
            "type": "string",
            "enum": ["procedure", "diagnosis", "both"],
        },
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["query", "code_type", "user_id"],
    "additionalProperties": False,
}

SEARCH_MEDICAL_CODES_RESPONSE = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "procedures": {
            "type": ["array", "null"],
            "items": {"type": "object"},
        },
        "diagnoses": {
            "type": ["array", "null"],
            "items": {"type": "object"},
        },
        "error": {"type": "string"},
    },
}

SEARCH_POLICY_INFO_REQUEST = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "plan_type": {
            "type": "string",
            "enum": ["HMO", "PPO", "EPO", "POS"],
        },
        "user_id": {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["query", "plan_type", "user_id"],
    "additionalProperties": False,
}

SEARCH_POLICY_INFO_RESPONSE = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "count": {"type": "integer"},
        "results": {
            "type": ["array", "null"],
            "items": {"type": "object"},
        },
        "error": {"type": "string"},
    },
}

# ============================================================================
# SCHEMA REGISTRY – maps "tool_name:direction" → JSON Schema
# ============================================================================

def build_search_schema_registry() -> dict:
    """
    Build the complete schema registry for Search services SecureMessageBus.

    Returns a dict keyed by ``"tool_name:request"`` or ``"tool_name:response"``.
    """
    return {
        # Search services
        "search_knowledge_base:request":  SEARCH_KNOWLEDGE_BASE_REQUEST,
        "search_knowledge_base:response": SEARCH_KNOWLEDGE_BASE_RESPONSE,
        "search_medical_codes:request":   SEARCH_MEDICAL_CODES_REQUEST,
        "search_medical_codes:response":  SEARCH_MEDICAL_CODES_RESPONSE,
        "search_policy_info:request":     SEARCH_POLICY_INFO_REQUEST,
        "search_policy_info:response":    SEARCH_POLICY_INFO_RESPONSE,
    }
