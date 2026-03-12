"""
JSON Schema definitions for inter-agent tool communication.

Each tool has a request schema (arguments sent to the remote MCP agent)
and a response schema (result returned by the remote MCP agent).
These schemas are validated by SecureMessageBus before encryption
(sender side) and after decryption (receiver side).
"""


# ============================================================================
# PROVIDER SERVICES TOOL SCHEMAS
# ============================================================================

PROVIDER_LOOKUP_REQUEST = {
    "type": "object",
    "properties": {
        "provider_id": {"type": "string"},
        "user_id":     {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["provider_id", "user_id"],
    "additionalProperties": False,
}

PROVIDER_LOOKUP_RESPONSE = {
    "type": "object",
    "properties": {
        # Core identity
        "providerId":       {"type": "string"},
        "npi":              {"type": ["string", "null"]},
        "providerType":     {"type": "string"},          # "INDIVIDUAL" | "ORGANIZATION"
        "specialty":        {"type": ["string", "null"]},
        "phone":            {"type": ["string", "null"]},
        # Address
        "street":           {"type": ["string", "null"]},
        "city":             {"type": ["string", "null"]},
        "state":            {"type": ["string", "null"]},
        "zipCode":          {"type": ["string", "null"]},
        # Name — mutually exclusive by providerType
        "organizationName": {"type": ["string", "null"]},  # ORGANIZATION only
        "firstName":        {"type": ["string", "null"]},  # INDIVIDUAL only
        "lastName":         {"type": ["string", "null"]},  # INDIVIDUAL only
        # Error
        "error":            {"type": "string"},
    },
}

PROVIDER_NETWORK_CHECK_REQUEST = {
    "type": "object",
    "properties": {
        "provider_id": {"type": "string"},
        "policy_id":   {"type": "string"},
        "user_id":     {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["provider_id", "policy_id", "user_id"],
    "additionalProperties": False,
}

PROVIDER_NETWORK_CHECK_RESPONSE = {
    "type": "object",
    "properties": {
        # has_history=True path
        "has_history": {"type": "boolean"},
        "provider":    {"type": ["object", "null"]},
        "policy":      {"type": ["object", "null"]},
        "claimCount":  {"type": ["integer", "null"]},
        # has_history=False path
        "reason":      {"type": ["string", "null"]},
        # Error
        "error":       {"type": "string"},
    },
}

PROVIDER_SEARCH_BY_SPECIALTY_REQUEST = {
    "type": "object",
    "properties": {
        "specialty": {"type": "string"},
        "zip_code":  {"type": "string"},
        "user_id":   {"type": "string", "minLength": 1},
        "user_role": {
            "type": "string",
            "enum": ["CSR_TIER1", "CSR_TIER2", "CSR_SUPERVISOR", "CSR_READONLY"],
        },
    },
    "required": ["specialty", "zip_code", "user_id"],
    "additionalProperties": False,
}

PROVIDER_SEARCH_BY_SPECIALTY_RESPONSE = {
    "type": "object",
    "properties": {
        "count": {"type": "integer"},
        "providers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "providerId":       {"type": "string"},
                    "npi":              {"type": ["string", "null"]},
                    "providerType":     {"type": "string"},
                    "specialty":        {"type": ["string", "null"]},
                    "phone":            {"type": ["string", "null"]},
                    "street":           {"type": ["string", "null"]},
                    "city":             {"type": ["string", "null"]},
                    "state":            {"type": ["string", "null"]},
                    "zipCode":          {"type": ["string", "null"]},
                    "organizationName": {"type": ["string", "null"]},
                    "firstName":        {"type": ["string", "null"]},
                    "lastName":         {"type": ["string", "null"]},
                },
            },
        },
        # Error
        "error": {"type": "string"},
    },
}


# ============================================================================
# SCHEMA REGISTRY – maps "tool_name:direction" → JSON Schema
# ============================================================================

def build_provider_schema_registry() -> dict:
    """
    Build the complete schema registry for Provider services SecureMessageBus.

    Returns a dict keyed by ``"tool_name:request"`` or ``"tool_name:response"``.
    """
    return {
        # Provider services
        "provider_lookup:request":                PROVIDER_LOOKUP_REQUEST,
        "provider_lookup:response":               PROVIDER_LOOKUP_RESPONSE,
        "provider_network_check:request":         PROVIDER_NETWORK_CHECK_REQUEST,
        "provider_network_check:response":        PROVIDER_NETWORK_CHECK_RESPONSE,
        "provider_search_by_specialty:request":   PROVIDER_SEARCH_BY_SPECIALTY_REQUEST,
        "provider_search_by_specialty:response":  PROVIDER_SEARCH_BY_SPECIALTY_RESPONSE,
    }
