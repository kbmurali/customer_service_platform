"""
Provider Services A2A Agent Card Module
===========================================
Implements Google's Agent-to-Agent (A2A) protocol Agent Cards.

Each team supervisor publishes an Agent Card at /.well-known/agent.json
that describes its identity, capabilities, skills, and endpoint URL.
This enables dynamic discovery and skill-based routing by the
CentralSupervisor instead of hardcoded RemoteMCPNode environment variables.

A2A Spec Reference:
    - Agent Cards are JSON documents served at /.well-known/agent.json
    - They describe agent capabilities, supported content types, and
      authentication requirements
    - Clients discover agents by fetching the Agent Card before sending tasks

A2A is used for inter-supervisor (agent-to-agent) communication.
"""

from agents.core.a2a_agent_card import A2AAgentCard, A2AAuthentication, A2ASkill, A2ACapabilities

# ---------------------------------------------------------------------------
# Pre-built Agent Cards for CSIP Team Supervisors
# ---------------------------------------------------------------------------
def build_provider_services_agent_card(base_url: str) -> A2AAgentCard:
    """
    Build the Agent Card for the Provider Services team supervisor.

    This card is served at GET /.well-known/agent.json on the
    provider_services_server deployment. The CentralSupervisor fetches
    it to discover skills and the A2A task endpoint.

    Args:
        base_url: The base URL where the provider services agent is deployed
                  (e.g., "https://api-gateway:8443/a2a/provider").

    Returns:
        A2AAgentCard with provider services skills.
    """
    return A2AAgentCard(
        name="provider_services_team",
        description=(
            "Health insurance provider services agent. Handles provider lookup, "
            "network status verification, and provider search by specialty. "
            "Routes queries to specialized workers: provider_lookup, "
            "provider_network_check, provider_search_by_specialty."
        ),
        url=f"{base_url}/a2a",
        capabilities=A2ACapabilities(
            streaming=False,
            push_notifications=False,
            state_transition_history=True,
        ),
        authentication=A2AAuthentication(schemes=["hmac"]),
        skills=[
            A2ASkill(
                id="provider_lookup",
                name="Provider Lookup",
                description=(
                    "Look up full provider details by provider ID, including NPI, "
                    "specialty, contact information, address, and provider type "
                    "(INDIVIDUAL or ORGANIZATION)."
                ),
                tags=["provider", "lookup", "npi", "details"],
                examples=[
                    "Look up provider P123456",
                    "Find provider information for ID abc-123-def",
                ],
            ),
            A2ASkill(
                id="provider_network_check",
                name="Provider Network Check",
                description=(
                    "Check whether a provider has serviced claims under a specific policy. "
                    "Network status is inferred from claim history — a provider is considered "
                    "in-network if they have serviced claims under the given policy."
                ),
                tags=["provider", "network", "in-network", "policy", "claims"],
                examples=[
                    "Is provider P123456 in-network for policy POL-789?",
                    "Check network status of provider abc-123 under policy xyz-456",
                ],
            ),
            A2ASkill(
                id="provider_search_by_specialty",
                name="Provider Search by Specialty",
                description=(
                    "Search for providers by medical specialty and ZIP code location. "
                    "Returns up to 10 matching providers with full contact and address details."
                ),
                tags=["provider", "search", "specialty", "location", "zip"],
                examples=[
                    "Find Cardiology providers near ZIP 60601",
                    "Search for Orthopedics specialists in ZIP 90210",
                ],
            ),
        ],
    )
