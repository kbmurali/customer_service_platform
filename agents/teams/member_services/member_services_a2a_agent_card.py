"""
Member Services A2A Agent Card Module
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
def build_member_services_agent_card(base_url: str) -> A2AAgentCard:
    """
    Build the Agent Card for the Member Services team supervisor.

    This card is served at GET /.well-known/agent.json on the
    member_services_server deployment. The CentralSupervisor fetches
    it to discover skills and the A2A task endpoint.

    Args:
        base_url: The base URL where the member services agent is deployed
                  (e.g., "https://mcp-member:8443").

    Returns:
        A2AAgentCard with member services skills.
    """
    return A2AAgentCard(
        name="member_services_team",
        description=(
            "Health insurance member services agent. Handles member lookup, "
            "eligibility verification, and coverage/benefits inquiries. "
            "Routes queries to specialized workers: member_lookup, "
            "eligibility_check, coverage_lookup."
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
                id="member_lookup",
                name="Member Lookup",
                description="Look up member information by member ID including demographics and contact details.",
                tags=["member", "lookup", "demographics"],
                examples=[
                    "Look up member M123456",
                    "Find member information for ID M789012",
                ],
            ),
            A2ASkill(
                id="eligibility_check",
                name="Eligibility Check",
                description="Verify member eligibility and active coverage status for a given service date.",
                tags=["eligibility", "verification", "coverage"],
                examples=[
                    "Check eligibility for member M123456",
                    "Is member M789012 eligible for services on 2026-01-15?",
                ],
            ),
            A2ASkill(
                id="coverage_lookup",
                name="Coverage Lookup",
                description="Retrieve detailed coverage information including deductibles, copays, and benefits.",
                tags=["coverage", "benefits", "deductible", "copay"],
                examples=[
                    "What is the coverage for member M123456?",
                    "Show deductible and copay details for M789012",
                ],
            ),
        ],
    )
