"""
Claims Services A2A Agent Card Module
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
def build_claims_services_agent_card(base_url: str) -> A2AAgentCard:
    """
    Build the Agent Card for the Claims Services team supervisor.

    This card is served at GET /.well-known/agent.json on the
    claims_services_server deployment. The CentralSupervisor fetches
    it to discover skills and the A2A task endpoint.

    Args:
        base_url: The base URL where the claims services agent is deployed
                  (e.g., "https://api-gateway:8443/claims-services").

    Returns:
        A2AAgentCard with claims services skills.
    """
    return A2AAgentCard(
        name="claims_services_team",
        description=(
            "Health insurance claims services agent. Handles claim lookup, "
            "claim status checks, and payment information retrieval. "
            "Routes queries to specialized workers: claim_lookup, "
            "claim_status, claim_payment_info."
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
                id="claim_lookup",
                name="Claim Lookup",
                description=(
                    "Look up full claim details by claim ID, including service date, "
                    "submission date, status, amounts, member, provider, and policy context."
                ),
                tags=["claim", "lookup", "details"],
                examples=[
                    "Look up claim 7799c06c-0883-4dca-b1f0-bded6d1027a5",
                    "Get full details for claim ID abc123",
                ],
            ),
            A2ASkill(
                id="claim_status",
                name="Claim Status",
                description=(
                    "Check the current processing status of a claim by claim number "
                    "(e.g. CLM-123456). Returns status, processing date, and denial "
                    "reason if applicable."
                ),
                tags=["claim", "status", "processing"],
                examples=[
                    "What is the status of claim CLM-2024-0001?",
                    "Check if claim CLM-123456 has been approved",
                ],
            ),
            A2ASkill(
                id="claim_payment_info",
                name="Claim Payment Info",
                description=(
                    "Retrieve payment information for a claim by claim ID, including "
                    "total billed amount, paid amount, processing date, and denial "
                    "reason if the claim was denied."
                ),
                tags=["claim", "payment", "amount", "reimbursement"],
                examples=[
                    "How much was paid on claim 7799c06c-0883-4dca-b1f0-bded6d1027a5?",
                    "Show payment details for claim ID abc123",
                ],
            ),
        ],
    )
