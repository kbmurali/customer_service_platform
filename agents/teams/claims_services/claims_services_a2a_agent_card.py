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
                  (e.g., "https://api-gateway:8443/a2a/claims").

    Returns:
        A2AAgentCard with claims services skills.
    """
    return A2AAgentCard(
        name="claims_services_team",
        description=(
            "Health insurance claims services agent. Handles claim lookup, "
            "claim status checks, payment information retrieval, and "
            "claim adjudication recommendations. "
            "Routes queries to specialized workers: claim_lookup, "
            "claim_status, claim_payment_info, update_claim_status, "
            "member_claims, claim_adjudication."
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
            A2ASkill(
                id="update_claim_status",
                name="Update Claim Status",
                description=(
                    "Update the processing status of a claim. Requires the claim ID, "
                    "the new target status (SUBMITTED, UNDER_REVIEW, APPROVED, DENIED), "
                    "and a reason for the change. High-impact write operation — "
                    "requires human approval before execution."
                ),
                tags=["claim", "update", "status", "write"],
                examples=[
                    "Approve claim 7799c06c-0883-4dca-b1f0-bded6d1027a5 because all documents are verified",
                    "Update claim abc123 status to DENIED — duplicate submission",
                ],
            ),
            A2ASkill(
                id="member_claims",
                name="Member Claims",
                description=(
                    "Retrieve all claims filed by a specific member. Requires a "
                    "member ID (e.g. M-12345). Returns a list of claims with "
                    "claim number, service date, status, amounts, and processing date. "
                    "Optionally filter by claim status."
                ),
                tags=["claim", "member", "lookup", "list"],
                examples=[
                    "What claims does member M-12345 have?",
                    "Show me all claims for member M-67890",
                    "List pending claims for member M-11111",
                ],
            ),
            A2ASkill(
                id="claim_adjudication",
                name="Claim Adjudication",
                description=(
                    "Evaluate whether a claim should be approved, denied, or sent "
                    "for review based on coverage rules. Decision agent — does not "
                    "call MCP tools. Requires evidence from prior steps: claim_lookup "
                    "(claim details), check_eligibility (member eligibility on service "
                    "date), and provider_network_check (in-network status). Produces "
                    "a structured recommendation with justification persisted in the "
                    "Context Graph."
                ),
                tags=["claim", "adjudication", "decision", "approve", "deny", "review"],
                examples=[
                    "Is claim CLM-123456 valid?",
                    "Should this claim be approved based on the evidence gathered?",
                    "Evaluate claim eligibility and coverage for adjudication",
                ],
            ),
        ],
    )
