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
                  (e.g., "https://api-gateway:8443/a2a/member").

    Returns:
        A2AAgentCard with member services skills.
    """
    return A2AAgentCard(
        name="member_services_team",
        description=(
            "Health insurance member services agent. Handles member lookup, "
            "eligibility verification, coverage/benefits inquiries, "
            "member policy lookup, and treatment history retrieval. "
            "Routes queries to specialized workers: member_lookup, "
            "check_eligibility, coverage_lookup, update_member_info, "
            "member_policy_lookup, treatment_history."
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
                id="check_eligibility",
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
            A2ASkill(
                id="update_member_info",
                name="Update Member Info",
                description=(
                    "Update a member's information. Requires the member ID, "
                    "the field to update (phone, email, address_street, address_city, "
                    "address_state, address_zip, enrollmentDate, status), the new value, and a reason for the change. "
                    "High-impact write operation — requires human approval before execution."
                ),
                tags=["member", "update", "contact", "address", "enrollment", "status", "write"],
                examples=[
                    "Update phone number for member M123456 to 555-9876 — member requested change",
                    "Change email for member M789012 to new@example.com — returned mail",
                    "Update enrollmentDate for member M345678 to 2025-10-01 — AEP re-enrollment",
                    "Change status for member M456789 to ACTIVE — eligibility confirmed",
                ],
            ),
            A2ASkill(
                id="member_policy_lookup",
                name="Member Policy Lookup",
                description=(
                    "Look up member information together with their associated insurance policy. "
                    "Returns member demographics and full policy details including policyId, "
                    "policyNumber, planName, planType, effectiveDate, expirationDate, status, "
                    "premium, deductible, and outOfPocketMax."
                ),
                tags=["member", "policy", "lookup", "plan", "insurance"],
                examples=[
                    "What policy does member M123456 have?",
                    "Show me the insurance plan for member M789012",
                    "Look up the policy details for member M345678",
                ],
            ),
            A2ASkill(
                id="treatment_history",
                name="Treatment History",
                description=(
                    "Retrieve treatment history for a member including physical therapy "
                    "sessions, medication trials, injections, imaging, and other "
                    "conservative treatments. Supports optional filtering by treatment "
                    "type or procedure code. Used by the PA recommendation decision "
                    "agent to verify conservative treatment requirements."
                ),
                tags=["member", "treatment", "history", "therapy", "conservative", "pa"],
                examples=[
                    "Get treatment history for member M123456",
                    "Show physical therapy sessions for member M789012",
                    "What conservative treatments has member M345678 had?",
                ],
            ),
        ],
    )
