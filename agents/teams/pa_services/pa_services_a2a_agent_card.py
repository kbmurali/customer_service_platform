"""
PA Services A2A Agent Card Module
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
def build_pa_services_agent_card(base_url: str) -> A2AAgentCard:
    """
    Build the Agent Card for the PA Services team supervisor.
    This card is served at GET /.well-known/agent.json on the
    pa_services_server deployment. The CentralSupervisor fetches
    it to discover skills and the A2A task endpoint.
    Args:
        base_url: The base URL where the PA services agent is deployed
                  (e.g., "https://api-gateway:8443/a2a/pa").
    Returns:
        A2AAgentCard with PA services skills.
    """
    return A2AAgentCard(
        name="pa_services_team",
        description=(
            "Health insurance prior authorization services agent. Handles PA lookup, "
            "PA status checks, PA requirements lookup, PA approval/denial, "
            "member PA history, and PA recommendation decisions. "
            "Routes queries to specialized workers: pa_lookup, "
            "pa_status, pa_requirements, approve_prior_auth, deny_prior_auth, "
            "member_prior_authorizations, pa_recommendation."
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
                id="pa_lookup",
                name="PA Lookup",
                description=(
                    "Look up full prior authorization details by PA ID, including "
                    "procedure code, procedure description, request date, status, "
                    "urgency, approval date, expiration date, and denial reason."
                ),
                tags=["prior_authorization", "pa", "lookup", "details"],
                examples=[
                    "Look up prior authorization 7799c06c-0883-4dca-b1f0-bded6d1027a5",
                    "Get full details for PA ID abc123",
                ],
            ),
            A2ASkill(
                id="pa_status",
                name="PA Status",
                description=(
                    "Check the current status of a prior authorization by PA ID. "
                    "Returns status, urgency, request date, approval date, "
                    "expiration date, and denial reason if applicable."
                ),
                tags=["prior_authorization", "pa", "status"],
                examples=[
                    "What is the status of PA ID abc123?",
                    "Has prior authorization 7799c06c-0883-4dca-b1f0-bded6d1027a5 been approved?",
                ],
            ),
            A2ASkill(
                id="pa_requirements",
                name="PA Requirements",
                description=(
                    "Determine whether a procedure requires prior authorization "
                    "under a given policy type. Requires a procedure code (CPT code) "
                    "and policy type (HMO, PPO, EPO, or POS)."
                ),
                tags=["prior_authorization", "pa", "requirements", "procedure", "policy"],
                examples=[
                    "Does procedure code 27447 require PA under a PPO plan?",
                    "Check if CPT 43239 needs prior authorization for an HMO policy",
                ],
            ),
            A2ASkill(
                id="member_prior_authorizations",
                name="Member Prior Authorizations",
                description=(
                    "Retrieve all prior authorizations for a specific member. "
                    "Requires a member ID (e.g. M-12345). Returns a list of PAs "
                    "with PA number, procedure code, status, dates, and requesting "
                    "provider. Optionally filter by PA status."
                ),
                tags=["prior_authorization", "pa", "member", "lookup", "list"],
                examples=[
                    "What prior authorizations does member M-12345 have?",
                    "Show me all PAs for member M-67890",
                    "List pending prior authorizations for member M-11111",
                ],
            ),
            A2ASkill(
                id="approve_prior_auth",
                name="Approve Prior Authorization",
                description=(
                    "Approve a prior authorization request. Requires the PA ID and "
                    "a reason for approval. High-impact write operation — requires "
                    "human approval before execution."
                ),
                tags=["prior_authorization", "pa", "approve", "write"],
                examples=[
                    "Approve PA abc123 — clinical criteria met",
                    "Approve prior authorization 7799c06c — peer-to-peer review completed",
                ],
            ),
            A2ASkill(
                id="deny_prior_auth",
                name="Deny Prior Authorization",
                description=(
                    "Deny a prior authorization request. Requires the PA ID and "
                    "a reason for denial. High-impact write operation — requires "
                    "human approval before execution."
                ),
                tags=["prior_authorization", "pa", "deny", "write"],
                examples=[
                    "Deny PA abc123 — does not meet medical necessity criteria",
                    "Deny prior authorization 7799c06c — alternative treatment available",
                ],
            ),
            A2ASkill(
                id="pa_recommendation",
                name="PA Recommendation",
                description=(
                    "Evaluate whether a prior authorization should be approved, "
                    "denied, or if additional information is needed, based on "
                    "clinical criteria. Decision agent — does not call MCP tools. "
                    "Requires evidence from prior steps: pa_lookup (PA details), "
                    "pa_requirements (whether procedure requires PA), "
                    "search_knowledge_base (clinical guidelines with approval criteria), "
                    "and treatment_history from member_services_team (conservative "
                    "treatment records). Produces a structured recommendation with "
                    "clinical justification persisted in the Context Graph."
                ),
                tags=["prior_authorization", "pa", "recommendation", "decision", "clinical", "approve", "deny"],
                examples=[
                    "Should this prior authorization be approved based on the clinical evidence?",
                    "Evaluate PA for rotator cuff repair — has conservative treatment been met?",
                    "Recommend approval or denial for PA on lumbar spinal fusion",
                ],
            ),
        ],
    )
