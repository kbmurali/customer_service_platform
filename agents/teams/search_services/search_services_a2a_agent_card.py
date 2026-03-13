"""
Search Services A2A Agent Card Module
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
def build_search_services_agent_card(base_url: str) -> A2AAgentCard:
    """
    Build the Agent Card for the Search Services team supervisor.

    This card is served at GET /.well-known/agent.json on the
    search_services_server deployment. The CentralSupervisor fetches
    it to discover skills and the A2A task endpoint.

    Args:
        base_url: The base URL where the search services agent is deployed
                  (e.g., "https://api-gateway:8443/a2a/search").

    Returns:
        A2AAgentCard with search services skills.
    """
    return A2AAgentCard(
        name="search_services_team",
        description=(
            "Health insurance search services agent. Handles semantic search over "
            "knowledge base content, medical codes, and policy documents. "
            "Routes queries to specialized workers: search_knowledge_base, "
            "search_medical_codes, search_policy_info."
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
                id="search_knowledge_base",
                name="Search Knowledge Base",
                description=(
                    "Semantic search over FAQs, clinical guidelines, and regulations. "
                    "Accepts a natural-language query and an optional source filter: "
                    "'faqs', 'guidelines', 'regulations', or 'all'."
                ),
                tags=["search", "knowledge_base", "faqs", "guidelines", "regulations"],
                examples=[
                    "What are the referral requirements for specialist visits?",
                    "Search FAQs about emergency care coverage",
                    "Find clinical guidelines for diabetes management",
                ],
            ),
            A2ASkill(
                id="search_medical_codes",
                name="Search Medical Codes",
                description=(
                    "Semantic search over CPT procedure codes and ICD-10 diagnosis codes. "
                    "Accepts a natural-language description and an optional code type filter: "
                    "'procedure', 'diagnosis', or 'both'."
                ),
                tags=["search", "medical_codes", "CPT", "ICD-10", "procedure", "diagnosis"],
                examples=[
                    "Find the CPT code for knee replacement surgery",
                    "What is the ICD-10 code for type 2 diabetes?",
                    "Search for procedure and diagnosis codes related to colonoscopy",
                ],
            ),
            A2ASkill(
                id="search_policy_info",
                name="Search Policy Info",
                description=(
                    "Semantic search over policy documents for plan-specific details "
                    "including premiums, deductibles, and out-of-pocket maximums. "
                    "Requires a natural-language query and a plan type: "
                    "HMO, PPO, EPO, or POS."
                ),
                tags=["search", "policy", "plan", "deductible", "premium", "HMO", "PPO", "EPO", "POS"],
                examples=[
                    "What is the deductible for my PPO plan?",
                    "Find out-of-pocket maximum for HMO coverage",
                    "What does my EPO plan say about specialist referrals?",
                ],
            ),
        ],
    )
