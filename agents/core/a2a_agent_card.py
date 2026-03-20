"""
A2A Agent Card Module
=====================
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

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A2A Protocol Data Structures
# ---------------------------------------------------------------------------

class A2AAuthScheme(str, Enum):
    """Supported authentication schemes for A2A communication."""
    BEARER = "bearer"
    HMAC = "hmac"
    MTLS = "mtls"
    NONE = "none"


@dataclass
class A2ASkill:
    """
    Describes a specific skill that an agent can perform.

    In CSIP, each skill maps to a worker capability within a team supervisor.
    For example, the member_services agent has skills: member_lookup,
    eligibility_check, coverage_lookup.
    """
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class A2ACapabilities:
    """
    Declares what the agent supports in terms of A2A protocol features.

    - streaming: Whether the agent supports SSE-based streaming responses
    - push_notifications: Whether the agent can send webhook callbacks
    - state_transition_history: Whether the agent returns full task state history
    """
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "stateTransitionHistory": self.state_transition_history,
        }


@dataclass
class A2AAuthentication:
    """Authentication configuration for the agent's A2A endpoint."""
    schemes: List[str] = field(default_factory=lambda: ["hmac"])

    def to_dict(self) -> Dict[str, Any]:
        return {"schemes": self.schemes}


@dataclass
class A2AAgentCard:
    """
    A2A Agent Card — the discovery document for a remote agent.

    Published at GET /.well-known/agent.json on each remote agent server.
    The CentralSupervisor fetches these cards to discover available agents,
    their skills, and endpoint URLs before delegating tasks via A2A.

    Attributes:
        name:           Human-readable agent name
        description:    What this agent does
        url:            Base URL of the agent's A2A endpoint
        version:        Agent Card schema version (default "1.0")
        protocol_version: A2A protocol version supported
        capabilities:   A2ACapabilities declaring supported features
        authentication: A2AAuthentication with supported auth schemes
        skills:         List of A2ASkill objects describing agent abilities
        default_input_modes:  MIME types accepted as input
        default_output_modes: MIME types produced as output
    """
    name: str
    description: str
    url: str
    version: str = "1.0"
    protocol_version: str = "0.2.2"
    capabilities: A2ACapabilities = field(default_factory=A2ACapabilities)
    authentication: A2AAuthentication = field(default_factory=A2AAuthentication)
    skills: List[A2ASkill] = field(default_factory=list)
    default_input_modes: List[str] = field(
        default_factory=lambda: ["application/json", "text/plain"]
    )
    default_output_modes: List[str] = field(
        default_factory=lambda: ["application/json", "text/plain"]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the A2A Agent Card JSON format."""
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "authentication": self.authentication.to_dict(),
            "skills": [s.to_dict() for s in self.skills],
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
        }


# ---------------------------------------------------------------------------
# Agent Card Registry (used by CentralSupervisor for discovery)
# ---------------------------------------------------------------------------

class AgentCardRegistry:
    """
    Registry that discovers and caches A2A Agent Cards from remote agents.

    The CentralSupervisor uses this registry to:
    1. Fetch Agent Cards from remote agent URLs on startup or on-demand
    2. Match incoming queries to agent skills for routing decisions
    3. Resolve the A2A task endpoint URL for delegating tasks

    Agent Cards are cached after first fetch and can be refreshed on demand.
    """

    def __init__(self):
        self._cards: Dict[str, A2AAgentCard] = {}
        self._card_urls: Dict[str, str] = {}  # agent_name -> well-known URL

    def register_card_url(self, agent_name: str, base_url: str) -> None:
        """
        Register a remote agent's base URL for Agent Card discovery.

        The Agent Card will be fetched from {base_url}/.well-known/agent.json
        when needed.

        Args:
            agent_name: Logical name of the agent (e.g., "member_services_team")
            base_url:   Base URL of the remote agent server
        """
        self._card_urls[agent_name] = f"{base_url}/.well-known/agent.json"
        logger.info(f"Registered Agent Card URL for {agent_name}: {base_url}")

    def register_card(self, agent_name: str, card: A2AAgentCard) -> None:
        """
        Directly register a pre-built Agent Card (used in testing or
        when the card is built locally rather than fetched over HTTP).

        Args:
            agent_name: Logical name of the agent
            card:       The A2AAgentCard instance
        """
        self._cards[agent_name] = card
        logger.info(f"Registered Agent Card for {agent_name}: {card.name}")

    async def fetch_card(self, agent_name: str) -> Optional[A2AAgentCard]:
        """
        Fetch an Agent Card from the remote agent's well-known URL.

        Uses httpx for async HTTP. Falls back to cached card on failure.

        Args:
            agent_name: The agent whose card to fetch

        Returns:
            A2AAgentCard if successful, None if both fetch and cache miss.
        """
        url = self._card_urls.get(agent_name)
        if not url:
            logger.warning(f"No Agent Card URL registered for {agent_name}")
            return self._cards.get(agent_name)

        try:
            import httpx

            async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

            card = A2AAgentCard(
                name=data["name"],
                description=data["description"],
                url=data["url"],
                version=data.get("version", "1.0"),
                protocol_version=data.get("protocolVersion", "0.2.2"),
                capabilities=A2ACapabilities(
                    streaming=data.get("capabilities", {}).get("streaming", False),
                    push_notifications=data.get("capabilities", {}).get("pushNotifications", False),
                    state_transition_history=data.get("capabilities", {}).get("stateTransitionHistory", True),
                ),
                authentication=A2AAuthentication(
                    schemes=data.get("authentication", {}).get("schemes", ["hmac"]),
                ),
                skills=[
                    A2ASkill(
                        id=s["id"],
                        name=s["name"],
                        description=s["description"],
                        tags=s.get("tags", []),
                        examples=s.get("examples", []),
                    )
                    for s in data.get("skills", [])
                ],
                default_input_modes=data.get("defaultInputModes", ["application/json"]),
                default_output_modes=data.get("defaultOutputModes", ["application/json"]),
            )

            self._cards[agent_name] = card
            logger.info(f"Fetched Agent Card for {agent_name} from {url}")
            return card

        except Exception as e:
            logger.warning(f"Failed to fetch Agent Card for {agent_name} from {url}: {e}")
            return self._cards.get(agent_name)

    def get_card(self, agent_name: str) -> Optional[A2AAgentCard]:
        """
        Get a cached Agent Card (synchronous).

        Args:
            agent_name: The agent whose card to retrieve

        Returns:
            Cached A2AAgentCard or None if not yet fetched.
        """
        return self._cards.get(agent_name)

    def get_all_cards(self) -> Dict[str, A2AAgentCard]:
        """Return all cached Agent Cards."""
        return dict(self._cards)

    def get_skills_summary(self) -> str:
        """
        Build a text summary of all registered agents and their skills.

        Used by the CentralSupervisor to include available agent capabilities
        in the routing prompt, enabling skill-based routing decisions.

        Returns:
            Formatted string listing each agent and its skills.
        """
        lines = []
        for agent_name, card in self._cards.items():
            lines.append(f"\nAgent: {card.name}")
            lines.append(f"  Description: {card.description}")
            lines.append(f"  Endpoint: {card.url}")
            lines.append(f"  Skills:")
            for skill in card.skills:
                lines.append(f"    - {skill.name}: {skill.description}")
                if skill.tags:
                    lines.append(f"      Tags: {', '.join(skill.tags)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_agent_card_registry: Optional[AgentCardRegistry] = None


def get_agent_card_registry() -> AgentCardRegistry:
    """Get or create the singleton AgentCardRegistry."""
    global _agent_card_registry
    if _agent_card_registry is None:
        _agent_card_registry = AgentCardRegistry()
    return _agent_card_registry
