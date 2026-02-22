import logging
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from observability.langfuse_integration import get_langfuse_tracer
        
from agents.teams.member_services.member_lookup_worker import MemberLookupWorker
from agents.teams.member_services.check_eligibility_worker import EligibilityCheckWorker
from agents.teams.member_services.coverage_lookup_worker import CoverageLookupWorker

from databases.chroma_vector_data_access import get_chroma_data_access

from llm_providers.llm_provider_factory import LLMProviderFactory, get_factory, ChatModel

from agents.security import RBACService, AuditLogger

from security.approval_workflow import get_approval_workflow, CircuitBreakerError
from security.presidio_memory_security import get_presidio_security
from agents.core.context_graph import get_context_graph_manager
from agents.core.state import SupervisorState

from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)
settings = get_settings()

class MemberServicesSupervisor:
    """
    LangGraph-based supervisor for Member Services team.
    Routes queries to appropriate workers using LangGraph state machine.
    """
    
    def __init__(self):
        self.name = "member_services_supervisor"
        self.workers = {
            "member_lookup": MemberLookupWorker(),
            "check_eligibility": EligibilityCheckWorker(),
            "coverage_lookup": CoverageLookupWorker()
        }
        
        self.rbac = RBACService()
        self.audit = AuditLogger()
        self.presidio = get_presidio_security()
        self.cg_manager = get_context_graph_manager()
        
        llm_factory: LLMProviderFactory = get_factory()
        self.llm: ChatModel = llm_factory.get_llm_provider()
        
        # System prompt for routing
        self.system_prompt = """You are a supervisor for a member services team in a health insurance company.

                                Available workers:
                                - member_lookup: Look up member information by ID
                                - check_eligibility: Check member eligibility and coverage status
                                - coverage_lookup: Get detailed coverage and benefits information

                                Route the query to the most appropriate worker based on the content.
                                If the query has been answered, respond with "FINISH".

                                Respond with JSON: {{"next": "worker_name", "reasoning": "why this worker"}}
                            """
                            
        self.planning_prompt = """
                                    You are a planning agent for a health insurance customer service system.
                                    Analyze the user's query and create a structured execution plan.

                                    User Query: {user_query}

                                    Relevant Knowledge Base Context:
                                    {semantic_context}

                                    Available workers: member_lookup_worker, check_eligibility_worker, coverage_lookup_worker
                                    Available semantic search tools: search_policy_info, search_knowledge_base

                                    Create a plan with ordered goals. Return JSON:
                                    {{
                                        "goals": [
                                            {{
                                                "id": "goal_1",
                                                "description": "Clear description of what to accomplish",
                                                "priority": 1,
                                                "required_workers": ["worker_name"]
                                            }}
                                        ],
                                        "steps": [
                                            {{
                                                "step_id": "step_1",
                                                "goal_id": "goal_1",
                                                "action": "Specific action to take",
                                                "worker": "worker_name"
                                            }}
                                        ]
                                    }}
                                """
    
    def create_routing_chain(self):
        """Create the LangGraph routing chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{messages}")
        ])
        
        return prompt | self.llm | JsonOutputParser()
    

    def create_plan_node(self, state: SupervisorState) -> SupervisorState:
        """
        First LLM call: Create execution plan with ordered goals.
        """
        # Check if plan already exists
        if state.get("plan"):
            return state
        
        tracer = get_langfuse_tracer()
        session_id = state.get("session_id", "default")
        
        try:
            # Get user query
            user_query = state["messages"][-1].content if state["messages"] else "No query"
            
            # Enrich planning with semantic search from Chroma vector DB
            semantic_context_json = {}
            
            try:
                chroma = get_chroma_data_access()
                policy_context = chroma.search_policies( query=user_query, n_results=2)
                faq_context = chroma.search_faqs(query=user_query, n_results=2)
                
                semantic_context_json = {
                    'relevant_policies': [r['document'] for r in policy_context],
                    'relevant_faqs': [r['document'] for r in faq_context]
                }
            except Exception:
                semantic_context = {}

            semantic_context = json.dumps(semantic_context_json, indent=2) if semantic_context_json else 'No additional context available.'
            
            # Create planning prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a planning agent. Create structured execution plans in JSON format."),
                ("human", self.planning_prompt)
            ])
            

            # Trace plan creation
            callback_handler = tracer.get_callback_handler() if tracer.enabled else None
            
            # Call LLM to create plan
            inputs = {
                "user_query" : user_query,
                "semantic_context" : semantic_context
            }
            
            if callback_handler:
                result = (prompt | self.llm).invoke( inputs, config={"callbacks": [callback_handler]})
            else:
                result = (prompt | self.llm).invoke(inputs )
            
            # Parse plan
            raw = result.content
            
            plan_text = re.sub(r"```json|```", "", str(raw)).strip()
            
            plan = json.loads( plan_text )
            
            # Store plan in Context Graph
            plan_id = self.cg_manager.store_plan(
                session_id=session_id,
                plan=plan,
                agent_name=self.name
            )
            
            # Update state
            state["plan_id"] = plan_id
            state["plan"] = plan
            state["current_goal_index"] = 0
            state["completed_goals"] = []
            
            logger.info(f"{self.name}: Created plan with {len(plan.get('goals', []))} goals")
            
            return state
            
        except Exception as e:
            logger.error(f"{self.name}: Error creating plan: {e}")
            
            # Fallback plan
            state["plan"] = {
                "goals": [{"id": "goal_1", "description": "Handle user query", "priority": 1}],
                "steps": [{"step_id": "step_1", "goal_id": "goal_1", "action": "process_query"}]
            }
            state["current_goal_index"] = 0
            state["completed_goals"] = []
            return state
        
    def supervisor_node(self, state: SupervisorState) -> SupervisorState:
        """
        LangGraph supervisor node - makes routing decisions with Context Graph integration.
        This replaces custom routing logic with LangGraph state machine.
        """
        user_id=state.get("user_id", "unknown")
        
        tracer = get_langfuse_tracer()
        
        # ── Control 5: Circuit Breaker Gate ──
        try:
            approval_wf = get_approval_workflow()
            
            if approval_wf.is_circuit_breaker_active():
                logger.warning(f"{self.name}: Circuit breaker active – halting team processing")
                
                # Audit action
                self.audit.log_action(
                    user_id=user_id,
                    action="circuit_breaker_block",
                    resource_type="MEMBER_SERVICES_AGENT",
                    resource_id=""
                )
                
                return {
                    "next": "FINISH",
                    "execution_path": state.get("execution_path", []) + [f"{self.name} -> BLOCKED_BY_CIRCUIT_BREAKER"],
                    "messages": [AIMessage(content="System is temporarily unavailable due to an emergency stop." )]
                }
        except Exception as e:
            logger.warning(f"{self.name}: Circuit breaker check failed (fail-open): {e}")
            #Will continue with the execution

        # Get plan from state
        session_id=state.get( 'session_id' ) or ""
        plan_id = state.get( 'plan_id' ) or ""
        plan = state.get("plan", {})
        current_goal_index = state.get("current_goal_index", 0)
        completed_goals = state.get("completed_goals", [])
        
        # Check if all goals completed
        goals = plan.get("goals", [])
        if current_goal_index >= len(goals):
            self.cg_manager.complete_plan( session_id, plan_id )
            return {"next": "FINISH", "execution_path": state.get("execution_path", [])}
        
        # Get current goal
        current_goal = goals[current_goal_index] if goals else None
        
        # ── Retrieve context from Context Graph (Neo4j) ──────────────
        session_id = state.get("session_id", "default")
        
        # Retrieve session context from Context Graph
        session_context = self.cg_manager.get_session_context(session_id)
        
        conversation_history = self.cg_manager.get_conversation_history(session_id, limit=5)
        
        # Track supervisor execution in Context Graph
        execution_id = self.cg_manager.track_agent_execution(
            session_id=session_id,
            agent_name=self.name,
            agent_type="supervisor",
            status="running"
        )
        
        
        # ── Build the message list for the routing LLM ───────────────
        #
        # Start with the current state messages (list[BaseMessage]).
        # Then append context as properly typed LangChain messages.
        #
        routing_messages: list[BaseMessage] = list(state.get("messages", []))
        
        # 1) Session context → SystemMessage
        if session_context:
            session_node = session_context.get("session", {})
            execution_count = session_context.get("execution_count", 0)
            session_context_text = (
                f"Session Context:\n"
                f"- User ID: {user_id}\n"
                f"- User Role: {session_node.get('userRole', 'unknown')}\n"
                f"- Member ID: {session_node.get('memberId', 'N/A')}\n"
                f"- Channel: {session_node.get('channel', 'web')}\n"
                f"- Prior Agent Executions in Session: {execution_count}"
            )
            
            routing_messages.append(SystemMessage(content=session_context_text))
        
        # 2) Conversation history → reconstruct as original message types
        if conversation_history:
            # Map Neo4j role strings to LangChain message classes
            role_to_message = {
                "user": HumanMessage,
                "human": HumanMessage,
                "assistant": AIMessage,
                "ai": AIMessage,
                "system": SystemMessage,
            }

            # Prepend a SystemMessage header so the LLM knows these
            # are prior turns, not new instructions.
            routing_messages.append(
                SystemMessage(
                    content=(
                        f"The following are the last "
                        f"{len(conversation_history)} messages from "
                        f"this conversation session:"
                    )
                )
            )

            for msg in reversed(conversation_history):  # oldest first
                role = msg.get("role", "system").lower()
                content = msg.get("content", "")
                msg_class = role_to_message.get(role, SystemMessage)
                routing_messages.append(msg_class(content=content))
        
        # 3) Plan context → SystemMessage
        if plan and current_goal:
            plan_context_text = (
                f"Current Execution Plan:\n"
                f"- Total Goals: {len(goals)}\n"
                f"- Current Goal ({current_goal_index + 1}/{len(goals)}): "
                f"{current_goal.get('description', '')}\n"
                f"- Completed Goals: {len(completed_goals)}\n"
                f"- Required Workers: "
                f"{current_goal.get('required_workers', [])}"
            )
            routing_messages.append(SystemMessage(content=plan_context_text))
        
        
        # ── Invoke the routing chain ─────────────────────────────────
        # Trace plan creation
        callback_handler = tracer.get_callback_handler() if tracer.enabled else None
                
        chain = self.create_routing_chain()
        
        if callback_handler:
            result = chain.invoke(
                                    { "messages": routing_messages },
                                    config={"callbacks": [callback_handler]} 
                            )
        else:
            result = chain.invoke(
                                    { "messages": routing_messages }
                            )

        next_worker = result.get("next", "FINISH")
        
        # ── Update plan progress if goal completed ───────────────────
        if next_worker == "FINISH" and current_goal:
            completed_goals.append(
                current_goal.get("id", f"goal_{current_goal_index}")
            )
            
            self.cg_manager.update_plan_progress(
                session_id=session_id,
                plan_id=plan_id,
                completed_goal_index=current_goal_index,
                goal_result="completed",
            )

            # Move to next goal
            current_goal_index += 1
            state["current_goal_index"] = current_goal_index
            state["completed_goals"] = completed_goals

            # Check if more goals remain
            if current_goal_index < len(goals):
                logger.info(
                    f"{self.name}: Goal {current_goal_index} completed, "
                    f"moving to goal {current_goal_index + 1}"
                )
                # Re-route: the next supervisor_node call will pick up
                # the new current_goal_index and route for the next goal.
                next_worker = "CONTINUE"  # Signal to re-enter supervisor
        
        # ── Update execution status in Context Graph ─────────────────
        if execution_id:
            self.cg_manager.update_execution_status(execution_id, "completed")
        
        # Add routing decision to Context Graph conversation
        self.cg_manager.add_message_to_session(
            session_id=session_id,
            role="system",
            content=f"Supervisor routing: {next_worker} - {result.get('reasoning', '')}",
            tool_calls=[],
        )
        
        # ── Update execution path ────────────────────────────────────
        execution_path = state.get("execution_path", [])
        execution_path.append(f"member_services_supervisor -> {next_worker}")
        
        # ── Audit routing decision ───────────────────────────────────
        self.audit.log_action(
            user_id=user_id,
            action="member_services_routing",
            resource_type="MEMBER_SERVICES_AGENT",
            resource_id=""
        )
        
        
        return {
            "next": next_worker,
            "execution_path": execution_path,
        }