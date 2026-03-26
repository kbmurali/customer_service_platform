"""
Context Graph utilities for supervisor-level context retrieval and tracking.

This module integrates Neo4j Context Graph (CG) at the supervisor level to:
1. Retrieve conversation history and context
2. Track agent executions and tool usage
3. Monitor security events
4. Update session information

All operations use the ContextGraphDataAccess layer with pure Cypher queries.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging

from databases.context_graph_data_access import get_cg_data_access

logger = logging.getLogger(__name__)


class ContextGraphManager:
    """
    Manager for Context Graph operations at supervisor level.
    
    Provides methods to:
    - Retrieve session context
    - Track agent executions
    - Monitor security events
    - Update conversation history
    - Store and manage execution plans
    """
    
    def __init__(self):
        self.cg_data_access = get_cg_data_access()
    
    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session context from Context Graph.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session context including user info, history, and metadata
        """
        try:
            session = self.cg_data_access.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found in Context Graph")
                return None
            
            # Get agent executions for this session
            agent_executions = self.cg_data_access.get_execution_history(session_id)
            
            return {
                "session": session,
                "agent_executions": agent_executions,
                "execution_count": len(agent_executions)
            }
        except Exception as e:
            logger.error(f"Error retrieving session context: {e}")
            return None
    
    def link_follow_up_session(
        self,
        prior_session_id: str,
        new_session_id: str,
    ) -> bool:
        """
        Create a HAS_FOLLOW_UP relationship between two Session nodes.

        Establishes the conversation chain:
            Session1 -[:HAS_FOLLOW_UP]-> Session2 -[:HAS_FOLLOW_UP]-> Session3

        Called from request_processor.py after create_session() when
        prior_session_id is provided.
        """
        try:
            return self.cg_data_access.link_follow_up_session(
                prior_session_id=prior_session_id,
                new_session_id=new_session_id,
            )
        except Exception as exc:
            logger.error(
                "Error linking follow-up session %s -> %s: %s",
                prior_session_id, new_session_id, exc,
            )
            return False

    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
        
        Returns:
            List of conversation messages
        """
        try:
            return self.cg_data_access.get_conversation_history(session_id, limit)
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def track_agent_execution(
        self,
        session_id: str,
        agent_name: str,
        agent_type: str,
        status: str = "running",
        tools_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Track agent execution in Context Graph.

        Delegates to DAL which only creates (Session)-[:HAS_EXECUTION] for
        a2a_client — all other types (supervisor, a2a_server, worker) are
        standalone nodes linked via the plan graph.

        Args:
            session_id: Session identifier
            agent_name: Name of the agent
            agent_type: Type of agent (a2a_client, a2a_server, supervisor, worker)
            status:     Execution status (running, completed, failed)
            tools_used: List of tools used (stored as toolCallCount)
            metadata:   Optional metadata dict

        Returns:
            Execution ID if successful, None otherwise
        """
        try:
            merged_metadata = dict(metadata or {})
            if tools_used:
                merged_metadata["toolCallCount"] = len(tools_used)
            return self.cg_data_access.track_agent_execution(
                session_id=session_id,
                agent_name=agent_name,
                agent_type=agent_type,
                status=status,
                metadata=merged_metadata or None,
            )
        except Exception as e:
            logger.error(f"Error tracking agent execution: {e}")
            return None
    
    def update_execution_status(
        self,
        execution_id: str,
        status: str,
        error_message: Optional[str] = None,
        routing_note: Optional[str] = None,
        worker_name: Optional[str] = None,
    ) -> bool:
        """
        Update agent execution status.

        Args:
            execution_id: Execution identifier
            status:       New status (completed, failed)
            error_message: Error message if failed
            routing_note: Routing decision text stored directly on the node,
                          e.g. "Routing: member_lookup — goal is to look up member"
            worker_name:  Worker name to set as agentName once routing is known,
                          e.g. "member_lookup_worker"
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.cg_data_access.update_execution_status(
                execution_id, status,
                error=error_message,
                routing_note=routing_note,
                worker_name=worker_name,
            )
        except Exception as e:
            logger.error(f"Error updating execution status: {e}")
            return False


    def create_tool_error(
        self,
        tool_execution_id: str,
        error_type: str,
        error_message: str,
    ) -> Optional[str]:
        """
        Attach a ToolError node to a ToolExecution.

        Covers all errors within the tool boundary: decorator guards
        (rate limit, circuit breaker, permission denied, pending approval)
        and runtime exceptions inside the tool function itself.
        Creates (ToolExecution)-[:HAD_ERROR]->(ToolError).

        Args:
            tool_execution_id: ToolExecution.toolExecutionId
            error_type:        failed | not_found | rate_limited |
                               rate_limit_exceeded | circuit_breaker_active |
                               tool_permission_denied |
                               resource_permission_denied | pending_approval
            error_message:     Full human-readable error detail
        """
        try:
            return self.cg_data_access.create_tool_error(
                tool_execution_id=tool_execution_id,
                error_type=error_type,
                error_message=error_message,
            )
        except Exception as e:
            logger.error(f"Error creating tool error node: {e}")
            return None

    def create_agent_error(
        self,
        execution_id: str,
        error_type: str,
        error_message: str,
    ) -> Optional[str]:
        """
        Attach an AgentError node to an AgentExecution.

        Used for failures outside the tool boundary: worker-level logic
        errors, LLM failures, state machine errors, unhandled exceptions
        in the agent loop. These have no associated ToolExecution.
        Creates (AgentExecution)-[:HAD_ERROR]->(AgentError) and sets
        AgentExecution.status = 'failed'.

        Args:
            execution_id:  AgentExecution.executionId
            error_type:    e.g. "llm_error", "state_error",
                           "unhandled_exception"
            error_message: Full human-readable error detail
        """
        try:
            return self.cg_data_access.create_agent_error(
                execution_id=execution_id,
                error_type=error_type,
                error_message=error_message,
            )
        except Exception as e:
            logger.error(f"Error creating agent error node: {e}")
            return None

    def get_recent_security_events(
        self,
        session_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent security events from Context Graph.
        
        Args:
            session_id: Optional session ID to filter by
            severity: Optional severity level (critical, high, medium, low)
            limit: Maximum number of events to retrieve
        
        Returns:
            List of security events
        """
        try:
            if session_id:
                events = self.cg_data_access.get_security_events(session_id, limit)
            else:
                # Query all recent security events without session filter
                query = """
                MATCH (e:SecurityEvent)
                WHERE ($severity IS NULL OR e.severity = $severity)
                RETURN e {
                    .eventId,
                    .eventType,
                    .severity,
                    .details,
                    .timestamp
                } AS event
                ORDER BY e.timestamp DESC
                LIMIT $limit
                """
                results = self.cg_data_access.conn.execute_query(
                    query, {"severity": severity, "limit": limit}
                )
                events = [r.get("event") for r in results if r.get("event")]
            
            # Filter by severity if specified and session-filtered
            if severity and session_id:
                events = [e for e in events if e.get("severity") == severity]
            
            return events
        except Exception as e:
            logger.error(f"Error retrieving security events: {e}")
            return []
    
    def create_session(
        self,
        session_id: str,
        user_id: str,
        user_role: str,
        member_id: Optional[str] = None,
        channel: str = "web"
    ) -> bool:
        """
        Create a new session in Context Graph.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            user_role: User role
            member_id: Optional member ID
            channel: Communication channel
        
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
            CREATE (s:Session {
                sessionId: $sessionId,
                userId: $userId,
                userRole: $userRole,
                memberId: $memberId,
                startTime: datetime(),
                status: 'active',
                channel: $channel,
                interactionCount: 0
            })
            RETURN s.sessionId as sessionId
            """
            
            results = self.cg_data_access.conn.execute_query(
                query,
                {
                    "sessionId": session_id,
                    "userId": user_id,
                    "userRole": user_role,
                    "memberId": member_id,
                    "channel": channel
                }
            )
            
            if results:
                logger.info(f"Created session in Context Graph: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a session in Context Graph.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.cg_data_access.close_session(session_id)
        except Exception as e:
            logger.error(f"Error closing session: {e}")
            return False
    

    def link_session_to_execution(self, session_id: str, execution_id: str) -> None:
        """
        Create (Session)-[:HAS_EXECUTION]->(AgentExecution).

        The DAL's track_agent_execution() only creates this edge for
        agent_type='a2a_client'. For the central supervisor planner
        (agent_type='supervisor') the node is created standalone.
        This method explicitly stitches it to the Session so the
        planner AgentExecution is reachable from the Session root.

        Called from central_supervisor.create_plan_node() immediately
        after track_agent_execution returns the planning_execution_id.
        """
        try:
            self.cg_data_access.conn.execute_query("""
                MATCH (s:Session {sessionId: $sessionId})
                MATCH (e:AgentExecution {executionId: $executionId})
                MERGE (s)-[:HAS_EXECUTION]->(e)
            """, {
                "sessionId":   session_id,
                "executionId": execution_id,
            })
            logger.debug(
                "Linked Session %s -[:HAS_EXECUTION]-> AgentExecution %s",
                session_id, execution_id,
            )
        except Exception as e:
            logger.warning(
                "link_session_to_execution failed (non-fatal): %s", e
            )

    def link_planner_to_plan(self, execution_id: str, plan_id: str) -> None:
        """
        Create (AgentExecution "central_supervisor_planner")-[:HAS_PLAN]->(CentralPlan).

        This mirrors how the team side works:
            (a2a_server AgentExecution)-[:HAS_PLAN]->(TeamPlan)

        The CentralPlan is therefore reachable via:
            Session -[HAS_EXECUTION]→ planner AE -[HAS_PLAN]→ CentralPlan

        The DAL's store_plan() creates the Plan node standalone — it does
        NOT create any HAS_PLAN edge. This method creates it explicitly
        after store_plan() returns a valid plan_id.

        Called from central_supervisor.create_plan_node() immediately
        after store_plan() returns a valid plan_id.

        Note: Session-[:HAS_PLAN]→Plan is intentionally NOT created.
        The Session has no direct HAS_PLAN edge — plans are reachable
        only via AgentExecution nodes (planner for central, a2a_server
        for team), keeping the graph design consistent across both levels.
        """
        try:
            self.cg_data_access.conn.execute_query("""
                MATCH (e:AgentExecution {executionId: $executionId})
                MATCH (p:Plan {planId: $planId})
                MERGE (e)-[:HAS_PLAN]->(p)
            """, {
                "executionId": execution_id,
                "planId":      plan_id,
            })
            logger.debug(
                "Linked planner AgentExecution %s -[:HAS_PLAN]-> CentralPlan %s",
                execution_id, plan_id,
            )
        except Exception as e:
            logger.warning(
                "link_planner_to_plan failed (non-fatal): %s", e
            )

    def link_step_to_execution(self, plan_id: str, step_id: str, execution_id: str) -> None:
        """Create (Step)-[:EXECUTED_BY]->(AgentExecution) scoped by planId."""
        self.cg_data_access.link_step_to_execution(plan_id, step_id, execution_id)

    def set_langfuse_trace_id(self, execution_id: str, langfuse_trace_id: str) -> None:
        """
        Store the LangFuse trace ID on an AgentExecution node.

        Delegates to the underlying DAL method.  Called after each LLM
        .invoke() to write the auto-generated LangFuse trace ID back to
        the CG node so the CG Explorer can render inline trace cards.
        """
        self.cg_data_access.set_langfuse_trace_id(execution_id, langfuse_trace_id)

    def link_router_to_worker(self, router_execution_id: str, worker_execution_id: str) -> None:
        """Create (router)-[:DISPATCHED_TO]->(worker)."""
        self.cg_data_access.link_router_to_worker(router_execution_id, worker_execution_id)


    def link_supervisor_to_a2a_client(
        self,
        routing_execution_id: str,
        a2a_task_id: str,
    ) -> None:
        """
        Create (routing AgentExecution)-[:CALLED_AGENT]->(a2a_client AgentExecution).

        This is the missing edge that joins the central supervisor's routing
        decision node to the A2A client dispatch node, completing the chain:

            (Step)-[:EXECUTED_BY]->(routing AgentExecution)
                -[:CALLED_AGENT]->(a2a_client AgentExecution)
                    -[:CALLED_AGENT]->(a2a_server AgentExecution)
                        -[:HAS_PLAN]->(TeamPlan)

        The routing AgentExecution is matched by its executionId directly.
        The a2a_client AgentExecution is matched by the shared a2a_task_id
        stored in its metadata JSON — the same strategy used by
        link_a2a_client_to_server.

        Args:
            routing_execution_id: AgentExecution.executionId of the central
                                  supervisor routing node (current_execution_id).
            a2a_task_id:          The A2A task UUID shared between the central
                                  supervisor's agent_node and A2AClientNode.
        """
        try:
            self.cg_data_access.conn.execute_query("""
                MATCH (routing:AgentExecution {executionId: $routingId})
                MATCH (client:AgentExecution {agentType: "a2a_client"})
                WHERE client.metadata CONTAINS $taskId
                MERGE (routing)-[:CALLED_AGENT]->(client)
            """, {
                "routingId": routing_execution_id,
                "taskId":    a2a_task_id,
            })
            logger.debug(
                "Linked routing execution %s -> a2a_client for task %s",
                routing_execution_id, a2a_task_id,
            )
        except Exception as e:
            logger.warning("Failed to link supervisor to a2a_client (non-fatal): %s", e)

    def link_a2a_client_to_server(self, session_id: str, a2a_task_id: str) -> None:
        """Create (a2a_client)-[:CALLED_AGENT]->(a2a_server) via shared a2a_task_id."""
        self.cg_data_access.link_a2a_client_to_server(session_id, a2a_task_id)

    def link_a2a_server_to_plan(self, session_id: str, a2a_task_id: str, plan_id: str) -> None:
        """Create (a2a_server)-[:HAS_PLAN]->(Plan)."""
        self.cg_data_access.link_a2a_server_to_plan(session_id, a2a_task_id, plan_id)

    def link_a2a_server_to_planner(self, a2a_task_id: str, plan_id: str) -> None:
        """
        Create (a2a_server)-[:HAS_EXECUTION]->(team_planner) and remove
        the direct (a2a_server)-[:HAS_PLAN]->(Plan) edge.
        """
        self.cg_data_access.link_a2a_server_to_planner(a2a_task_id, plan_id)

    def store_plan(
        self,
        session_id: str,
        plan: Dict[str, Any],
        agent_name: str = "supervisor",
        plan_type: str = "central",
        team_name: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Store execution plan as a proper node graph.

        Creates (Plan)-[:HAS_GOAL]->(Goal)-[:HAS_STEP]->(Step).
        The Plan is linked to its owner (planner AE or a2a_server AE)
        via link_planner_to_plan() or link_a2a_server_to_plan() respectively.

        Returns:
            Dict {"plan_id": str, "step_map": {step_id: step_id}} or None
        """
        try:
            return self.cg_data_access.store_plan(
                session_id=session_id,
                plan=plan,
                agent_name=agent_name,
                plan_type=plan_type,
                team_name=team_name,
            )
        except Exception as e:
            logger.error(f"Error storing plan: {e}")
            return None
    
    def get_active_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve active plan for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Active plan if found, None otherwise
        """
        try:
            return self.cg_data_access.get_active_plan(session_id)
        except Exception as e:
            logger.error(f"Error retrieving active plan: {e}")
            return None
    
    def update_plan_progress(
        self,
        session_id: str,
        plan_id: str,
        goal_id: str,
        goal_result: str,
    ) -> bool:
        """
        Mark a Goal and its Steps as completed or skipped.

        Args:
            session_id:  Session identifier
            plan_id:     Plan ID
            goal_id:     Goal.goalId to mark
            goal_result: "completed" or "skipped"
        """
        try:
            return self.cg_data_access.update_plan_progress(
                session_id=session_id,
                plan_id=plan_id,
                goal_id=goal_id,
                goal_result=goal_result,
            )
        except Exception as e:
            logger.error(f"Error updating plan progress: {e}")
            return False
    
    def fail_plan(self, session_id: str, plan_id: str) -> bool:
        """Mark plan as incomplete due to a goal failure."""
        try:
            return self.cg_data_access.fail_plan(session_id, plan_id)
        except Exception as e:
            logger.error(f"Error marking plan incomplete: {e}")
            return False

    def cancel_remaining_goals(self, session_id: str, plan_id: str) -> int:
        """
        Mark all pending Goals and Steps as 'cancelled' after a goal failure.

        Returns:
            Number of goals cancelled, or 0 on error.
        """
        try:
            return self.cg_data_access.cancel_remaining_goals(session_id, plan_id)
        except Exception as e:
            logger.error(f"Error cancelling remaining goals: {e}")
            return 0

    def complete_plan(self, session_id: str, plan_id:str ) -> bool:
        """
        Mark plan as completed.
        
        Args:
            session_id: Session identifier
            plan_id: Plan identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.cg_data_access.complete_plan( session_id, plan_id )
        except Exception as e:
            logger.error(f"Error completing plan: {e}")
            return False


# Singleton instance
_cg_manager = None


def get_context_graph_manager() -> ContextGraphManager:
    """Get singleton instance of Context Graph Manager."""
    global _cg_manager
    if _cg_manager is None:
        _cg_manager = ContextGraphManager()
    return _cg_manager
