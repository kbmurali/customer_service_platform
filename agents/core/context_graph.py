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
        tools_used: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Track agent execution in Context Graph.
        
        Args:
            session_id: Session identifier
            agent_name: Name of the agent
            agent_type: Type of agent (supervisor, worker)
            status: Execution status (running, completed, failed)
            tools_used: List of tools used by the agent
        
        Returns:
            Execution ID if successful, None otherwise
        """
        try:
            execution_id = f"{session_id}_{agent_name}_{datetime.now(timezone.utc).isoformat()}"
            
            query = """
            MATCH (s:Session {sessionId: $sessionId})
            CREATE (e:AgentExecution {
                executionId: $executionId,
                agentName: $agentName,
                agentType: $agentType,
                startTime: datetime(),
                status: $status,
                toolCallCount: $toolCallCount
            })
            CREATE (s)-[:HAS_EXECUTION]->(e)
            RETURN e.executionId as executionId
            """
            
            results = self.cg_data_access.conn.execute_query(
                query,
                {
                    "sessionId": session_id,
                    "executionId": execution_id,
                    "agentName": agent_name,
                    "agentType": agent_type,
                    "status": status,
                    "toolCallCount": len(tools_used) if tools_used else 0
                }
            )
            
            if results:
                logger.info(f"Tracked agent execution: {execution_id}")
                return results[0]["executionId"]
            return None
        except Exception as e:
            logger.error(f"Error tracking agent execution: {e}")
            return None
    
    def update_execution_status(
        self,
        execution_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update agent execution status.
        
        Args:
            execution_id: Execution identifier
            status: New status (completed, failed)
            error_message: Error message if failed
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.cg_data_access.update_execution_status(
                execution_id, status, error_message
            )
        except Exception as e:
            logger.error(f"Error updating execution status: {e}")
            return False
    
    def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Add a message to the conversation session.
        
        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            tool_calls: List of tool calls made
        
        Returns:
            Message ID if successful, None otherwise
        """
        try:
            return self.cg_data_access.add_message(
                session_id, role, content, tool_calls
            )
        except Exception as e:
            logger.error(f"Error adding message to session: {e}")
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
    
    def store_plan(
        self,
        session_id: str,
        plan: Dict[str, Any],
        agent_name: str = "supervisor"
    ) -> Optional[str]:
        """
        Store execution plan in Context Graph.
        
        Args:
            session_id: Session identifier
            plan: Plan dictionary with goals and steps
            agent_name: Name of agent that created the plan
        
        Returns:
            True if successful, False otherwise
        """
        try:
            plan_id = self.cg_data_access.store_plan( 
                                                  session_id=session_id,
                                                  plan=plan,
                                                  agent_name=agent_name
            )
            
            return plan_id
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
        completed_goal_index: int,
        goal_result: str
    ) -> bool:
        """
        Update plan progress when a goal is completed.
        
        Args:
            session_id: Session identifier
            completed_goal_index: Index of completed goal
            goal_result: Result of the completed goal
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.cg_data_access.update_plan_progress( 
                            session_id=session_id,
                            plan_id=plan_id,
                            completed_goal_index=completed_goal_index,
                            goal_result=goal_result
                        )
        except Exception as e:
            logger.error(f"Error updating plan progress: {e}")
            return False
    
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
