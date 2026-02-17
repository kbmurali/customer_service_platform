"""
Context Graph Data Access Layer
Pure Cypher-based data access for Neo4j Context Graph (CG)
"""

from typing import Dict, Any, List, Optional
import logging
from databases.connections import get_neo4j_cg

logger = logging.getLogger(__name__)


class ContextGraphDataAccess:
    """
    Data access layer for Neo4j Context Graph.
    Provides methods for tracking conversations, agent executions, plans, and tool usage.
    Uses pure Cypher queries.
    """
    
    def __init__(self):
        """Initialize Context Graph data access."""
        self.conn = get_neo4j_cg()
        self.driver = self.conn.driver if hasattr(self.conn, 'driver') else None
    
    # ==================== Session Operations ====================
    
    def create_session(self, session_id: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new session in the Context Graph.
        
        Args:
            session_id: Unique session ID
            user_id: User ID
            metadata: Optional session metadata
            
        Returns:
            True if successful
        """
        query = """
        MERGE (s:Session {sessionId: $sessionId})
        ON CREATE SET 
            s.userId = $userId,
            s.startTime = datetime(),
            s.status = 'active',
            s.metadata = $metadata
        RETURN s.sessionId AS sessionId
        """
        
        try:
            result = self.conn.execute_query(query, {
                "sessionId": session_id,
                "userId": user_id,
                "metadata": metadata or {}
            })
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None if not found
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})
        RETURN s {
            .sessionId,
            .userId,
            .startTime,
            .endTime,
            .status,
            .metadata
        } AS session
        """
        
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id})
            if result and len(result) > 0:
                return result[0].get("session")
            return None
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})
        SET s.endTime = datetime(),
            s.status = 'closed',
            s.duration = duration.between(s.startTime, datetime()).seconds
        RETURN s.sessionId AS sessionId
        """
        
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id})
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False
    
    # ==================== Message Operations ====================
    
    def add_message(self, 
                   session_id: str,
                   role: str,
                   content: str,
                   tool_calls: Optional[List[Dict[str, Any]]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Add a message to a session.
        
        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            tool_calls: Optional tool calls
            metadata: Optional metadata
            
        Returns:
            Message ID if successful
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})
        CREATE (m:Message {
            messageId: randomUUID(),
            role: $role,
            content: $content,
            toolCalls: $toolCalls,
            metadata: $metadata,
            timestamp: datetime()
        })
        CREATE (s)-[:HAS_MESSAGE]->(m)
        RETURN m.messageId AS messageId
        """
        
        try:
            result = self.conn.execute_query(query, {
                "sessionId": session_id,
                "role": role,
                "content": content,
                "toolCalls": tool_calls or [],
                "metadata": metadata or {}
            })
            if result and len(result) > 0:
                return result[0].get("messageId")
            return None
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            return None
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages
            
        Returns:
            List of messages
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})-[:HAS_MESSAGE]->(m:Message)
        RETURN m {
            .messageId,
            .role,
            .content,
            .toolCalls,
            .timestamp
        } AS message
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """
        
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id, "limit": limit})
            return [r.get("message") for r in result if r.get("message")]
        except Exception as e:
            logger.error(f"Error retrieving conversation history for session {session_id}: {e}")
            return []
    
    # ==================== Agent Execution Operations ====================
    
    def track_agent_execution(self,
                             session_id: str,
                             agent_name: str,
                             agent_type: str,
                             status: str = "running",
                             metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Track agent execution.
        
        Args:
            session_id: Session ID
            agent_name: Agent name
            agent_type: Agent type (supervisor, worker)
            status: Execution status
            metadata: Optional metadata
            
        Returns:
            Execution ID if successful
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})
        CREATE (e:AgentExecution {
            executionId: randomUUID(),
            agentName: $agentName,
            agentType: $agentType,
            status: $status,
            metadata: $metadata,
            startTime: datetime()
        })
        CREATE (s)-[:HAS_EXECUTION]->(e)
        RETURN e.executionId AS executionId
        """
        
        try:
            result = self.conn.execute_query(query, {
                "sessionId": session_id,
                "agentName": agent_name,
                "agentType": agent_type,
                "status": status,
                "metadata": metadata or {}
            })
            if result and len(result) > 0:
                return result[0].get("executionId")
            return None
        except Exception as e:
            logger.error(f"Error tracking agent execution: {e}")
            return None
    
    def update_execution_status(self, 
                               execution_id: str,
                               status: str,
                               result: Optional[Dict[str, Any]] = None,
                               error: Optional[str] = None) -> bool:
        """
        Update agent execution status.
        
        Args:
            execution_id: Execution ID
            status: New status
            result: Optional execution result
            error: Optional error message
            
        Returns:
            True if successful
        """
        query = """
        MATCH (e:AgentExecution {executionId: $executionId})
        SET e.status = $status,
            e.endTime = datetime(),
            e.duration = duration.between(e.startTime, datetime()).milliseconds
        """
        
        if result:
            query += ", e.result = $result"
        if error:
            query += ", e.error = $error"
        
        query += " RETURN e.executionId AS executionId"
        
        params = {"executionId": execution_id, "status": status}
        if result:
            params["result"] = result
        if error:
            params["error"] = error
        
        try:
            result_data = self.conn.execute_query(query, params)
            return len(result_data) > 0
        except Exception as e:
            logger.error(f"Error updating execution status {execution_id}: {e}")
            return False
    
    def get_execution_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve execution history for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of executions
            
        Returns:
            List of executions
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})-[:HAS_EXECUTION]->(e:AgentExecution)
        RETURN e {
            .executionId,
            .agentName,
            .agentType,
            .status,
            .startTime,
            .endTime,
            .duration,
            .error
        } AS execution
        ORDER BY e.startTime DESC
        LIMIT $limit
        """
        
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id, "limit": limit})
            return [r.get("execution") for r in result if r.get("execution")]
        except Exception as e:
            logger.error(f"Error retrieving execution history for session {session_id}: {e}")
            return []
    
    # ==================== Plan Operations ====================
    
    def store_plan(self,
                  session_id: str,
                  plan: Dict[str, Any],
                  agent_name: str) -> Optional[str]:
        """
        Store execution plan in Context Graph.
        
        Args:
            session_id: Session ID
            plan: Plan data with goals and steps
            agent_name: Agent that created the plan
            
        Returns:
            Plan ID if successful
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})
        CREATE (p:Plan {
            planId: randomUUID(),
            goals: $goals,
            steps: $steps,
            agentName: $agentName,
            status: 'active',
            createdAt: datetime()
        })
        CREATE (s)-[:HAS_PLAN]->(p)
        RETURN p.planId AS planId
        """
        
        try:
            result = self.conn.execute_query(query, {
                "sessionId": session_id,
                "goals": plan.get("goals", []),
                "steps": plan.get("steps", []),
                "agentName": agent_name
            })
            if result and len(result) > 0:
                return result[0].get("planId")
            return None
        except Exception as e:
            logger.error(f"Error storing plan for session {session_id}: {e}")
            return None
    
    def get_active_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve active plan for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Plan data or None if not found
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})-[:HAS_PLAN]->(p:Plan {status: 'active'})
        RETURN p {
            .planId,
            .goals,
            .steps,
            .agentName,
            .status,
            .createdAt
        } AS plan
        ORDER BY p.createdAt DESC
        LIMIT 1
        """
        
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id})
            if result and len(result) > 0:
                return result[0].get("plan")
            return None
        except Exception as e:
            logger.error(f"Error retrieving active plan for session {session_id}: {e}")
            return None
    
    def update_plan_progress(self,
                            session_id: str,
                            completed_goal_index: int,
                            goal_result: str) -> bool:
        """
        Update plan progress with completed goal.
        
        Args:
            session_id: Session ID
            completed_goal_index: Index of completed goal
            goal_result: Result of goal execution
            
        Returns:
            True if successful
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})-[:HAS_PLAN]->(p:Plan {status: 'active'})
        CREATE (gc:GoalCompletion {
            goalIndex: $goalIndex,
            result: $result,
            completedAt: datetime()
        })
        CREATE (p)-[:COMPLETED_GOAL]->(gc)
        RETURN p.planId AS planId
        """
        
        try:
            result = self.conn.execute_query(query, {
                "sessionId": session_id,
                "goalIndex": completed_goal_index,
                "result": goal_result
            })
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error updating plan progress for session {session_id}: {e}")
            return False
    
    def complete_plan(self, session_id: str) -> bool:
        """
        Mark plan as completed.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})-[:HAS_PLAN]->(p:Plan {status: 'active'})
        SET p.status = 'completed',
            p.completedAt = datetime()
        RETURN p.planId AS planId
        """
        
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id})
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error completing plan for session {session_id}: {e}")
            return False
    
    # ==================== Tool Execution Operations ====================
    
    def track_tool_execution(self,
                            session_id: str,
                            tool_name: str,
                            input_data: Dict[str, Any],
                            output_data: Optional[Dict[str, Any]] = None,
                            status: str = "success",
                            execution_time_ms: Optional[float] = None,
                            error: Optional[str] = None) -> Optional[str]:
        """
        Track tool execution.
        
        Args:
            session_id: Session ID
            tool_name: Tool name
            input_data: Tool input
            output_data: Tool output
            status: Execution status
            execution_time_ms: Execution time in milliseconds
            error: Optional error message
            
        Returns:
            Tool execution ID if successful
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})
        CREATE (t:ToolExecution {
            toolExecutionId: randomUUID(),
            toolName: $toolName,
            input: $input,
            output: $output,
            status: $status,
            execution_time_ms: $execution_time_ms,
            error: $error,
            timestamp: datetime()
        })
        CREATE (s)-[:USED_TOOL]->(t)
        RETURN t.toolExecutionId AS toolExecutionId
        """
        
        try:
            result = self.conn.execute_query(query, {
                "sessionId": session_id,
                "toolName": tool_name,
                "input": input_data,
                "output": output_data or {},
                "status": status,
                "execution_time_ms": execution_time_ms,
                "error": error
            })
            if result and len(result) > 0:
                return result[0].get("toolExecutionId")
            return None
        except Exception as e:
            logger.error(f"Error tracking tool execution: {e}")
            return None
    
    def get_tool_usage_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get tool usage statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Tool usage statistics
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})-[:USED_TOOL]->(t:ToolExecution)
        RETURN 
            count(t) AS totalToolCalls,
            count(CASE WHEN t.status = 'success' THEN 1 END) AS successfulCalls,
            count(CASE WHEN t.status = 'error' THEN 1 END) AS failedCalls,
            collect(DISTINCT t.toolName) AS toolsUsed
        """
        
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id})
            if result and len(result) > 0:
                return result[0]
            return {"totalToolCalls": 0, "successfulCalls": 0, "failedCalls": 0, "toolsUsed": []}
        except Exception as e:
            logger.error(f"Error retrieving tool usage stats for session {session_id}: {e}")
            return {"totalToolCalls": 0, "successfulCalls": 0, "failedCalls": 0, "toolsUsed": []}
    
    # ==================== Security Event Operations ====================
    
    def log_security_event(self,
                          session_id: str,
                          event_type: str,
                          severity: str,
                          details: Dict[str, Any]) -> Optional[str]:
        """
        Log a security event.
        
        Args:
            session_id: Session ID
            event_type: Event type (permission_denied, rate_limit, etc.)
            severity: Severity level (low, medium, high, critical)
            details: Event details
            
        Returns:
            Event ID if successful
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})
        CREATE (e:SecurityEvent {
            eventId: randomUUID(),
            eventType: $eventType,
            severity: $severity,
            details: $details,
            timestamp: datetime()
        })
        CREATE (s)-[:HAS_SECURITY_EVENT]->(e)
        RETURN e.eventId AS eventId
        """
        
        try:
            result = self.conn.execute_query(query, {
                "sessionId": session_id,
                "eventType": event_type,
                "severity": severity,
                "details": details
            })
            if result and len(result) > 0:
                return result[0].get("eventId")
            return None
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
            return None
    
    def get_security_events(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve security events for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of events
            
        Returns:
            List of security events
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})-[:HAS_SECURITY_EVENT]->(e:SecurityEvent)
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
        
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id, "limit": limit})
            return [r.get("event") for r in result if r.get("event")]
        except Exception as e:
            logger.error(f"Error retrieving security events for session {session_id}: {e}")
            return []
    
    # ==================== Utility Methods ====================
    
    def execute_custom_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.
        
        Args:
            query: Cypher query
            params: Query parameters
            
        Returns:
            Query results
        """
        try:
            return self.conn.execute_query(query, params or {})
        except Exception as e:
            logger.error(f"Error executing custom query: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()


# Singleton instance
_cg_data_access_instance = None

def get_cg_data_access() -> ContextGraphDataAccess:
    """Get singleton instance of Context Graph Data Access."""
    global _cg_data_access_instance
    if _cg_data_access_instance is None:
        _cg_data_access_instance = ContextGraphDataAccess()
    return _cg_data_access_instance
