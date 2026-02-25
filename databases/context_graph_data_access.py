"""
Context Graph Data Access Layer
Pure Cypher-based data access for Neo4j Context Graph (CG)
"""

from typing import Dict, Any, List, Optional
import logging
import json

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
                "metadata": json.dumps( metadata or {} )
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
    
    # ==================== Conversation History ====================

    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve routing history for a session from AgentExecution.routingNote.

        Routing decisions are stored directly on the AgentExecution that produced
        them, eliminating separate Message nodes and relationships.
        Traverses: Session->a2a_client->a2a_server->Plan->Goal->Step->AgentExecution

        Args:
            session_id: Session ID
            limit:      Maximum number of entries to return

        Returns:
            List of dicts with keys: role, content, timestamp
        """
        query = """
        MATCH (:Session {sessionId: $sessionId})-[:HAS_EXECUTION]->
              (:AgentExecution {agentType: 'a2a_client'})-[:CALLED_AGENT]->
              (:AgentExecution {agentType: 'a2a_server'})-[:HAS_PLAN]->
              (:Plan)-[:HAS_GOAL]->(:Goal)-[:HAS_STEP]->(:Step)
              -[:EXECUTED_BY]->(e:AgentExecution)
        WHERE e.routingNote IS NOT NULL
        RETURN {
            role:      'system',
            content:   e.routingNote,
            timestamp: toString(e.endTime)
        } AS message
        ORDER BY e.endTime DESC
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

        Only a2a_client executions are linked directly to the Session via
        HAS_EXECUTION — they are the top-level entry point for a user request.

        All other agent types (supervisor, a2a_server, worker) are reachable
        through the plan graph:
            (Session)-[:HAS_EXECUTION]->(a2a_client)
                -[:CALLED_AGENT]->(a2a_server)
                    -[:HAS_PLAN]->(Plan)
                        -[:HAS_GOAL]->(Goal)
                            -[:HAS_STEP]->(Step)
                                -[:EXECUTED_BY]->(supervisor AgentExecution)
                                    -[:CALLED_TOOL]->(ToolExecution)

        Args:
            session_id: Session ID
            agent_name: Agent name
            agent_type: Agent type (a2a_client, a2a_server, supervisor, worker)
            status:     Execution status
            metadata:   Optional metadata

        Returns:
            Execution ID if successful
        """
        if agent_type == "a2a_client":
            # Top-level entry point — link directly to Session
            query = """
            MATCH (s:Session {sessionId: $sessionId})
            CREATE (e:AgentExecution {
                executionId: randomUUID(),
                agentName:   $agentName,
                agentType:   $agentType,
                status:      $status,
                metadata:    $metadata,
                startTime:   datetime()
            })
            CREATE (s)-[:HAS_EXECUTION]->(e)
            RETURN e.executionId AS executionId
            """
        else:
            # Lower-level: standalone node, linked via plan graph relationships
            # (a2a_server via CALLED_AGENT, supervisor via EXECUTED_BY)
            query = """
            CREATE (e:AgentExecution {
                executionId: randomUUID(),
                agentName:   $agentName,
                agentType:   $agentType,
                status:      $status,
                metadata:    $metadata,
                startTime:   datetime()
            })
            RETURN e.executionId AS executionId
            """

        try:
            result = self.conn.execute_query(query, {
                "sessionId": session_id,
                "agentName": agent_name,
                "agentType": agent_type,
                "status":    status,
                "metadata":  json.dumps(metadata or {}),
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
                               error: Optional[str] = None,
                               routing_note: Optional[str] = None,
                               worker_name: Optional[str] = None) -> bool:
        """
        Update agent execution status.
        
        Args:
            execution_id: Execution ID
            status:       New status
            result:       Optional execution result
            error:        Optional error message
            routing_note: Optional routing decision stored directly on the node,
                          replacing separate Message nodes.
                          e.g. "Routing: member_lookup — goal is to look up member"
            worker_name:  Optional worker name to set as agentName, e.g.
                          "member_lookup_worker". Set when the routing decision
                          is known (after LLM call) so the node reflects the
                          actual worker rather than the supervisor name.
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
        if routing_note:
            query += ", e.routingNote = $routingNote"
        if worker_name:
            query += ", e.agentName = $workerName"
        
        query += " RETURN e.executionId AS executionId"
        
        params = {"executionId": execution_id, "status": status}
        if result:
            params["result"] = json.dumps( result or {} )
        if error:
            params["error"] = error
        if routing_note:
            params["routingNote"] = routing_note
        if worker_name:
            params["workerName"] = worker_name
        
        try:
            result_data = self.conn.execute_query(query, params)
            return len(result_data) > 0
        except Exception as e:
            logger.error(f"Error updating execution status {execution_id}: {e}")
            return False
    
    def get_execution_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve execution history for a session.

        Collects all AgentExecutions reachable from the session:
          - a2a_client (direct HAS_EXECUTION)
          - a2a_server (via CALLED_AGENT)
          - supervisor  (via HAS_PLAN->Goal->Step->EXECUTED_BY)

        Args:
            session_id: Session ID
            limit:      Maximum number of executions

        Returns:
            List of executions
        """
        query = """
        MATCH (s:Session {sessionId: $sessionId})-[:HAS_EXECUTION]->(e:AgentExecution)
        RETURN e {
            .executionId, .agentName, .agentType,
            .status, .startTime, .endTime, .duration, .error
        } AS execution
        UNION
        MATCH (:Session {sessionId: $sessionId})-[:HAS_EXECUTION]->
              (:AgentExecution {agentType: 'a2a_client'})-[:CALLED_AGENT]->(e:AgentExecution)
        RETURN e {
            .executionId, .agentName, .agentType,
            .status, .startTime, .endTime, .duration, .error
        } AS execution
        UNION
        MATCH (:Session {sessionId: $sessionId})-[:HAS_EXECUTION]->
              (:AgentExecution {agentType: 'a2a_client'})-[:CALLED_AGENT]->
              (:AgentExecution {agentType: 'a2a_server'})-[:HAS_PLAN]->
              (:Plan)-[:HAS_GOAL]->(:Goal)-[:HAS_STEP]->(:Step)-[:EXECUTED_BY]->(e:AgentExecution)
        RETURN e {
            .executionId, .agentName, .agentType,
            .status, .startTime, .endTime, .duration, .error
        } AS execution
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
                  agent_name: str,
                  plan_type: str = "central",
                  team_name: str = "",
                  central_step_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Store execution plan as a proper node graph.

        Creates:
            (Session)-[:HAS_PLAN]->(Plan)
            (Plan)-[:HAS_GOAL]->(Goal)
            (Goal)-[:HAS_STEP]->(Step)

        When central_step_id is provided (team plan from A2A delegation):
            (CentralStep)-[:DELEGATED_TO]->(TeamPlan)

        Returns:
            Dict {"plan_id": str, "step_map": {step_id: step_id}} or None
        """
        try:
            goals = plan.get("goals", [])
            steps = plan.get("steps", [])

            # ── Create Plan node ──────────────────────────────────────────
            result = self.conn.execute_query("""
                MERGE (s:Session {sessionId: $sessionId})
                ON CREATE SET s.createdAt = datetime(), s.status = 'active'
                CREATE (p:Plan {
                    planId:     randomUUID(),
                    agentName:  $agentName,
                    planType:   $planType,
                    teamName:   $teamName,
                    status:     'active',
                    totalGoals: $totalGoals,
                    createdAt:  datetime()
                })
                RETURN p.planId AS planId
            """, {
                "sessionId":  session_id,
                "agentName":  agent_name,
                "planType":   plan_type,
                "teamName":   team_name,
                "totalGoals": len(goals),
            })
            if not result:
                return None
            plan_id = result[0].get("planId")

            # ── Link CentralStep -> TeamPlan via DELEGATED_TO ─────────────
            if central_step_id:
                self.conn.execute_query("""
                    OPTIONAL MATCH (cs:Step {stepId: $centralStepId})
                    MATCH (p:Plan {planId: $planId})
                    FOREACH (_ IN CASE WHEN cs IS NOT NULL THEN [1] ELSE [] END |
                        CREATE (cs)-[:DELEGATED_TO]->(p)
                    )
                """, {"centralStepId": central_step_id, "planId": plan_id})
                logger.info("Linked central step %s -[:DELEGATED_TO]-> team plan %s",
                            central_step_id, plan_id)

            # ── Create Goal and Step nodes ────────────────────────────────
            steps_by_goal: Dict[str, List[Dict[str, Any]]] = {}
            for step in steps:
                steps_by_goal.setdefault(step.get("goal_id", ""), []).append(step)

            step_map: Dict[str, str] = {}

            for goal in goals:
                goal_id = goal.get("id", "")
                self.conn.execute_query("""
                    MATCH (p:Plan {planId: $planId})
                    MERGE (g:Goal {goalId: $goalId, planId: $planId})
                    ON CREATE SET
                        g.description     = $description,
                        g.priority        = $priority,
                        g.requiredWorkers = $requiredWorkers,
                        g.status          = 'pending',
                        g.createdAt       = datetime()
                    MERGE (p)-[:HAS_GOAL]->(g)
                """, {
                    "planId":          plan_id,
                    "goalId":          goal_id,
                    "description":     goal.get("description", ""),
                    "priority":        goal.get("priority", 1),
                    "requiredWorkers": json.dumps(goal.get("required_workers", [])),
                })

                for step in steps_by_goal.get(goal_id, []):
                    step_id = step.get("step_id", "")
                    self.conn.execute_query("""
                        MATCH (g:Goal {goalId: $goalId, planId: $planId})
                        MERGE (st:Step {stepId: $stepId, planId: $planId})
                        ON CREATE SET
                            st.action    = $action,
                            st.worker    = $worker,
                            st.status    = 'pending',
                            st.createdAt = datetime()
                        MERGE (g)-[:HAS_STEP]->(st)
                    """, {
                        "planId": plan_id,
                        "goalId": goal_id,
                        "stepId": step_id,
                        "action": step.get("action", ""),
                        "worker": step.get("worker", ""),
                    })
                    step_map[step_id] = step_id

            logger.info("Stored %s plan %s for session %s: %d goals, %d steps",
                        plan_type, plan_id, session_id, len(goals), len(steps))
            return {"plan_id": plan_id, "step_map": step_map}

        except Exception as e:
            logger.error(f"Error storing plan for session {session_id}: {e}")
            return None
    
    def get_active_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve active plan, reconstructing goals and steps from their nodes.

        Returns:
            Plan dict with goals and steps lists, or None
        """
        try:
            result = self.conn.execute_query("""
                MATCH (:Session {sessionId: $sessionId})-[:HAS_EXECUTION]->
                      (:AgentExecution {agentType: 'a2a_client'})-[:CALLED_AGENT]->
                      (:AgentExecution {agentType: 'a2a_server'})-[:HAS_PLAN]->(p:Plan {status: 'active'})
                OPTIONAL MATCH (p)-[:HAS_GOAL]->(g:Goal)
                OPTIONAL MATCH (g)-[:HAS_STEP]->(st:Step)
                WITH p,
                     collect(DISTINCT {
                         id: g.goalId, description: g.description,
                         priority: g.priority, status: g.status,
                         required_workers: g.requiredWorkers
                     }) AS goals,
                     collect(DISTINCT {
                         step_id: st.stepId, goal_id: g.goalId,
                         action: st.action, worker: st.worker, status: st.status
                     }) AS steps
                RETURN p { .planId, .agentName, .planType, .teamName, .status, .totalGoals } AS plan,
                       goals, steps
                ORDER BY p.createdAt DESC
                LIMIT 1
            """, {"sessionId": session_id})
            if not result:
                return None
            row  = result[0]
            plan = row.get("plan", {})
            if plan:
                plan["goals"] = row.get("goals", [])
                plan["steps"] = row.get("steps", [])
            return plan
        except Exception as e:
            logger.error(f"Error retrieving active plan for session {session_id}: {e}")
            return None
    
    def update_plan_progress(self,
                            session_id: str,
                            plan_id: str,
                            goal_id: str,
                            goal_result: str) -> bool:
        """
        Mark a Goal node and all its Steps with the given result.

        Args:
            session_id:  Session ID
            plan_id:     Plan ID
            goal_id:     Goal.goalId to mark
            goal_result: "completed" | "skipped" | "failed" | "cancelled"
        """
        try:
            # Use the appropriate timestamp field for each terminal status
            if goal_result == "failed":
                timestamp_clause = "g.failedAt = datetime()"
                step_ts_clause   = "st.failedAt = datetime()"
            elif goal_result == "cancelled":
                timestamp_clause = "g.cancelledAt = datetime()"
                step_ts_clause   = "st.cancelledAt = datetime()"
            else:
                timestamp_clause = "g.completedAt = datetime()"
                step_ts_clause   = "st.completedAt = datetime()"

            result = self.conn.execute_query(f"""
                MATCH (p:Plan {{planId: $planId}})
                MATCH (p)-[:HAS_GOAL]->(g:Goal {{goalId: $goalId}})
                SET g.status = $goalResult,
                    {timestamp_clause}
                WITH g
                OPTIONAL MATCH (g)-[:HAS_STEP]->(st:Step)
                SET st.status = $goalResult,
                    {step_ts_clause}
                RETURN g.goalId AS goalId
            """, {
                "sessionId":  session_id,
                "planId":     plan_id,
                "goalId":     goal_id,
                "goalResult": goal_result,
            })
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error updating plan progress for session {session_id}: {e}")
            return False
    
    def cancel_remaining_goals(self, session_id: str, plan_id: str) -> int:
        """
        Mark all Goals (and their Steps) that are still in 'pending' status
        as 'cancelled'. Called when a goal fails so that downstream goals
        are not left as orphaned pending nodes.

        Args:
            session_id: Session ID (for logging)
            plan_id:    Plan ID

        Returns:
            Number of goals cancelled, or 0 on error.
        """
        try:
            result = self.conn.execute_query("""
                MATCH (p:Plan {planId: $planId})
                MATCH (p)-[:HAS_GOAL]->(g:Goal)
                WHERE g.status = 'pending'
                SET g.status      = 'cancelled',
                    g.cancelledAt = datetime()
                WITH g
                OPTIONAL MATCH (g)-[:HAS_STEP]->(st:Step)
                WHERE st.status = 'pending'
                SET st.status      = 'cancelled',
                    st.cancelledAt = datetime()
                RETURN count(DISTINCT g) AS cancelledCount
            """, {
                "planId": plan_id,
            })
            count = result[0].get("cancelledCount", 0) if result else 0
            if count:
                logger.info(
                    "Cancelled %d pending goal(s) in plan %s after failure",
                    count, plan_id,
                )
            return count
        except Exception as e:
            logger.error(f"Error cancelling remaining goals for plan {plan_id}: {e}")
            return 0

    def complete_plan(self, session_id: str, plan_id: str ) -> bool:
        """
        Mark plan as completed.
        
        Args:
            session_id: Session ID
            plan_id: Plan ID
            
        Returns:
            True if successful
        """
        query = """
        MATCH (p:Plan {planId: $planId, status: 'active'})
        SET p.status = 'completed',
            p.completedAt = datetime()
        RETURN p.planId AS planId
        """
        
        try:
            result = self.conn.execute_query( query, {"sessionId": session_id, "planId": plan_id })
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error completing plan with {plan_id} for session {session_id} : {e}")
            return False
    
    def fail_plan(self, session_id: str, plan_id: str) -> bool:
        """
        Mark plan as incomplete due to a goal failure.

        Args:
            session_id: Session ID
            plan_id:    Plan ID

        Returns:
            True if successful
        """
        query = """
        MATCH (p:Plan {planId: $planId})
        WHERE p.status IN ['active', 'running']
        SET p.status     = 'incomplete',
            p.failedAt   = datetime()
        RETURN p.planId AS planId
        """
        try:
            result = self.conn.execute_query(query, {"sessionId": session_id, "planId": plan_id})
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error marking plan {plan_id} incomplete for session {session_id}: {e}")
            return False

    # ==================== Tool Execution Operations ====================
    

    def link_step_to_execution(self, plan_id: str, step_id: str, execution_id: str) -> None:
        """
        Create (Step)-[:EXECUTED_BY]->(AgentExecution).

        Scoped by planId so stepId collisions across sessions are impossible.
        Called by the supervisor immediately after routing to a worker, before
        the worker fires the MCP tool.
        """
        try:
            self.conn.execute_query("""
                MATCH (st:Step {stepId: $stepId, planId: $planId})
                MATCH (e:AgentExecution {executionId: $executionId})
                MERGE (st)-[:EXECUTED_BY]->(e)
            """, {
                "planId":      plan_id,
                "stepId":      step_id,
                "executionId": execution_id,
            })
            logger.debug(f"Linked step {step_id} -> execution {execution_id}")
        except Exception as e:
            logger.warning(f"Failed to link step to execution (non-fatal): {e}")

    def track_tool_execution(self,
                            session_id: str,
                            tool_name: str,
                            input_data: Dict[str, Any],
                            output_data: Optional[Dict[str, Any]] = None,
                            status: str = "success",
                            execution_time_ms: Optional[float] = None,
                            error: Optional[str] = None,
                            execution_id: Optional[str] = None,
                            ) -> Optional[str]:
        """
        Track tool execution and wire it into the plan graph.

        Always creates:
            (ToolExecution) node

        When execution_id provided:
            (AgentExecution)-[:CALLED_TOOL]->(ToolExecution)

        Note: (Session)-[:USED_TOOL] is intentionally omitted — ToolExecution
        is reachable via the Step->AgentExecution->ToolExecution chain.

        Note: (Step)-[:EXECUTED_BY]->(AgentExecution) is created by
        link_step_to_execution(), called from the supervisor before the
        worker fires, scoped by planId to avoid cross-session contamination.
        """
        # Base query — creates ToolExecution node (no Session link needed;
        # reachable via Step->AgentExecution->ToolExecution chain)
        query = """
            MATCH (s:Session {sessionId: $sessionId})
            CREATE (t:ToolExecution {
                toolExecutionId: randomUUID(),
                toolName:        $toolName,
                input:           $input,
                output:          $output,
                status:          $status,
                executionTimeMs: $executionTimeMs,
                error:           $error,
                timestamp:       datetime()
            })
        """

        if execution_id:
            query += """
            WITH t
            OPTIONAL MATCH (e:AgentExecution {executionId: $executionId})
            FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
                CREATE (e)-[:CALLED_TOOL]->(t)
            )
            """

        query += " RETURN t.toolExecutionId AS toolExecutionId"

        params: Dict[str, Any] = {
            "sessionId":      session_id,
            "toolName":       tool_name,
            "input":          json.dumps(input_data or {}),
            "output":         json.dumps(output_data or {}),
            "status":         status,
            "executionTimeMs": execution_time_ms,
            "error":          error,
        }
        if execution_id:
            params["executionId"] = execution_id

        try:
            result = self.conn.execute_query(query, params)
            if result and len(result) > 0:
                return result[0].get("toolExecutionId")
            return None
        except Exception as e:
            logger.error(f"Error tracking tool execution: {e}")
            return None
    

    def link_a2a_client_to_server(self, session_id: str, a2a_task_id: str) -> None:
        """
        Create (a2a_client)-[:CALLED_AGENT]->(a2a_server) by matching on
        shared a2a_task_id stored in AgentExecution.metadata JSON.

        client is matched via Session (only a2a_client has HAS_EXECUTION from Session).
        server is matched directly by agentType + metadata since it has no Session link.
        Works generically across all teams (member_services, claims, provider, etc.)
        """
        try:
            self.conn.execute_query("""
                MATCH (s:Session {sessionId: $sessionId})
                      -[:HAS_EXECUTION]->(client:AgentExecution {agentType: "a2a_client"})
                MATCH (server:AgentExecution {agentType: "a2a_server"})
                WHERE client.metadata CONTAINS $taskId
                  AND server.metadata CONTAINS $taskId
                MERGE (client)-[:CALLED_AGENT]->(server)
            """, {
                "sessionId": session_id,
                "taskId":    a2a_task_id,
            })
            logger.debug(f"Linked a2a_client -> a2a_server for task {a2a_task_id}")
        except Exception as e:
            logger.warning(f"Failed to link a2a client to server (non-fatal): {e}")

    def link_a2a_server_to_plan(self, session_id: str, a2a_task_id: str, plan_id: str) -> None:
        """
        Create (a2a_server)-[:HAS_PLAN]->(Plan).

        server is matched directly by agentType + metadata (no Session link).
        This is the seam between the A2A transport layer and the team execution
        graph. With (a2a_client)-[:CALLED_AGENT]->(a2a_server)-[:HAS_PLAN]->(Plan)
        in place, the full chain from central supervisor down to every tool call
        across every team becomes traversable in a single Cypher query.
        """
        try:
            self.conn.execute_query("""
                MATCH (server:AgentExecution {agentType: "a2a_server"})
                MATCH (plan:Plan {planId: $planId})
                WHERE server.metadata CONTAINS $taskId
                MERGE (server)-[:HAS_PLAN]->(plan)
            """, {
                "planId":  plan_id,
                "taskId":  a2a_task_id,
            })
            logger.debug(f"Linked a2a_server -> plan {plan_id} for task {a2a_task_id}")
        except Exception as e:
            logger.warning(f"Failed to link a2a server to plan (non-fatal): {e}")

    def create_tool_error(
        self,
        tool_execution_id: str,
        error_type: str,
        error_message: str,
    ) -> Optional[str]:
        """
        Attach a ToolError node to a ToolExecution.

        Every error originating within the tool boundary — decorator guards
        (rate limit, circuit breaker, permission denied, pending approval) and
        runtime exceptions inside the tool function — is attached here.
        A ToolExecution is always created first so this node always has a
        parent to link to.

        Creates:
            (ToolExecution)-[:HAD_ERROR]->(ToolError)

        Args:
            tool_execution_id: ToolExecution.toolExecutionId
            error_type:        One of: failed, not_found, rate_limited,
                               rate_limit_exceeded, circuit_breaker_active,
                               tool_permission_denied,
                               resource_permission_denied, pending_approval
            error_message:     Full human-readable error detail

        Returns:
            errorId if created, None on failure
        """
        try:
            result = self.conn.execute_query("""
                MATCH (t:ToolExecution {toolExecutionId: $toolExecutionId})
                CREATE (err:ToolError {
                    errorId:      randomUUID(),
                    errorType:    $errorType,
                    errorMessage: $errorMessage,
                    timestamp:    datetime()
                })
                CREATE (t)-[:HAD_ERROR]->(err)
                RETURN err.errorId AS errorId
            """, {
                "toolExecutionId": tool_execution_id,
                "errorType":       error_type,
                "errorMessage":    error_message,
            })
            if result and len(result) > 0:
                logger.debug(
                    "Created ToolError %s for tool_execution %s (type=%s)",
                    result[0].get("errorId"), tool_execution_id, error_type,
                )
                return result[0].get("errorId")
            return None
        except Exception as e:
            logger.warning(f"Failed to create tool error node (non-fatal): {e}")
            return None

    def create_agent_error(
        self,
        execution_id: str,
        error_type: str,
        error_message: str,
    ) -> Optional[str]:
        """
        Attach an AgentError node to an AgentExecution.

        Used for failures that occur outside the tool boundary entirely:
        worker-level logic errors, LLM invocation failures, state machine
        errors, unhandled exceptions in the agent loop. These have no
        associated ToolExecution.

        Creates:
            (AgentExecution)-[:HAD_ERROR]->(AgentError)

        Sets AgentExecution.status = 'failed'.

        Args:
            execution_id:  AgentExecution.executionId
            error_type:    Short category string, e.g. "llm_error",
                           "state_error", "unhandled_exception"
            error_message: Full human-readable error detail

        Returns:
            errorId if created, None on failure
        """
        try:
            result = self.conn.execute_query("""
                MATCH (e:AgentExecution {executionId: $executionId})
                CREATE (err:AgentError {
                    errorId:      randomUUID(),
                    errorType:    $errorType,
                    errorMessage: $errorMessage,
                    timestamp:    datetime()
                })
                CREATE (e)-[:HAD_ERROR]->(err)
                SET e.status = 'failed'
                RETURN err.errorId AS errorId
            """, {
                "executionId":  execution_id,
                "errorType":    error_type,
                "errorMessage": error_message,
            })
            if result and len(result) > 0:
                logger.debug(
                    "Created AgentError %s for execution %s (type=%s)",
                    result[0].get("errorId"), execution_id, error_type,
                )
                return result[0].get("errorId")
            return None
        except Exception as e:
            logger.warning(f"Failed to create agent error node (non-fatal): {e}")
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
        MATCH (:Session {sessionId: $sessionId})-[:HAS_EXECUTION]->
              (:AgentExecution {agentType: 'a2a_client'})-[:CALLED_AGENT]->
              (:AgentExecution {agentType: 'a2a_server'})-[:HAS_PLAN]->
              (:Plan)-[:HAS_GOAL]->(:Goal)-[:HAS_STEP]->(:Step)
              -[:EXECUTED_BY]->(e:AgentExecution)-[:CALLED_TOOL]->(t:ToolExecution)
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
                "details": json.dumps( details or {} )
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
