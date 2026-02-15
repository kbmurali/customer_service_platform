"""
GraphQL to Cypher query translator for Neo4j
Uses neo4j-graphql-py for deterministic translation
"""
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from graphql import build_schema, GraphQLSchema
from neo4j_graphql_py import neo4j_graphql, make_executable_schema

from databases.connections import Neo4jKGConnection, Neo4jCGConnection

logger = logging.getLogger(__name__)


class GraphQLTranslator:
    """Translates GraphQL queries to Cypher and executes them"""
    
    def __init__(self, schema_file: str, connection: Neo4jKGConnection):
        """
        Initialize translator with GraphQL schema
        
        Args:
            schema_file: Path to GraphQL schema file
            connection: Neo4j connection instance
        """
        self.connection = connection
        self.schema_file = schema_file
        self.schema: Optional[GraphQLSchema] = None
        self._load_schema()
    
    def _load_schema(self):
        """Load GraphQL schema from file"""
        try:
            schema_path = Path(self.schema_file)
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {self.schema_file}")
            
            with open(schema_path, 'r') as f:
                schema_content = f.read()
            
            # Build GraphQL schema
            self.schema = build_schema(schema_content)
            logger.info(f"Loaded GraphQL schema from {self.schema_file}")
        except Exception as e:
            logger.error(f"Failed to load GraphQL schema: {e}")
            raise
    
    def translate_and_execute(
        self,
        graphql_query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate GraphQL query to Cypher and execute
        
        Args:
            graphql_query: GraphQL query string
            variables: Query variables
            operation_name: Operation name for multi-operation queries
        
        Returns:
            Query results as dictionary
        """
        try:
            # Create resolver context with Neo4j driver
            context = {
                'driver': self.connection.connect(),
                'database': self.connection.config['database']
            }
            
            # Use neo4j-graphql-py to translate and execute
            # This will automatically convert GraphQL to Cypher
            result = neo4j_graphql(
                obj=None,
                context=context,
                info=None,  # Will be populated by the library
                query=graphql_query,
                variables=variables or {},
                operation_name=operation_name
            )
            
            logger.debug(f"Executed GraphQL query: {graphql_query[:100]}...")
            return result
        
        except Exception as e:
            logger.error(f"GraphQL query execution failed: {e}")
            raise
    
    def execute_cypher_directly(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query directly (bypass GraphQL)
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
        
        Returns:
            Query results as list of dictionaries
        """
        try:
            results = self.connection.execute_query(cypher_query, parameters)
            logger.debug(f"Executed Cypher query: {cypher_query[:100]}...")
            return results
        except Exception as e:
            logger.error(f"Cypher query execution failed: {e}")
            raise


class KnowledgeGraphTranslator(GraphQLTranslator):
    """GraphQL translator for Knowledge Graph"""
    
    def __init__(self, connection: Neo4jKGConnection):
        schema_file = "databases/neo4j_kg_schema.graphql"
        super().__init__(schema_file, connection)
    
    def get_member(self, member_id: str) -> Optional[Dict[str, Any]]:
        """Get member by ID using GraphQL"""
        query = """
        query GetMember($memberId: ID!) {
            member(memberId: $memberId) {
                memberId
                firstName
                lastName
                dateOfBirth
                email
                phone
                status
                enrollmentDate
                totalClaims
            }
        }
        """
        result = self.translate_and_execute(query, {"memberId": member_id})
        return result.get("data", {}).get("member")
    
    def get_member_policies(self, member_id: str) -> List[Dict[str, Any]]:
        """Get member's policies using GraphQL"""
        query = """
        query GetMemberPolicies($memberId: ID!) {
            member(memberId: $memberId) {
                policies {
                    policyId
                    policyNumber
                    planName
                    planType
                    status
                    effectiveDate
                    expirationDate
                    premium
                    deductible
                    remainingDeductible
                }
            }
        }
        """
        result = self.translate_and_execute(query, {"memberId": member_id})
        member = result.get("data", {}).get("member", {})
        return member.get("policies", [])
    
    def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Get claim by ID using GraphQL"""
        query = """
        query GetClaim($claimId: ID!) {
            claim(claimId: $claimId) {
                claimId
                claimNumber
                serviceDate
                submissionDate
                status
                totalAmount
                paidAmount
                denialReason
                member {
                    memberId
                    firstName
                    lastName
                }
                provider {
                    providerId
                    organizationName
                    specialty
                }
                diagnoses {
                    icdCode
                    description
                }
                procedures {
                    cptCode
                    description
                }
            }
        }
        """
        result = self.translate_and_execute(query, {"claimId": claim_id})
        return result.get("data", {}).get("claim")
    
    def get_prior_authorization(self, pa_id: str) -> Optional[Dict[str, Any]]:
        """Get prior authorization by ID using GraphQL"""
        query = """
        query GetPA($paId: ID!) {
            priorAuthorization(paId: $paId) {
                paId
                paNumber
                requestDate
                status
                approvalDate
                expirationDate
                denialReason
                urgency
                approvalProbability
                member {
                    memberId
                    firstName
                    lastName
                }
                procedure {
                    cptCode
                    description
                    requiresPriorAuth
                }
                provider {
                    providerId
                    organizationName
                }
            }
        }
        """
        result = self.translate_and_execute(query, {"paId": pa_id})
        return result.get("data", {}).get("priorAuthorization")
    
    def search_providers(
        self,
        specialty: Optional[str] = None,
        network_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search providers using GraphQL"""
        query = """
        query SearchProviders($specialty: String, $networkId: ID, $limit: Int) {
            providers(specialty: $specialty, networkId: $networkId, first: $limit) {
                providerId
                npi
                organizationName
                specialty
                phone
                claimVolume
                averageClaimAmount
            }
        }
        """
        variables = {
            "specialty": specialty,
            "networkId": network_id,
            "limit": limit
        }
        result = self.translate_and_execute(query, variables)
        return result.get("data", {}).get("providers", [])


class ContextGraphTranslator(GraphQLTranslator):
    """GraphQL translator for Context Graph"""
    
    def __init__(self, connection: Neo4jCGConnection):
        schema_file = "databases/neo4j_cg_schema.graphql"
        super().__init__(schema_file, connection)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation session by ID"""
        query = """
        query GetSession($sessionId: ID!) {
            session(sessionId: $sessionId) {
                sessionId
                userId
                userRole
                memberId
                startTime
                endTime
                status
                channel
                duration
                interactionCount
            }
        }
        """
        result = self.translate_and_execute(query, {"sessionId": session_id})
        return result.get("data", {}).get("session")
    
    def get_agent_executions(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Get agent executions for a session"""
        query = """
        query GetAgentExecutions($sessionId: ID!) {
            session(sessionId: $sessionId) {
                agentExecutions {
                    executionId
                    agentName
                    agentType
                    startTime
                    endTime
                    status
                    executionTime
                    toolCallCount
                }
            }
        }
        """
        result = self.translate_and_execute(query, {"sessionId": session_id})
        session = result.get("data", {}).get("session", {})
        return session.get("agentExecutions", [])
    
    def get_security_events(
        self,
        severity: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent security events"""
        query = """
        query GetSecurityEvents($severity: String, $limit: Int) {
            securityEvents(severity: $severity, first: $limit) {
                eventId
                timestamp
                eventType
                severity
                userId
                resourceType
                details
                age
            }
        }
        """
        result = self.translate_and_execute(
            query,
            {"severity": severity, "limit": limit}
        )
        return result.get("data", {}).get("securityEvents", [])


# Factory functions
def create_kg_translator(connection: Neo4jKGConnection) -> KnowledgeGraphTranslator:
    """Create Knowledge Graph translator"""
    return KnowledgeGraphTranslator(connection)


def create_cg_translator(connection: Neo4jCGConnection) -> ContextGraphTranslator:
    """Create Context Graph translator"""
    return ContextGraphTranslator(connection)
