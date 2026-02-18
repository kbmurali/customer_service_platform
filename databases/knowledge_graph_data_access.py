"""
Knowledge Graph Data Access Layer
Pure Cypher-based data access for Neo4j Knowledge Graph (KG)

SCHEMA REFERENCE:
  Nodes: Member, Policy, Provider, Claim, PriorAuthorization
  Relationships:
    (Member)-[:HAS_POLICY]->(Policy)
    (Member)-[:FILED_CLAIM]->(Claim)
    (Claim)-[:UNDER_POLICY]->(Policy)
    (Claim)-[:SERVICED_BY]->(Provider)
    (Member)-[:REQUESTED_PA]->(PriorAuthorization)
    (PriorAuthorization)-[:REQUESTED_BY]->(Provider)
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from databases.connections import get_neo4j_kg

logger = logging.getLogger(__name__)


class KnowledgeGraphDataAccess:
    """
    Data access layer for Neo4j Knowledge Graph.
    Provides methods for querying and updating health insurance domain data.
    Uses pure Cypher queries.
    """
    
    def __init__(self):
        """Initialize Knowledge Graph data access."""
        self.conn = get_neo4j_kg()
        self.driver = self.conn.driver if hasattr(self.conn, 'driver') else None
    
    # ==================== Member Operations ====================
    
    def get_member(self, member_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve member information by ID.
        
        Member properties (from schema):
            memberId, firstName, lastName, dateOfBirth, email, phone,
            street, city, state, zipCode, enrollmentDate, status
        
        Args:
            member_id: Member ID
            
        Returns:
            Member data or None if not found
        """
        query = """
        MATCH (m:Member {memberId: $memberId})
        RETURN m {
            .memberId,
            .firstName,
            .lastName,
            .dateOfBirth,
            .email,
            .phone,
            .street,
            .city,
            .state,
            .zipCode,
            .enrollmentDate,
            .status
        } AS member
        """
        
        try:
            result = self.conn.execute_query(query, {"memberId": member_id})
            if result and len(result) > 0:
                return result[0].get("member")
            return None
        except Exception as e:
            logger.error(f"Error retrieving member {member_id}: {e}")
            return None
    
    def get_member_with_policy(self, member_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve member with their insurance policy.
        
        Uses relationship: (Member)-[:HAS_POLICY]->(Policy)
        
        Args:
            member_id: Member ID
            
        Returns:
            Member data with policy information
        """
        query = """
        MATCH (m:Member {memberId: $memberId})
        OPTIONAL MATCH (m)-[:HAS_POLICY]->(p:Policy)
        RETURN m {
            .memberId,
            .firstName,
            .lastName,
            .dateOfBirth,
            .email,
            .phone,
            .street,
            .city,
            .state,
            .zipCode,
            .enrollmentDate,
            .status
        } AS member,
        p {
            .policyId,
            .policyNumber,
            .policyType,
            .planName,
            .planType,
            .effectiveDate,
            .expirationDate,
            .status,
            .premium,
            .deductible,
            .outOfPocketMax
        } AS policy
        """
        
        try:
            result = self.conn.execute_query(query, {"memberId": member_id})
            if result and len(result) > 0:
                data = result[0]
                member = data.get("member")
                if member:
                    member["policy"] = data.get("policy")
                return member
            return None
        except Exception as e:
            logger.error(f"Error retrieving member with policy {member_id}: {e}")
            return None
    
    def search_members(self, 
                      first_name: Optional[str] = None,
                      last_name: Optional[str] = None,
                      date_of_birth: Optional[str] = None,
                      status: Optional[str] = None,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for members by criteria.
        
        Args:
            first_name: First name (partial match)
            last_name: Last name (partial match)
            date_of_birth: Date of birth (exact match)
            status: Member status (ACTIVE, INACTIVE)
            limit: Maximum number of results
            
        Returns:
            List of matching members
        """
        conditions = []
        params = {"limit": limit}
        
        if first_name:
            conditions.append("toLower(m.firstName) CONTAINS toLower($firstName)")
            params["firstName"] = first_name
        
        if last_name:
            conditions.append("toLower(m.lastName) CONTAINS toLower($lastName)")
            params["lastName"] = last_name
        
        if date_of_birth:
            conditions.append("m.dateOfBirth = $dateOfBirth")
            params["dateOfBirth"] = date_of_birth
        
        if status:
            conditions.append("m.status = $status")
            params["status"] = status
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        query = f"""
        MATCH (m:Member)
        WHERE {where_clause}
        RETURN m {{
            .memberId,
            .firstName,
            .lastName,
            .dateOfBirth,
            .email,
            .phone,
            .city,
            .state,
            .status
        }} AS member
        LIMIT $limit
        """
        
        try:
            result = self.conn.execute_query(query, params)
            return [r.get("member") for r in result if r.get("member")]
        except Exception as e:
            logger.error(f"Error searching members: {e}")
            return []
    
    # ==================== Policy Operations ====================
    
    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve policy information by ID.
        
        Policy properties (from schema):
            policyId, policyNumber, policyType, planName, planType,
            effectiveDate, expirationDate, status, premium, deductible, outOfPocketMax
        
        Args:
            policy_id: Policy ID
            
        Returns:
            Policy data or None if not found
        """
        query = """
        MATCH (p:Policy {policyId: $policyId})
        RETURN p {
            .policyId,
            .policyNumber,
            .policyType,
            .planName,
            .planType,
            .effectiveDate,
            .expirationDate,
            .status,
            .premium,
            .deductible,
            .outOfPocketMax
        } AS policy
        """
        
        try:
            result = self.conn.execute_query(query, {"policyId": policy_id})
            if result and len(result) > 0:
                return result[0].get("policy")
            return None
        except Exception as e:
            logger.error(f"Error retrieving policy {policy_id}: {e}")
            return None
    
    def get_member_policies(self, member_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all policies for a member.
        
        Uses relationship: (Member)-[:HAS_POLICY]->(Policy)
        
        Args:
            member_id: Member ID
            
        Returns:
            List of policies
        """
        query = """
        MATCH (m:Member {memberId: $memberId})-[:HAS_POLICY]->(p:Policy)
        RETURN p {
            .policyId,
            .policyNumber,
            .policyType,
            .planName,
            .planType,
            .effectiveDate,
            .expirationDate,
            .status,
            .premium,
            .deductible,
            .outOfPocketMax
        } AS policy
        ORDER BY p.effectiveDate DESC
        """
        
        try:
            result = self.conn.execute_query(query, {"memberId": member_id})
            return [r.get("policy") for r in result if r.get("policy")]
        except Exception as e:
            logger.error(f"Error retrieving policies for member {member_id}: {e}")
            return []
    
    # ==================== Claim Operations ====================
    
    def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve claim information by ID.
        
        Claim properties (from schema):
            claimId, claimNumber, serviceDate, submissionDate, status,
            totalAmount, paidAmount, denialReason, processingDate
        
        Uses relationships:
            (Member)-[:FILED_CLAIM]->(Claim)
            (Claim)-[:UNDER_POLICY]->(Policy)
            (Claim)-[:SERVICED_BY]->(Provider)
        
        Args:
            claim_id: Claim ID
            
        Returns:
            Claim data with related member, policy, and provider info
        """
        query = """
        MATCH (c:Claim {claimId: $claimId})
        OPTIONAL MATCH (m:Member)-[:FILED_CLAIM]->(c)
        OPTIONAL MATCH (c)-[:UNDER_POLICY]->(pol:Policy)
        OPTIONAL MATCH (c)-[:SERVICED_BY]->(prov:Provider)
        RETURN c {
            .claimId,
            .claimNumber,
            .serviceDate,
            .submissionDate,
            .status,
            .totalAmount,
            .paidAmount,
            .denialReason,
            .processingDate
        } AS claim,
        m {.memberId, .firstName, .lastName} AS member,
        pol {.policyId, .policyNumber, .planName} AS policy,
        prov {
            .providerId,
            .providerType,
            .organizationName,
            .firstName,
            .lastName,
            .specialty
        } AS provider
        """
        
        try:
            result = self.conn.execute_query(query, {"claimId": claim_id})
            if result and len(result) > 0:
                data = result[0]
                claim = data.get("claim")
                if claim:
                    claim["member"] = data.get("member")
                    claim["policy"] = data.get("policy")
                    claim["provider"] = data.get("provider")
                return claim
            return None
        except Exception as e:
            logger.error(f"Error retrieving claim {claim_id}: {e}")
            return None
    
    def get_member_claims(self, member_id: str, status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve claims for a member.
        
        Uses relationship: (Member)-[:FILED_CLAIM]->(Claim)
        
        Args:
            member_id: Member ID
            status: Optional claim status filter
            limit: Maximum number of claims
            
        Returns:
            List of claims
        """
        conditions = ["m.memberId = $memberId"]
        params = {"memberId": member_id, "limit": limit}
        
        if status:
            conditions.append("c.status = $status")
            params["status"] = status
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        MATCH (m:Member)-[:FILED_CLAIM]->(c:Claim)
        WHERE {where_clause}
        RETURN c {{
            .claimId,
            .claimNumber,
            .serviceDate,
            .submissionDate,
            .status,
            .totalAmount,
            .paidAmount,
            .processingDate
        }} AS claim
        ORDER BY c.serviceDate DESC
        LIMIT $limit
        """
        
        try:
            result = self.conn.execute_query(query, params)
            return [r.get("claim") for r in result if r.get("claim")]
        except Exception as e:
            logger.error(f"Error retrieving claims for member {member_id}: {e}")
            return []
    
    def get_claim_status(self, claim_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve claim status by claim number.
        
        Args:
            claim_number: Claim number (e.g., CLM-123456)
            
        Returns:
            Claim status data or None if not found
        """
        query = """
        MATCH (c:Claim {claimNumber: $claimNumber})
        RETURN c {
            .claimId,
            .claimNumber,
            .status,
            .submissionDate,
            .processingDate,
            .totalAmount,
            .paidAmount,
            .denialReason
        } AS claim
        """
        
        try:
            result = self.conn.execute_query(query, {"claimNumber": claim_number})
            if result and len(result) > 0:
                return result[0].get("claim")
            return None
        except Exception as e:
            logger.error(f"Error retrieving claim status {claim_number}: {e}")
            return None
    
    def update_member_field(self, member_id: str, field: str, value: str) -> bool:
        """
        Update a single field on a Member node.

        Allowed fields: phone, email, address_street, address_city,
        address_state, address_zip

        Args:
            member_id: Member ID
            field: Property name to update
            value: New value

        Returns:
            True if successful
        """
        ALLOWED = {"phone", "email", "address_street", "address_city",
                   "address_state", "address_zip"}
        if field not in ALLOWED:
            logger.error(f"Field '{field}' is not updatable on Member")
            return False

        # Use parameterised property name via APOC or string-safe SET
        # Neo4j does not support parameterised property keys, so we
        # whitelist above and interpolate safely.
        query = f"""
        MATCH (m:Member {{memberId: $memberId}})
        SET m.`{field}` = $value
        RETURN m.memberId AS memberId
        """

        try:
            result = self.conn.execute_query(
                query, {"memberId": member_id, "value": value}
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error updating member {member_id} field {field}: {e}")
            return False
            
    def update_claim_status(self, claim_id: str, status: str) -> bool:
        """
        Update claim status.
        
        Args:
            claim_id: Claim ID
            status: New status (SUBMITTED, UNDER_REVIEW, APPROVED, DENIED)
            
        Returns:
            True if successful
        """
        query = """
        MATCH (c:Claim {claimId: $claimId})
        SET c.status = $status
        RETURN c.claimId AS claimId
        """
        
        try:
            result = self.conn.execute_query(query, {"claimId": claim_id, "status": status})
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error updating claim status {claim_id}: {e}")
            return False
    
    # ==================== Prior Authorization Operations ====================
    
    def get_prior_authorization(self, pa_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve prior authorization by ID.
        
        PriorAuthorization properties (from schema):
            paId, paNumber, procedureCode, procedureDescription,
            requestDate, status, urgency, approvalDate, expirationDate, denialReason
        
        Uses relationships:
            (Member)-[:REQUESTED_PA]->(PriorAuthorization)
            (PriorAuthorization)-[:REQUESTED_BY]->(Provider)
        
        Args:
            pa_id: Prior authorization ID
            
        Returns:
            PA data with related member and provider info
        """
        query = """
        MATCH (pa:PriorAuthorization {paId: $paId})
        OPTIONAL MATCH (m:Member)-[:REQUESTED_PA]->(pa)
        OPTIONAL MATCH (pa)-[:REQUESTED_BY]->(prov:Provider)
        RETURN pa {
            .paId,
            .paNumber,
            .procedureCode,
            .procedureDescription,
            .requestDate,
            .status,
            .urgency,
            .approvalDate,
            .expirationDate,
            .denialReason
        } AS pa,
        m {.memberId, .firstName, .lastName} AS member,
        prov {
            .providerId,
            .providerType,
            .organizationName,
            .firstName,
            .lastName,
            .specialty
        } AS provider
        """
        
        try:
            result = self.conn.execute_query(query, {"paId": pa_id})
            if result and len(result) > 0:
                data = result[0]
                pa = data.get("pa")
                if pa:
                    pa["member"] = data.get("member")
                    pa["provider"] = data.get("provider")
                return pa
            return None
        except Exception as e:
            logger.error(f"Error retrieving PA {pa_id}: {e}")
            return None
    
    def get_member_prior_authorizations(self, member_id: str, status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve prior authorizations for a member.
        
        Uses relationship: (Member)-[:REQUESTED_PA]->(PriorAuthorization)
        
        Args:
            member_id: Member ID
            status: Optional PA status filter
            limit: Maximum number of PAs
            
        Returns:
            List of prior authorizations
        """
        conditions = ["m.memberId = $memberId"]
        params = {"memberId": member_id, "limit": limit}
        
        if status:
            conditions.append("pa.status = $status")
            params["status"] = status
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        MATCH (m:Member)-[:REQUESTED_PA]->(pa:PriorAuthorization)
        WHERE {where_clause}
        OPTIONAL MATCH (pa)-[:REQUESTED_BY]->(prov:Provider)
        RETURN pa {{
            .paId,
            .paNumber,
            .procedureCode,
            .procedureDescription,
            .requestDate,
            .status,
            .urgency,
            .approvalDate,
            .expirationDate,
            .denialReason
        }} AS pa,
        prov {{
            .providerId,
            .organizationName,
            .firstName,
            .lastName
        }} AS provider
        ORDER BY pa.requestDate DESC
        LIMIT $limit
        """
        
        try:
            result = self.conn.execute_query(query, params)
            pas = []
            for r in result:
                pa = r.get("pa")
                if pa:
                    pa["provider"] = r.get("provider")
                    pas.append(pa)
            return pas
        except Exception as e:
            logger.error(f"Error retrieving PAs for member {member_id}: {e}")
            return []
    
    def update_pa_status(self, pa_id: str, status: str) -> bool:
        """
        Update prior authorization status.
        
        Args:
            pa_id: Prior authorization ID
            status: New status (PENDING, APPROVED, DENIED, EXPIRED)
            
        Returns:
            True if successful
        """
        query = """
        MATCH (pa:PriorAuthorization {paId: $paId})
        SET pa.status = $status
        RETURN pa.paId AS paId
        """
        
        try:
            result = self.conn.execute_query(query, {"paId": pa_id, "status": status})
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error updating PA status {pa_id}: {e}")
            return False
    
    # ==================== Provider Operations ====================
    
    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve provider information by ID.
        
        Provider properties (from schema):
            providerId, npi, providerType, specialty, phone,
            street, city, state, zipCode
            IF ORGANIZATION: organizationName
            IF INDIVIDUAL: firstName, lastName
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Provider data or None if not found
        """
        query = """
        MATCH (p:Provider {providerId: $providerId})
        RETURN p {
            .providerId,
            .npi,
            .providerType,
            .specialty,
            .phone,
            .street,
            .city,
            .state,
            .zipCode,
            .organizationName,
            .firstName,
            .lastName
        } AS provider
        """
        
        try:
            result = self.conn.execute_query(query, {"providerId": provider_id})
            if result and len(result) > 0:
                return result[0].get("provider")
            return None
        except Exception as e:
            logger.error(f"Error retrieving provider {provider_id}: {e}")
            return None
    
    def search_providers(self,
                        specialty: Optional[str] = None,
                        provider_type: Optional[str] = None,
                        city: Optional[str] = None,
                        state: Optional[str] = None,
                        zip_code: Optional[str] = None,
                        limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for providers by criteria.
        
        Only uses properties that exist in the schema:
            specialty, providerType, city, state, zipCode
        
        Args:
            specialty: Provider specialty (partial match)
            provider_type: ORGANIZATION or INDIVIDUAL
            city: City name
            state: State code
            zip_code: ZIP code
            limit: Maximum number of results
            
        Returns:
            List of matching providers
        """
        conditions = []
        params = {"limit": limit}
        
        if specialty:
            conditions.append("toLower(p.specialty) CONTAINS toLower($specialty)")
            params["specialty"] = specialty
        
        if provider_type:
            conditions.append("p.providerType = $providerType")
            params["providerType"] = provider_type
        
        if city:
            conditions.append("toLower(p.city) CONTAINS toLower($city)")
            params["city"] = city
        
        if state:
            conditions.append("p.state = $state")
            params["state"] = state
        
        if zip_code:
            conditions.append("p.zipCode = $zipCode")
            params["zipCode"] = zip_code
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        query = f"""
        MATCH (p:Provider)
        WHERE {where_clause}
        RETURN p {{
            .providerId,
            .npi,
            .providerType,
            .specialty,
            .phone,
            .city,
            .state,
            .zipCode,
            .organizationName,
            .firstName,
            .lastName
        }} AS provider
        LIMIT $limit
        """
        
        try:
            result = self.conn.execute_query(query, params)
            return [r.get("provider") for r in result if r.get("provider")]
        except Exception as e:
            logger.error(f"Error searching providers: {e}")
            return []
    
    # ==================== Coverage/Eligibility Operations ====================
    
    def get_member_coverage(self, member_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve coverage information for a member.
        Coverage is derived from the member's active Policy.
        
        NOTE: There is no separate Benefit or Coverage node in the schema.
        Coverage information comes from the Policy node properties.
        
        Uses relationship: (Member)-[:HAS_POLICY]->(Policy)
        
        Args:
            member_id: Member ID
            
        Returns:
            Policy/coverage data or None if not found
        """
        query = """
        MATCH (m:Member {memberId: $memberId})-[:HAS_POLICY]->(p:Policy)
        WHERE p.status = 'ACTIVE'
        RETURN p {
            .policyId,
            .policyNumber,
            .policyType,
            .planName,
            .planType,
            .effectiveDate,
            .expirationDate,
            .status,
            .premium,
            .deductible,
            .outOfPocketMax
        } AS policy,
        m {.memberId, .firstName, .lastName, .status} AS member
        """
        
        try:
            result = self.conn.execute_query(query, {"memberId": member_id})
            if result and len(result) > 0:
                data = result[0]
                policy = data.get("policy")
                if policy:
                    policy["member"] = data.get("member")
                return policy
            return None
        except Exception as e:
            logger.error(f"Error retrieving coverage for member {member_id}: {e}")
            return None
    
    def check_eligibility(self, member_id: str, service_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Check member eligibility based on member status and active policy.
        
        Eligibility is determined by:
        1. Member status is 'ACTIVE'
        2. Member has a Policy with status 'ACTIVE'
        3. Service date falls within policy effective/expiration dates
        
        Args:
            member_id: Member ID
            service_date: Service date string (YYYY-MM-DD format, defaults to today)
            
        Returns:
            Eligibility status dict
        """
        if not service_date:
            service_date = datetime.now().strftime("%Y-%m-%d")
        
        query = """
        MATCH (m:Member {memberId: $memberId})
        OPTIONAL MATCH (m)-[:HAS_POLICY]->(p:Policy)
        WHERE p.status = 'ACTIVE'
          AND p.effectiveDate <= $serviceDate
          AND p.expirationDate >= $serviceDate
        RETURN m {
            .memberId,
            .firstName,
            .lastName,
            .status
        } AS member,
        p {
            .policyId,
            .policyNumber,
            .planName,
            .planType,
            .effectiveDate,
            .expirationDate
        } AS policy,
        CASE 
            WHEN m.status = 'ACTIVE' AND p IS NOT NULL THEN true
            ELSE false
        END AS isEligible
        """
        
        params = {"memberId": member_id, "serviceDate": service_date}
        
        try:
            result = self.conn.execute_query(query, params)
            if result and len(result) > 0:
                data = result[0]
                return {
                    "member": data.get("member"),
                    "policy": data.get("policy"),
                    "isEligible": data.get("isEligible", False),
                    "serviceDate": service_date
                }
            return {"isEligible": False, "reason": "Member not found"}
        except Exception as e:
            logger.error(f"Error checking eligibility for member {member_id}: {e}")
            return {"isEligible": False, "reason": f"Error: {str(e)}"}
    
    # ==================== Cross-Domain Queries ====================
    
    def get_member_summary(self, member_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a comprehensive summary of a member including policies, claims, and PAs.
        
        Uses all relationships:
            (Member)-[:HAS_POLICY]->(Policy)
            (Member)-[:FILED_CLAIM]->(Claim)
            (Member)-[:REQUESTED_PA]->(PriorAuthorization)
        
        Args:
            member_id: Member ID
            
        Returns:
            Comprehensive member summary
        """
        query = """
        MATCH (m:Member {memberId: $memberId})
        OPTIONAL MATCH (m)-[:HAS_POLICY]->(p:Policy)
        OPTIONAL MATCH (m)-[:FILED_CLAIM]->(c:Claim)
        OPTIONAL MATCH (m)-[:REQUESTED_PA]->(pa:PriorAuthorization)
        RETURN m {
            .memberId,
            .firstName,
            .lastName,
            .dateOfBirth,
            .email,
            .phone,
            .street,
            .city,
            .state,
            .zipCode,
            .enrollmentDate,
            .status
        } AS member,
        collect(DISTINCT p {
            .policyId,
            .policyNumber,
            .planName,
            .planType,
            .status
        }) AS policies,
        collect(DISTINCT c {
            .claimId,
            .claimNumber,
            .status,
            .totalAmount,
            .serviceDate
        }) AS claims,
        collect(DISTINCT pa {
            .paId,
            .paNumber,
            .status,
            .procedureCode,
            .requestDate
        }) AS priorAuthorizations
        """
        
        try:
            result = self.conn.execute_query(query, {"memberId": member_id})
            if result and len(result) > 0:
                data = result[0]
                member = data.get("member")
                if member:
                    member["policies"] = data.get("policies", [])
                    member["claims"] = data.get("claims", [])
                    member["priorAuthorizations"] = data.get("priorAuthorizations", [])
                return member
            return None
        except Exception as e:
            logger.error(f"Error retrieving member summary {member_id}: {e}")
            return None
    
    def get_claim_with_full_context(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Get claim with full context including member, policy, and provider.
        
        Uses all claim-related relationships:
            (Member)-[:FILED_CLAIM]->(Claim)
            (Claim)-[:UNDER_POLICY]->(Policy)
            (Claim)-[:SERVICED_BY]->(Provider)
        
        Args:
            claim_id: Claim ID
            
        Returns:
            Claim with full context
        """
        query = """
        MATCH (c:Claim {claimId: $claimId})
        OPTIONAL MATCH (m:Member)-[:FILED_CLAIM]->(c)
        OPTIONAL MATCH (c)-[:UNDER_POLICY]->(pol:Policy)
        OPTIONAL MATCH (c)-[:SERVICED_BY]->(prov:Provider)
        RETURN c {
            .claimId,
            .claimNumber,
            .serviceDate,
            .submissionDate,
            .status,
            .totalAmount,
            .paidAmount,
            .denialReason,
            .processingDate
        } AS claim,
        m {
            .memberId,
            .firstName,
            .lastName,
            .email,
            .phone,
            .status
        } AS member,
        pol {
            .policyId,
            .policyNumber,
            .planName,
            .planType,
            .deductible,
            .outOfPocketMax
        } AS policy,
        prov {
            .providerId,
            .npi,
            .providerType,
            .specialty,
            .organizationName,
            .firstName,
            .lastName
        } AS provider
        """
        
        try:
            result = self.conn.execute_query(query, {"claimId": claim_id})
            if result and len(result) > 0:
                data = result[0]
                claim = data.get("claim")
                if claim:
                    claim["member"] = data.get("member")
                    claim["policy"] = data.get("policy")
                    claim["provider"] = data.get("provider")
                return claim
            return None
        except Exception as e:
            logger.error(f"Error retrieving claim with context {claim_id}: {e}")
            return None
    
    # ==================== Utility Methods ====================
    
    def execute_custom_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
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
_kg_data_access_instance = None

def get_kg_data_access() -> KnowledgeGraphDataAccess:
    """Get singleton instance of Knowledge Graph Data Access."""
    global _kg_data_access_instance
    if _kg_data_access_instance is None:
        _kg_data_access_instance = KnowledgeGraphDataAccess()
    return _kg_data_access_instance
