# ============================================
# Member Services Tools
# ============================================
import logging
import json
from datetime import datetime

from langchain_core.tools import tool

from databases.knowledge_graph_data_access import get_kg_data_access
from agents.tools_util import require_permissions, track_tool_execution_in_cg, scrub_output
from security.nh3_sanitization import sanitize_text

logger = logging.getLogger(__name__)


@tool
@require_permissions("MEMBER", "READ")
def member_lookup(member_id: str, user_role: str, session_id: str = "default") -> str:
    """
    Look up member information by member ID.

    Args:
        member_id:  The member's unique identifier
        user_role:  The role of the user making the request
        session_id: Session ID for audit and scrubbing

    Returns:
        JSON string with member information
    """
    start_time = datetime.now()

    # Sanitize input
    member_id = sanitize_text(member_id)

    try:
        # Query Neo4j KG via data access layer
        kg_data_access = get_kg_data_access()
        member = kg_data_access.get_member(member_id)

        if not member:
            error = f"Member not found: {member_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(session_id, "member_lookup", {"member_id": member_id}, status="not_found", execution_time_ms=execution_time, error=error)
            return json.dumps({"error": error})

        # Scrub PII/PHI from output
        output = json.dumps(member, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        # Track successful execution in Context Graph
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(session_id, "member_lookup", {"member_id": member_id}, status="success", execution_time_ms=execution_time)

        return scrubbed_output

    except Exception as e:
        logger.error(f"member_lookup failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(session_id, "member_lookup", {"member_id": member_id}, status="failed", execution_time_ms=execution_time, error=error)
        return json.dumps({"error": error})


@tool
@require_permissions("MEMBER", "READ")
def check_eligibility(member_id: str, service_date: str, user_role: str, session_id: str = "default") -> str:
    """
    Check member eligibility for services on a specific date.

    Args:
        member_id:    The member's unique identifier
        service_date: Date of service (YYYY-MM-DD format)
        user_role:    The role of the user making the request
        session_id:   Session ID for audit and scrubbing

    Returns:
        JSON string with eligibility information
    """
    start_time = datetime.now()

    # Sanitize inputs
    member_id    = sanitize_text(member_id)
    service_date = sanitize_text(service_date)

    try:
        # Use KG data access layer to check eligibility
        kg_data_access = get_kg_data_access()
        eligibility = kg_data_access.check_eligibility(member_id, service_date)

        output = json.dumps(eligibility, indent=2)

        # Scrub PII/PHI from output
        scrubbed_output = scrub_output(output, session_id)

        # Track successful execution in Context Graph
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = {"member_id": member_id, "service_date": service_date}
        track_tool_execution_in_cg(session_id, "check_eligibility", inputs, status="success", execution_time_ms=execution_time)

        return scrubbed_output

    except Exception as e:
        logger.error(f"check_eligibility failed: {e}")
        error = str(e)
        inputs = {"member_id": member_id, "service_date": service_date}
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(session_id, "check_eligibility", inputs, status="failed", execution_time_ms=execution_time, error=error)
        return json.dumps({"error": error})


@tool
@require_permissions("MEMBER", "READ")
def coverage_lookup(member_id: str, procedure_code: str, user_role: str, session_id: str = "default") -> str:
    """
    Look up coverage details for a member and optionally a specific procedure.

    Coverage is derived from the member's active Policy node.

    Uses relationship: (Member)-[:HAS_POLICY]->(Policy)

    Args:
        member_id:      The member's unique identifier
        procedure_code: CPT code of the procedure
        user_role:      The role of the user making the request
        session_id:     Session ID for audit and scrubbing

    Returns:
        JSON string with coverage information
    """
    start_time = datetime.now()

    # Sanitize inputs
    member_id      = sanitize_text(member_id)
    procedure_code = sanitize_text(procedure_code)

    try:
        # Get member's active policy from KG
        # Schema: (Member)-[:HAS_POLICY]->(Policy)
        kg_data_access = get_kg_data_access()
        coverage = kg_data_access.get_member_coverage(member_id)

        if not coverage:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error = "No active policy found for member"
            track_tool_execution_in_cg(session_id, "coverage_lookup", {"member_id": member_id}, status="not_found", execution_time_ms=execution_time, error=error)
            return json.dumps({"covered": False, "reason": error})

        # Add procedure code context to the response
        coverage["requestedProcedureCode"] = procedure_code

        output = json.dumps(coverage, indent=2)

        # Scrub PII/PHI from output
        scrubbed_output = scrub_output(output, session_id)

        # Track successful execution in Context Graph
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(session_id, "coverage_lookup", {"member_id": member_id}, status="success", execution_time_ms=execution_time)

        return scrubbed_output

    except Exception as e:
        logger.error(f"coverage_lookup failed: {e}")
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        error = str(e)
        track_tool_execution_in_cg(session_id, "coverage_lookup", {"member_id": member_id}, status="failed", execution_time_ms=execution_time, error=error)
        return json.dumps({"error": error})