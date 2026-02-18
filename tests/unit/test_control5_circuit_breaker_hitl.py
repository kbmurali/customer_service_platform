"""
Unit tests for control 5: circuit breaker and human-in-the-loop

Tests cover:
- Approval workflow (security.approval_workflow - MySQL + Redis)
- Circuit breaker / kill switch (security.approval_workflow)
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import json

from security.approval_workflow import (
    ApprovalWorkflow,
    AgentAction,
    ImpactLevel,
    ApprovalStatus,
    CircuitBreakerError,
)

# ============================================
# Approval Workflow Tests (MySQL + Redis)
# ============================================

class TestApprovalWorkflow:
    """Test human-in-the-loop approval workflow"""

    @pytest.fixture
    def mock_workflow(self):
        """Create approval workflow with mocked MySQL and Redis"""
        with patch('security.approval_workflow.get_mysql') as mock_mysql, \
             patch('security.approval_workflow.redis.Redis') as mock_redis_cls, \
             patch('security.approval_workflow.get_settings') as mock_settings:

            mock_db = Mock()
            mock_db.execute_query.return_value = [{"1": 1}]
            mock_db.execute_update.return_value = None
            mock_mysql.return_value = mock_db

            mock_redis = Mock()
            mock_redis.get.return_value = None
            mock_redis.set.return_value = True
            mock_redis.setex.return_value = True
            mock_redis.lpush.return_value = 1
            mock_redis.ltrim.return_value = True
            mock_redis.delete.return_value = 1
            mock_redis_cls.return_value = mock_redis

            settings_obj = Mock()
            settings_obj.REDIS_HOST = "localhost"
            settings_obj.REDIS_PORT = 6379
            mock_settings.return_value = settings_obj

            workflow = ApprovalWorkflow()

            yield {
                "workflow": workflow,
                "mysql": mock_db,
                "redis": mock_redis,
            }

    # ------------------------------------------------------------------
    # Impact Classification
    # ------------------------------------------------------------------

    def test_classify_low_impact(self, mock_workflow):
        """Test classification of low-impact tools"""
        wf = mock_workflow["workflow"]
        assert wf._classify_impact("lookup_member") == ImpactLevel.LOW
        assert wf._classify_impact("check_eligibility") == ImpactLevel.LOW
        assert wf._classify_impact("search_policy_info") == ImpactLevel.LOW

    def test_classify_medium_impact(self, mock_workflow):
        """Test classification of medium-impact tools (update_* prefix)"""
        wf = mock_workflow["workflow"]
        assert wf._classify_impact("update_member_address") == ImpactLevel.MEDIUM
        assert wf._classify_impact("delete_note") == ImpactLevel.MEDIUM

    def test_classify_high_impact(self, mock_workflow):
        """Test classification of high-impact tools"""
        wf = mock_workflow["workflow"]
        for tool in ApprovalWorkflow.HIGH_IMPACT_TOOLS:
            assert wf._classify_impact(tool) == ImpactLevel.HIGH

    def test_classify_critical_impact(self, mock_workflow):
        """Test classification of critical-impact tools"""
        wf = mock_workflow["workflow"]
        for tool in ApprovalWorkflow.CRITICAL_IMPACT_TOOLS:
            assert wf._classify_impact(tool) == ImpactLevel.CRITICAL

    # ------------------------------------------------------------------
    # Approval Request Creation
    # ------------------------------------------------------------------

    def test_create_approval_request(self, mock_workflow):
        """Test creating an approval request stores in MySQL and Redis"""
        wf = mock_workflow["workflow"]
        action = AgentAction(
            action_id="act_001",
            agent_id="member_services",
            tool_name="update_claim_status",
            parameters={"claim_id": "C123", "new_status": "approved"},
        )

        request = wf._create_approval_request(
            action, ImpactLevel.HIGH, "user123"
        )

        assert request.request_id.startswith("approval_act_001_")
        assert request.status == ApprovalStatus.PENDING
        assert request.impact_level == ImpactLevel.HIGH

        # Verify MySQL insert
        mock_workflow["mysql"].execute_update.assert_called()
        insert_call = mock_workflow["mysql"].execute_update.call_args
        assert "INSERT INTO approval_requests" in insert_call[0][0]

        # Verify Redis queue push
        mock_workflow["redis"].lpush.assert_called_once()
        mock_workflow["redis"].setex.assert_called_once()

    # ------------------------------------------------------------------
    # Reviewer Actions
    # ------------------------------------------------------------------

    def test_approve_request(self, mock_workflow):
        """Test approving an approval request"""
        wf = mock_workflow["workflow"]

        wf.approve_request("req_001", "supervisor_1", "Verified and approved")

        # Verify MySQL update
        mock_workflow["mysql"].execute_update.assert_called()
        update_call = mock_workflow["mysql"].execute_update.call_args
        assert "UPDATE approval_requests" in update_call[0][0]
        assert ApprovalStatus.APPROVED.value in update_call[0][1]

        # Verify Redis notification
        mock_workflow["redis"].setex.assert_called_once()
        redis_call = mock_workflow["redis"].setex.call_args
        assert "approval_status:req_001" in redis_call[0]
        status_data = json.loads(redis_call[0][2])
        assert status_data["approved"] is True

    def test_deny_request(self, mock_workflow):
        """Test denying an approval request"""
        wf = mock_workflow["workflow"]

        wf.deny_request("req_001", "supervisor_1", "Insufficient justification")

        # Verify MySQL update
        mock_workflow["mysql"].execute_update.assert_called()
        update_call = mock_workflow["mysql"].execute_update.call_args
        assert ApprovalStatus.DENIED.value in update_call[0][1]

        # Verify Redis notification
        redis_call = mock_workflow["redis"].setex.call_args
        status_data = json.loads(redis_call[0][2])
        assert status_data["approved"] is False

    # ------------------------------------------------------------------
    # Execute With Approval (async)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_low_impact_auto_executes(self, mock_workflow):
        """Test that low-impact actions auto-execute without approval"""
        wf = mock_workflow["workflow"]
        action = AgentAction(
            action_id="act_001",
            agent_id="member_services",
            tool_name="lookup_member",
            parameters={"member_id": "M123"},
        )

        result = await wf.execute_with_approval(action, "user123")

        assert result["status"] == "success"
        # No approval request should be created
        mock_workflow["redis"].lpush.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_medium_impact_logs_and_executes(self, mock_workflow):
        """Test that medium-impact actions log and auto-execute"""
        wf = mock_workflow["workflow"]
        action = AgentAction(
            action_id="act_002",
            agent_id="member_services",
            tool_name="update_member_address",
            parameters={"member_id": "M123", "address": "new"},
        )

        result = await wf.execute_with_approval(action, "user123")

        assert result["status"] == "success"
        # Should log to MySQL but not push to Redis queue
        mock_workflow["mysql"].execute_update.assert_called()

    @pytest.mark.asyncio
    async def test_execute_circuit_breaker_active_raises(self, mock_workflow):
        """Test that active circuit breaker raises error"""
        wf = mock_workflow["workflow"]
        mock_workflow["redis"].get.side_effect = lambda k: (
            "1" if k == "circuit_breaker:active" else "Emergency"
        )

        action = AgentAction(
            action_id="act_003",
            agent_id="member_services",
            tool_name="lookup_member",
            parameters={"member_id": "M123"},
        )

        with pytest.raises(CircuitBreakerError):
            await wf.execute_with_approval(action, "user123")

    # ------------------------------------------------------------------
    # Pending Approvals Query
    # ------------------------------------------------------------------

    def test_get_pending_approvals(self, mock_workflow):
        """Test querying pending approvals from MySQL"""
        wf = mock_workflow["workflow"]
        mock_workflow["mysql"].execute_query.return_value = [
            {"request_id": "req_001", "tool_name": "update_claim_status"},
            {"request_id": "req_002", "tool_name": "approve_prior_auth"},
        ]

        result = wf.get_pending_approvals(limit=10)

        assert len(result) == 2
        mock_workflow["mysql"].execute_query.assert_called()
        query_call = mock_workflow["mysql"].execute_query.call_args
        assert "PENDING" in query_call[0][1]


# ============================================
# Circuit Breaker / Kill Switch Tests
# ============================================

class TestCircuitBreaker:
    """Test circuit breaker / kill switch functionality"""

    @pytest.fixture
    def mock_workflow(self):
        """Create approval workflow with mocked MySQL and Redis"""
        with patch('security.approval_workflow.get_mysql') as mock_mysql, \
             patch('security.approval_workflow.redis.Redis') as mock_redis_cls, \
             patch('security.approval_workflow.get_settings') as mock_settings:

            mock_db = Mock()
            mock_db.execute_query.return_value = [{"1": 1}]
            mock_db.execute_update.return_value = None
            mock_mysql.return_value = mock_db

            mock_redis = Mock()
            mock_redis.get.return_value = None
            mock_redis.set.return_value = True
            mock_redis.delete.return_value = 1
            mock_redis.lpush.return_value = 1
            mock_redis.ltrim.return_value = True
            mock_redis_cls.return_value = mock_redis

            settings_obj = Mock()
            settings_obj.REDIS_HOST = "localhost"
            settings_obj.REDIS_PORT = 6379
            mock_settings.return_value = settings_obj

            workflow = ApprovalWorkflow()

            yield {
                "workflow": workflow,
                "mysql": mock_db,
                "redis": mock_redis,
            }

    def test_circuit_breaker_inactive_by_default(self, mock_workflow):
        """Test circuit breaker is inactive by default"""
        wf = mock_workflow["workflow"]
        mock_workflow["redis"].get.return_value = None

        assert wf._is_circuit_breaker_active() is False

    def test_activate_circuit_breaker(self, mock_workflow):
        """Test activating the circuit breaker"""
        wf = mock_workflow["workflow"]

        wf.activate_circuit_breaker(
            reason="Anomalous activity detected",
            activated_by="supervisor_1"
        )

        # Verify Redis flags set
        redis_calls = mock_workflow["redis"].set.call_args_list
        keys_set = [c[0][0] for c in redis_calls]
        assert "circuit_breaker:active" in keys_set
        assert "circuit_breaker:reason" in keys_set
        assert "circuit_breaker:activated_by" in keys_set
        assert "circuit_breaker:timestamp" in keys_set

        # Verify MySQL audit log
        mock_workflow["mysql"].execute_update.assert_called()
        insert_call = mock_workflow["mysql"].execute_update.call_args
        assert "circuit_breaker_events" in insert_call[0][0]
        assert "ACTIVATED" in insert_call[0][1]

        # Verify alert sent
        mock_workflow["redis"].lpush.assert_called()

    def test_deactivate_circuit_breaker(self, mock_workflow):
        """Test deactivating the circuit breaker"""
        wf = mock_workflow["workflow"]

        wf.deactivate_circuit_breaker(
            deactivated_by="supervisor_1",
            rationale="Issue resolved"
        )

        # Verify Redis flags deleted
        delete_calls = mock_workflow["redis"].delete.call_args_list
        keys_deleted = [c[0][0] for c in delete_calls]
        assert "circuit_breaker:active" in keys_deleted
        assert "circuit_breaker:reason" in keys_deleted
        assert "circuit_breaker:activated_by" in keys_deleted
        assert "circuit_breaker:timestamp" in keys_deleted

        # Verify MySQL audit log
        mock_workflow["mysql"].execute_update.assert_called()
        insert_call = mock_workflow["mysql"].execute_update.call_args
        assert "DEACTIVATED" in insert_call[0][1]

    def test_circuit_breaker_active_check(self, mock_workflow):
        """Test checking circuit breaker active state via private method"""
        wf = mock_workflow["workflow"]
        mock_workflow["redis"].get.return_value = "1"

        assert wf._is_circuit_breaker_active() is True

    def test_public_is_circuit_breaker_active_delegates(self, mock_workflow):
        """Test public is_circuit_breaker_active wraps private method"""
        wf = mock_workflow["workflow"]

        # Inactive
        mock_workflow["redis"].get.return_value = None
        assert wf.is_circuit_breaker_active() is False

        # Active
        mock_workflow["redis"].get.return_value = "1"
        assert wf.is_circuit_breaker_active() is True

    def test_circuit_breaker_history(self, mock_workflow):
        """Test querying circuit breaker event history"""
        wf = mock_workflow["workflow"]
        mock_workflow["mysql"].execute_query.return_value = [
            {
                "event_id": "evt_001",
                "event_type": "ACTIVATED",
                "reason": "Test",
                "triggered_by": "supervisor_1",
                "triggered_at": datetime.utcnow(),
                "metadata": json.dumps({"source": "manual"}),
            }
        ]

        result = wf.get_circuit_breaker_history(limit=10)

        assert len(result) == 1
        assert result[0]["event_type"] == "ACTIVATED"

    def test_send_alert(self, mock_workflow):
        """Test alert sending to Redis"""
        wf = mock_workflow["workflow"]

        wf._send_alert("TEST ALERT", "Test message", "system")

        mock_workflow["redis"].lpush.assert_called_once()
        mock_workflow["redis"].ltrim.assert_called_once()

        # Verify alert content
        alert_json = mock_workflow["redis"].lpush.call_args[0][1]
        alert = json.loads(alert_json)
        assert alert["title"] == "TEST ALERT"
        assert alert["message"] == "Test message"
