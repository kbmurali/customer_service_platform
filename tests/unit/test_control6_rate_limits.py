"""
Unit tests for Control 6: Rate Limits

Tests cover:
- Rate limiting (agents.security.RateLimiter - MySQL-based)
"""

import pytest
from unittest.mock import Mock, patch
import json

from agents.security import RBACService
from agents.tools_util import check_rate_limit_for_tool

from agents.security import (
    RateLimiter,
    RateLimitError,
)


# ============================================
# Rate Limiter Tests (MySQL-based)
# ============================================

class TestRateLimiter:
    """Test rate limiting functionality (MySQL sliding-window)"""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter with mocked MySQL"""
        with patch('agents.security.get_mysql') as mock_mysql:
            limiter = RateLimiter()
            limiter.mysql = Mock()
            return limiter

    def test_check_rate_limit_allowed(self, rate_limiter):
        """Test rate limit check when under limit"""
        rate_limiter.mysql.execute_query.return_value = [{"total_count": 5}]

        result = rate_limiter.check_rate_limit(
            user_id="user123",
            resource_type="TOOL",
            resource_name="lookup_member",
            limit_per_minute=30
        )

        assert result is True

    def test_check_rate_limit_exceeded(self, rate_limiter):
        """Test rate limit check when limit exceeded"""
        rate_limiter.mysql.execute_query.return_value = [{"total_count": 30}]

        with pytest.raises(RateLimitError):
            rate_limiter.check_rate_limit(
                user_id="user123",
                resource_type="TOOL",
                resource_name="lookup_member",
                limit_per_minute=30
            )

    def test_check_rate_limit_first_request(self, rate_limiter):
        """Test rate limit check for first request"""
        rate_limiter.mysql.execute_query.return_value = [{"total_count": 0}]

        result = rate_limiter.check_rate_limit(
            user_id="user123",
            resource_type="TOOL",
            resource_name="lookup_member",
            limit_per_minute=30
        )

        assert result is True

    def test_rate_limit_by_role(self, rate_limiter):
        """Test role-based rate limiting"""
        role_limits = {
            "CSR_TIER1": 30,
            "CSR_TIER2": 60,
            "CSR_SUPERVISOR": 120,
        }

        for role, limit in role_limits.items():
            rate_limiter.mysql.execute_query.return_value = [{"total_count": 5}]

            result = rate_limiter.check_rate_limit(
                user_id="user123",
                resource_type="TOOL",
                resource_name="lookup_member",
                limit_per_minute=limit
            )

            assert result is True

    def test_rate_limit_database_error_fail_open(self, rate_limiter):
        """Test rate limit when database error occurs (fail-open)"""
        rate_limiter.mysql.execute_query.side_effect = Exception("DB error")

        # Should allow request on error (fail-open for availability)
        result = rate_limiter.check_rate_limit(
            user_id="user123",
            resource_type="TOOL",
            resource_name="lookup_member",
            limit_per_minute=30
        )

        assert result is True
        
# ============================================
# Test: check_rate_limit_for_tool helper
# ============================================

class TestCheckRateLimitForTool:
    """Test the check_rate_limit_for_tool helper used by all tools.

    This helper combines RBACService.get_tool_rate_limit() with
    RateLimiter.check_rate_limit() to enforce Control 6 at the
    tool execution layer.
    """

    def test_rate_limit_within_limits_returns_none(self):
        """When within limits, helper returns None (no error)."""
        mock_rbac = Mock()
        mock_rbac.get_tool_rate_limit.return_value = 30

        mock_rl = Mock()
        mock_rl.check_rate_limit.return_value = True

        with patch('agents.tools_util.rbac_service', mock_rbac), \
             patch('agents.tools_util.rate_limiter', mock_rl), \
             patch('agents.tools_util.rate_limit_checks'), \
             patch('agents.tools_util.track_rate_limit_exceeded'):
            result = check_rate_limit_for_tool(
                tool_name="member_lookup",
                user_id="user1",
                user_role="CSR_TIER1",
                session_id="sess1"
            )
            assert result is None
            mock_rl.check_rate_limit.assert_called_once()

    def test_rate_limit_exceeded_returns_error_json(self):
        mock_rbac = Mock()
        mock_rbac.get_tool_rate_limit.return_value = 30

        mock_rl = Mock()
        mock_rl.check_rate_limit.side_effect = RateLimitError("Rate limit exceeded")

        with patch('agents.tools_util.rbac_service', mock_rbac), \
             patch('agents.tools_util.rate_limiter', mock_rl), \
             patch('agents.tools_util.rate_limit_checks'), \
             patch('agents.tools_util.track_rate_limit_exceeded'), \
             patch('agents.tools_util.track_tool_execution_in_cg'):
            result = check_rate_limit_for_tool(
                tool_name="member_lookup",
                user_id="user1",
                user_role="CSR_TIER1",
                session_id="sess1"
            )
            assert result is not None
            result_dict = json.loads(result)
            assert result_dict["rate_limited"] is True
            assert "member_lookup" in result_dict["error"]

    def test_rate_limit_zero_returns_none(self):
        mock_rbac = Mock()
        mock_rbac.get_tool_rate_limit.return_value = 0

        mock_rl = Mock()

        with patch('agents.tools_util.rbac_service', mock_rbac), \
             patch('agents.tools_util.rate_limiter', mock_rl), \
             patch('agents.tools_util.rate_limit_checks'), \
             patch('agents.tools_util.track_rate_limit_exceeded'):
            result = check_rate_limit_for_tool(
                tool_name="member_lookup",
                user_id="user1",
                user_role="CSR_READONLY",
                session_id="sess1"
            )
            assert result is None
            mock_rl.check_rate_limit.assert_not_called()

    def test_rate_limit_db_error_fails_open(self):
        mock_rbac = Mock()
        mock_rbac.get_tool_rate_limit.side_effect = Exception("DB down")

        with patch('agents.tools_util.rbac_service', mock_rbac), \
             patch('agents.tools_util.rate_limiter', Mock()), \
             patch('agents.tools_util.rate_limit_checks'), \
             patch('agents.tools_util.track_rate_limit_exceeded'):
            result = check_rate_limit_for_tool(
                tool_name="member_lookup",
                user_id="user1",
                user_role="CSR_TIER1",
                session_id="sess1"
            )
            assert result is None

    def test_rate_limit_per_role_differentiation(self):
        role_limits = {
            "CSR_TIER1": 30,
            "CSR_TIER2": 60,
            "CSR_SUPERVISOR": 120,
        }

        for role, expected_limit in role_limits.items():
            mock_rbac = Mock()
            mock_rbac.get_tool_rate_limit.return_value = expected_limit

            mock_rl = Mock()
            mock_rl.check_rate_limit.return_value = True

            with patch('agents.tools_util.rbac_service', mock_rbac), \
                 patch('agents.tools_util.rate_limiter', mock_rl), \
                 patch('agents.tools_util.rate_limit_checks'), \
                 patch('agents.tools_util.track_rate_limit_exceeded'):
                check_rate_limit_for_tool(
                    tool_name="member_lookup",
                    user_id="user1",
                    user_role=role,
                    session_id="sess1"
                )
                # Verify the rate limiter was called with the correct limit
                call_args = mock_rl.check_rate_limit.call_args
                assert call_args[1]["limit_per_minute"] == expected_limit


# ============================================
# Test: get_tool_rate_limit in RBACService
# ============================================

class TestGetToolRateLimit:
    """Test RBACService.get_tool_rate_limit() method."""

    @pytest.fixture
    def rbac_service(self):
        
        with patch('agents.security.get_mysql') as mock_mysql:
            svc = RBACService()
            svc.mysql = Mock()
            return svc

    def test_returns_rate_limit_from_db(self, rbac_service):
        """Should return rate_limit_per_minute from tool_permissions."""
        rbac_service.mysql.execute_query.return_value = [
            {"rate_limit_per_minute": 60}
        ]
        result = rbac_service.get_tool_rate_limit("CSR_TIER2", "member_lookup")
        assert result == 60

    def test_returns_zero_for_unpermitted_tool(self, rbac_service):
        """Should return 0 when tool is not permitted for role."""
        rbac_service.mysql.execute_query.return_value = []
        result = rbac_service.get_tool_rate_limit("CSR_READONLY", "update_claim_status")
        assert result == 0

    def test_returns_default_on_db_error(self, rbac_service):
        """Should return 30 (default) on database error."""
        rbac_service.mysql.execute_query.side_effect = Exception("DB error")
        result = rbac_service.get_tool_rate_limit("CSR_TIER1", "member_lookup")
        assert result == 30

    def test_caches_result(self, rbac_service):
        """Should cache rate limit after first lookup."""
        rbac_service.mysql.execute_query.return_value = [
            {"rate_limit_per_minute": 45}
        ]
        # First call
        result1 = rbac_service.get_tool_rate_limit("CSR_TIER1", "claim_lookup")
        # Second call (should use cache)
        result2 = rbac_service.get_tool_rate_limit("CSR_TIER1", "claim_lookup")

        assert result1 == 45
        assert result2 == 45
        # DB should only be called once
        assert rbac_service.mysql.execute_query.call_count == 1