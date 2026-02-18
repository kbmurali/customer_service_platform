"""
Unit tests for auding logging.

Tests cover:
- Audit logging (agents.security.AuditLogger - MySQL-based)
"""

import pytest
from unittest.mock import Mock, patch
from agents.security import AuditLogger


# ============================================
# Audit Logger Tests (MySQL-based)
# ============================================

class TestAuditLogger:
    """Test audit logging functionality"""

    @pytest.fixture
    def audit_logger(self):
        """Create audit logger with mocked database"""
        with patch('agents.security.get_mysql') as mock_mysql:
            logger = AuditLogger()
            logger.mysql = Mock()
            return logger

    def test_log_action_success(self, audit_logger):
        """Test successful audit log entry"""
        audit_logger.mysql.execute_update.return_value = None

        audit_logger.log_action(
            user_id="user123",
            action="MEMBER_LOOKUP",
            resource_type="MEMBER",
            resource_id="M123456",
            changes={"field": "status", "old": "active", "new": "inactive"},
            status="SUCCESS"
        )

        audit_logger.mysql.execute_update.assert_called_once()

        call_args = audit_logger.mysql.execute_update.call_args[0]
        assert "user123" in call_args[1]
        assert "MEMBER_LOOKUP" in call_args[1]

    def test_log_action_with_ip_and_user_agent(self, audit_logger):
        """Test audit log with IP address and user agent"""
        audit_logger.mysql.execute_update.return_value = None

        audit_logger.log_action(
            user_id="user123",
            action="LOGIN",
            resource_type="AUTH",
            resource_id="session123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            status="SUCCESS"
        )

        call_args = audit_logger.mysql.execute_update.call_args[0]
        assert "192.168.1.1" in call_args[1]

    def test_log_action_database_error(self, audit_logger):
        """Test audit log when database error occurs"""
        audit_logger.mysql.execute_update.side_effect = Exception("Insert failed")

        # Should not raise exception (logging failure shouldn't break app)
        try:
            audit_logger.log_action(
                user_id="user123",
                action="TEST",
                resource_type="TEST",
                resource_id="test123",
                status="SUCCESS"
            )
        except Exception:
            pytest.fail("Audit logging should not raise exceptions")
