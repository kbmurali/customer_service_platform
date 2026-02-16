"""
Unit tests for RBAC (Role-Based Access Control) service.

Tests cover:
- Permission checking for different roles
- Tool permission validation
- Permission caching
- Role hierarchy
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.security import RBACService, AuthorizationError


class TestRBACService:
    """Test suite for RBACService"""
    
    @pytest.fixture
    def rbac_service(self):
        """Create RBAC service instance with mocked database"""
        with patch('agents.security.get_mysql') as mock_mysql:
            service = RBACService()
            service.mysql = Mock()
            # Clear caches
            service._permission_cache = {}
            service._tool_permission_cache = {}
            return service
    
    # ============================================
    # Test: Permission Checking
    # ============================================
    
    def test_check_permission_allowed(self, rbac_service):
        """Test permission check when permission is granted"""
        # Mock database response - permission exists
        rbac_service.mysql.execute_query.return_value = [{"count": 1}]
        
        result = rbac_service.check_permission(
            user_role="CSR_TIER1",
            resource_type="MEMBER",
            action="READ"
        )
        
        assert result is True
        rbac_service.mysql.execute_query.assert_called_once()
        
        # Verify query parameters
        call_args = rbac_service.mysql.execute_query.call_args
        assert call_args[0][1] == ("CSR_TIER1", "MEMBER", "READ")
    
    def test_check_permission_denied(self, rbac_service):
        """Test permission check when permission is denied"""
        # Mock database response - no permission
        rbac_service.mysql.execute_query.return_value = [{"count": 0}]
        
        result = rbac_service.check_permission(
            user_role="CSR_READONLY",
            resource_type="CLAIM",
            action="UPDATE"
        )
        
        assert result is False
    
    def test_check_permission_caching(self, rbac_service):
        """Test that permission checks are cached"""
        # Mock database response
        rbac_service.mysql.execute_query.return_value = [{"count": 1}]
        
        # First call - should query database
        result1 = rbac_service.check_permission("CSR_TIER1", "MEMBER", "READ")
        assert result1 is True
        assert rbac_service.mysql.execute_query.call_count == 1
        
        # Second call - should use cache
        result2 = rbac_service.check_permission("CSR_TIER1", "MEMBER", "READ")
        assert result2 is True
        assert rbac_service.mysql.execute_query.call_count == 1  # No additional call
        
        # Verify cache key
        cache_key = "CSR_TIER1:MEMBER:READ"
        assert cache_key in rbac_service._permission_cache
        assert rbac_service._permission_cache[cache_key] is True
    
    def test_check_permission_different_roles(self, rbac_service):
        """Test permission checks for different role levels"""
        test_cases = [
            # (role, resource, action, expected_count, expected_result)
            ("CSR_TIER1", "MEMBER", "READ", 1, True),
            ("CSR_TIER2", "CLAIM", "UPDATE", 1, True),
            ("CSR_SUPERVISOR", "PA", "APPROVE", 1, True),
            ("CSR_READONLY", "CLAIM", "UPDATE", 0, False),
        ]
        
        for role, resource, action, count, expected in test_cases:
            # Clear cache
            rbac_service._permission_cache = {}
            
            # Mock response
            rbac_service.mysql.execute_query.return_value = [{"count": count}]
            
            result = rbac_service.check_permission(role, resource, action)
            assert result == expected, f"Failed for {role}:{resource}:{action}"
    
    def test_check_permission_database_error(self, rbac_service):
        """Test permission check when database error occurs"""
        # Mock database error
        rbac_service.mysql.execute_query.side_effect = Exception("Database connection failed")
        
        result = rbac_service.check_permission("CSR_TIER1", "MEMBER", "READ")
        
        # Should return False on error (fail-safe)
        assert result is False
    
    # ============================================
    # Test: Tool Permission Checking
    # ============================================
    
    def test_check_tool_permission_allowed(self, rbac_service):
        """Test tool permission when access is granted"""
        # Mock database response
        rbac_service.mysql.execute_query.return_value = [{"is_allowed": True}]
        
        result = rbac_service.check_tool_permission(
            user_role="CSR_TIER2",
            tool_name="lookup_member"
        )
        
        assert result is True
    
    def test_check_tool_permission_denied(self, rbac_service):
        """Test tool permission when access is denied"""
        # Mock database response
        rbac_service.mysql.execute_query.return_value = [{"is_allowed": False}]
        
        result = rbac_service.check_tool_permission(
            user_role="CSR_READONLY",
            tool_name="update_claim"
        )
        
        assert result is False
    
    def test_check_tool_permission_not_found(self, rbac_service):
        """Test tool permission when tool not found in database"""
        # Mock empty response
        rbac_service.mysql.execute_query.return_value = []
        
        result = rbac_service.check_tool_permission(
            user_role="CSR_TIER1",
            tool_name="nonexistent_tool"
        )
        
        assert result is False
    
    def test_check_tool_permission_caching(self, rbac_service):
        """Test that tool permission checks are cached"""
        # Mock database response
        rbac_service.mysql.execute_query.return_value = [{"is_allowed": True}]
        
        # First call
        result1 = rbac_service.check_tool_permission("CSR_TIER1", "lookup_member")
        assert result1 is True
        assert rbac_service.mysql.execute_query.call_count == 1
        
        # Second call - should use cache
        result2 = rbac_service.check_tool_permission("CSR_TIER1", "lookup_member")
        assert result2 is True
        assert rbac_service.mysql.execute_query.call_count == 1
        
        # Verify cache
        cache_key = "CSR_TIER1:lookup_member"
        assert cache_key in rbac_service._tool_permission_cache
    
    def test_check_tool_permission_database_error(self, rbac_service):
        """Test tool permission check when database error occurs"""
        rbac_service.mysql.execute_query.side_effect = Exception("Connection timeout")
        
        result = rbac_service.check_tool_permission("CSR_TIER1", "lookup_member")
        
        # Should return False on error
        assert result is False
    
    # ============================================
    # Test: Get User Permissions
    # ============================================
    
    def test_get_user_permissions_success(self, rbac_service):
        """Test retrieving all permissions for a role"""
        # Mock database response
        mock_permissions = [
            {
                "permission_name": "member_read",
                "resource_type": "MEMBER",
                "action": "READ",
                "description": "Read member information"
            },
            {
                "permission_name": "claim_read",
                "resource_type": "CLAIM",
                "action": "READ",
                "description": "Read claim information"
            }
        ]
        rbac_service.mysql.execute_query.return_value = mock_permissions
        
        result = rbac_service.get_user_permissions("CSR_TIER1")
        
        assert len(result) == 2
        assert result[0]["permission_name"] == "member_read"
        assert result[1]["resource_type"] == "CLAIM"
    
    def test_get_user_permissions_empty(self, rbac_service):
        """Test retrieving permissions for role with no permissions"""
        rbac_service.mysql.execute_query.return_value = []
        
        result = rbac_service.get_user_permissions("unknown_role")
        
        assert result == []
    
    def test_get_user_permissions_database_error(self, rbac_service):
        """Test get permissions when database error occurs"""
        rbac_service.mysql.execute_query.side_effect = Exception("Query failed")
        
        result = rbac_service.get_user_permissions("CSR_TIER1")
        
        # Should return empty list on error
        assert result == []
    
    # ============================================
    # Test: Get Tool Permissions
    # ============================================
    
    def test_get_user_tool_permissions_success(self, rbac_service):
        """Test retrieving all tool permissions for a role"""
        mock_tools = [
            {
                "tool_name": "lookup_member",
                "is_allowed": True,
                "rate_limit_per_minute": 60
            },
            {
                "tool_name": "lookup_claim",
                "is_allowed": True,
                "rate_limit_per_minute": 30
            }
        ]
        rbac_service.mysql.execute_query.return_value = mock_tools
        
        result = rbac_service.get_user_tool_permissions("CSR_TIER2")
        
        assert len(result) == 2
        assert result[0]["tool_name"] == "lookup_member"
        assert result[1]["rate_limit_per_minute"] == 30
    
    def test_get_user_tool_permissions_empty(self, rbac_service):
        """Test retrieving tool permissions for role with no tools"""
        rbac_service.mysql.execute_query.return_value = []
        
        result = rbac_service.get_user_tool_permissions("CSR_READONLY")
        
        assert result == []
    
    def test_get_user_tool_permissions_database_error(self, rbac_service):
        """Test get tool permissions when database error occurs"""
        rbac_service.mysql.execute_query.side_effect = Exception("Database error")
        
        result = rbac_service.get_user_tool_permissions("CSR_TIER1")
        
        assert result == []
    
    # ============================================
    # Test: Role Case Normalization
    # ============================================
    
    def test_role_case_sensitivity(self, rbac_service):
        """Test that role names are case-sensitive (documents the bug)"""
        # This test documents the known case mismatch bug
        # Database has uppercase roles (CSR_TIER1)
        # Code may pass lowercase (csr_tier1)
        
        rbac_service.mysql.execute_query.return_value = [{"count": 1}]
        
        # Uppercase should work
        result_lower = rbac_service.check_permission("CSR_TIER1", "MEMBER", "READ")
        assert result_lower is True
        
        # Clear cache
        rbac_service._permission_cache = {}
        
        # Lowercase might not work (depends on database collation)
        rbac_service.mysql.execute_query.return_value = [{"count": 0}]
        result_upper = rbac_service.check_permission("csr_tier1", "MEMBER", "READ")
        assert result_upper is False
    
    # ============================================
    # Test: Edge Cases
    # ============================================
    
    def test_check_permission_with_none_values(self, rbac_service):
        """Test permission check with None values"""
        rbac_service.mysql.execute_query.return_value = [{"count": 0}]
        
        # Should handle None gracefully
        result = rbac_service.check_permission(None, "MEMBER", "READ")
        assert result is False
    
    def test_check_permission_with_empty_strings(self, rbac_service):
        """Test permission check with empty strings"""
        rbac_service.mysql.execute_query.return_value = [{"count": 0}]
        
        result = rbac_service.check_permission("", "", "")
        assert result is False
    
    def test_cache_isolation(self, rbac_service):
        """Test that permission cache and tool cache are isolated"""
        # Add to permission cache
        rbac_service._permission_cache["test_key"] = True
        
        # Add to tool cache
        rbac_service._tool_permission_cache["test_key"] = False
        
        # Should be independent
        assert rbac_service._permission_cache["test_key"] is True
        assert rbac_service._tool_permission_cache["test_key"] is False


# ============================================
# Integration-style tests with realistic scenarios
# ============================================

class TestRBACScenarios:
    """Test realistic RBAC scenarios"""
    
    @pytest.fixture
    def rbac_service(self):
        """Create RBAC service with mock database"""
        with patch('agents.security.get_mysql') as mock_mysql:
            service = RBACService()
            service.mysql = Mock()
            service._permission_cache = {}
            service._tool_permission_cache = {}
            return service
    
    def test_tier1_csr_permissions(self, rbac_service):
        """Test typical Tier 1 CSR permissions"""
        # Tier 1 can read but not update
        permissions = {
            ("CSR_TIER1", "MEMBER", "READ"): True,
            ("CSR_TIER1", "MEMBER", "UPDATE"): False,
            ("CSR_TIER1", "CLAIM", "READ"): True,
            ("CSR_TIER1", "CLAIM", "UPDATE"): False,
            ("CSR_TIER1", "PA", "APPROVE"): False,
        }
        
        for (role, resource, action), expected in permissions.items():
            rbac_service._permission_cache = {}
            rbac_service.mysql.execute_query.return_value = [{"count": 1 if expected else 0}]
            
            result = rbac_service.check_permission(role, resource, action)
            assert result == expected, f"Failed: {role} {resource} {action}"
    
    def test_tier2_csr_permissions(self, rbac_service):
        """Test typical Tier 2 CSR permissions"""
        # Tier 2 can read and update
        permissions = {
            ("CSR_TIER2", "MEMBER", "READ"): True,
            ("CSR_TIER2", "MEMBER", "UPDATE"): True,
            ("CSR_TIER2", "CLAIM", "READ"): True,
            ("CSR_TIER2", "CLAIM", "UPDATE"): True,
            ("CSR_TIER2", "PA", "APPROVE"): False,  # Still can't approve
        }
        
        for (role, resource, action), expected in permissions.items():
            rbac_service._permission_cache = {}
            rbac_service.mysql.execute_query.return_value = [{"count": 1 if expected else 0}]
            
            result = rbac_service.check_permission(role, resource, action)
            assert result == expected
    
    def test_supervisor_permissions(self, rbac_service):
        """Test supervisor permissions (full access)"""
        # Supervisor can do everything
        permissions = {
            ("CSR_SUPERVISOR", "MEMBER", "READ"): True,
            ("CSR_SUPERVISOR", "MEMBER", "UPDATE"): True,
            ("CSR_SUPERVISOR", "CLAIM", "UPDATE"): True,
            ("CSR_SUPERVISOR", "PA", "APPROVE"): True,
            ("CSR_SUPERVISOR", "PA", "REJECT"): True,
        }
        
        for (role, resource, action), expected in permissions.items():
            rbac_service._permission_cache = {}
            rbac_service.mysql.execute_query.return_value = [{"count": 1 if expected else 0}]
            
            result = rbac_service.check_permission(role, resource, action)
            assert result == expected
    
    def test_readonly_permissions(self, rbac_service):
        """Test read-only role permissions"""
        # Read-only can only read
        permissions = {
            ("CSR_READONLY", "MEMBER", "READ"): True,
            ("CSR_READONLY", "MEMBER", "UPDATE"): False,
            ("CSR_READONLY", "CLAIM", "READ"): True,
            ("CSR_READONLY", "CLAIM", "UPDATE"): False,
            ("CSR_READONLY", "PA", "READ"): True,
            ("CSR_READONLY", "PA", "APPROVE"): False,
        }
        
        for (role, resource, action), expected in permissions.items():
            rbac_service._permission_cache = {}
            rbac_service.mysql.execute_query.return_value = [{"count": 1 if expected else 0}]
            
            result = rbac_service.check_permission(role, resource, action)
            assert result == expected
    
    def test_tool_access_by_role(self, rbac_service):
        """Test tool access patterns by role"""
        tool_access = {
            ("CSR_TIER1", "lookup_member"): True,
            ("CSR_TIER1", "update_member"): False,
            ("CSR_TIER2", "lookup_member"): True,
            ("CSR_TIER2", "update_member"): True,
            ("CSR_SUPERVISOR", "approve_pa"): True,
            ("CSR_READONLY", "update_member"): False,
        }
        
        for (role, tool), expected in tool_access.items():
            rbac_service._tool_permission_cache = {}
            rbac_service.mysql.execute_query.return_value = [{"is_allowed": expected}] if expected else []
            
            result = rbac_service.check_tool_permission(role, tool)
            assert result == expected, f"Failed: {role} {tool}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
