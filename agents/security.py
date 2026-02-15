"""
Security controls: Authentication, RBAC, and access control
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import hashlib
import secrets
from jose import JWTError, jwt
from passlib.context import CryptContext

from databases.connections import get_mysql
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthenticationError(Exception):
    """Authentication failed"""
    pass


class AuthorizationError(Exception):
    """Authorization failed"""
    pass


class RateLimitError(Exception):
    """Rate limit exceeded"""
    pass


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
    
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token data
    
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.error(f"Token decode failed: {e}")
        raise AuthenticationError("Invalid token")


class AuthService:
    """Authentication service"""
    
    def __init__(self):
        self.mysql = get_mysql()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user
        
        Args:
            username: Username
            password: Plain text password
        
        Returns:
            User data if authenticated, None otherwise
        """
        try:
            # Get user from database
            query = """
            SELECT user_id, username, email, password_hash, role, first_name, last_name, is_active
            FROM users
            WHERE username = %s
            """
            users = self.mysql.execute_query(query, (username,))
            
            if not users:
                logger.warning(f"User not found: {username}")
                return None
            
            user = users[0]
            
            # Check if user is active
            if not user["is_active"]:
                logger.warning(f"User inactive: {username}")
                return None
            
            # Verify password
            if not verify_password(password, user["password_hash"]):
                logger.warning(f"Invalid password for user: {username}")
                return None
            
            # Update last login
            update_query = "UPDATE users SET last_login = NOW() WHERE user_id = %s"
            self.mysql.execute_update(update_query, (user["user_id"],))
            
            # Remove password hash from return
            user.pop("password_hash", None)
            
            logger.info(f"User authenticated: {username}")
            return user
        
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """
        Create a user session
        
        Args:
            user_id: User ID
            ip_address: Client IP address
            user_agent: Client user agent
        
        Returns:
            Session token
        """
        try:
            session_id = secrets.token_urlsafe(32)
            session_token = secrets.token_urlsafe(64)
            expires_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
            
            query = """
            INSERT INTO user_sessions (session_id, user_id, session_token, ip_address, user_agent, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.mysql.execute_update(query, (
                session_id, user_id, session_token, ip_address, user_agent, expires_at
            ))
            
            logger.info(f"Session created for user: {user_id}")
            return session_token
        
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session token
        
        Args:
            session_token: Session token
        
        Returns:
            Session data if valid, None otherwise
        """
        try:
            query = """
            SELECT s.session_id, s.user_id, s.expires_at, u.username, u.role
            FROM user_sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.session_token = %s AND s.is_active = TRUE AND s.expires_at > NOW()
            """
            sessions = self.mysql.execute_query(query, (session_token,))
            
            if not sessions:
                return None
            
            return sessions[0]
        
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None


class RBACService:
    """Role-Based Access Control service"""
    
    def __init__(self):
        self.mysql = get_mysql()
        self._permission_cache = {}
        self._tool_permission_cache = {}
    
    def check_permission(
        self,
        user_role: str,
        resource_type: str,
        action: str
    ) -> bool:
        """
        Check if a role has permission for an action on a resource
        
        Args:
            user_role: User's role
            resource_type: Type of resource (MEMBER, CLAIM, PA, etc.)
            action: Action to perform (READ, WRITE, UPDATE, DELETE)
        
        Returns:
            True if permitted, False otherwise
        """
        cache_key = f"{user_role}:{resource_type}:{action}"
        
        # Check cache
        if cache_key in self._permission_cache:
            return self._permission_cache[cache_key]
        
        try:
            query = """
            SELECT COUNT(*) as count
            FROM role_permissions rp
            JOIN permissions p ON rp.permission_id = p.permission_id
            WHERE rp.role = %s AND p.resource_type = %s AND p.action = %s
            """
            result = self.mysql.execute_query(query, (user_role, resource_type, action))
            
            has_permission = result[0]["count"] > 0
            
            # Cache result
            self._permission_cache[cache_key] = has_permission
            
            return has_permission
        
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    def check_tool_permission(
        self,
        user_role: str,
        tool_name: str
    ) -> bool:
        """
        Check if a role has permission to use a tool
        
        Args:
            user_role: User's role
            tool_name: Name of the tool
        
        Returns:
            True if permitted, False otherwise
        """
        cache_key = f"{user_role}:{tool_name}"
        
        # Check cache
        if cache_key in self._tool_permission_cache:
            return self._tool_permission_cache[cache_key]
        
        try:
            query = """
            SELECT is_allowed
            FROM tool_permissions
            WHERE role = %s AND tool_name = %s
            """
            result = self.mysql.execute_query(query, (user_role, tool_name))
            
            has_permission = result[0]["is_allowed"] if result else False
            
            # Cache result
            self._tool_permission_cache[cache_key] = has_permission
            
            return has_permission
        
        except Exception as e:
            logger.error(f"Tool permission check failed: {e}")
            return False
    
    def get_user_permissions(self, user_role: str) -> List[Dict[str, Any]]:
        """
        Get all permissions for a role
        
        Args:
            user_role: User's role
        
        Returns:
            List of permissions
        """
        try:
            query = """
            SELECT p.permission_name, p.resource_type, p.action, p.description
            FROM role_permissions rp
            JOIN permissions p ON rp.permission_id = p.permission_id
            WHERE rp.role = %s
            """
            return self.mysql.execute_query(query, (user_role,))
        
        except Exception as e:
            logger.error(f"Get permissions failed: {e}")
            return []
    
    def get_user_tool_permissions(self, user_role: str) -> List[Dict[str, Any]]:
        """
        Get all tool permissions for a role
        
        Args:
            user_role: User's role
        
        Returns:
            List of tool permissions
        """
        try:
            query = """
            SELECT tool_name, is_allowed, rate_limit_per_minute
            FROM tool_permissions
            WHERE role = %s AND is_allowed = TRUE
            """
            return self.mysql.execute_query(query, (user_role,))
        
        except Exception as e:
            logger.error(f"Get tool permissions failed: {e}")
            return []


class RateLimiter:
    """Rate limiting service"""
    
    def __init__(self):
        self.mysql = get_mysql()
    
    def check_rate_limit(
        self,
        user_id: str,
        resource_type: str,
        resource_name: str,
        limit_per_minute: int
    ) -> bool:
        """
        Check if user has exceeded rate limit
        
        Args:
            user_id: User ID
            resource_type: Type of resource (API, TOOL, QUERY)
            resource_name: Name of resource
            limit_per_minute: Rate limit per minute
        
        Returns:
            True if within limit, False if exceeded
        
        Raises:
            RateLimitError: If rate limit exceeded
        """
        try:
            window_start = datetime.utcnow() - timedelta(minutes=1)
            window_end = datetime.utcnow()
            
            # Get current count
            query = """
            SELECT COALESCE(SUM(request_count), 0) as total_count
            FROM rate_limits
            WHERE user_id = %s AND resource_type = %s AND resource_name = %s
            AND window_end > %s
            """
            result = self.mysql.execute_query(query, (
                user_id, resource_type, resource_name, window_start
            ))
            
            current_count = result[0]["total_count"]
            
            if current_count >= limit_per_minute:
                logger.warning(f"Rate limit exceeded for user {user_id}: {resource_name}")
                raise RateLimitError(f"Rate limit exceeded: {limit_per_minute}/minute")
            
            # Increment count
            rate_limit_id = secrets.token_urlsafe(16)
            insert_query = """
            INSERT INTO rate_limits (rate_limit_id, user_id, resource_type, resource_name,
                                     request_count, window_start, window_end, limit_per_window)
            VALUES (%s, %s, %s, %s, 1, %s, %s, %s)
            ON DUPLICATE KEY UPDATE request_count = request_count + 1
            """
            self.mysql.execute_update(insert_query, (
                rate_limit_id, user_id, resource_type, resource_name,
                window_start, window_end, limit_per_minute
            ))
            
            return True
        
        except RateLimitError:
            raise
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Fail open


class AuditLogger:
    """Audit logging service"""
    
    def __init__(self):
        self.mysql = get_mysql()
    
    def log_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        changes: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "SUCCESS",
        error_message: Optional[str] = None
    ):
        """
        Log an audit event
        
        Args:
            user_id: User ID
            action: Action performed
            resource_type: Type of resource
            resource_id: Resource ID
            changes: Changes made (before/after)
            ip_address: Client IP
            user_agent: Client user agent
            status: Status (SUCCESS, FAILED)
            error_message: Error message if failed
        """
        try:
            import json
            audit_id = secrets.token_urlsafe(16)
            changes_json = json.dumps(changes) if changes else None
            
            query = """
            INSERT INTO audit_logs (audit_id, user_id, action, resource_type, resource_id,
                                    changes, ip_address, user_agent, status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.mysql.execute_update(query, (
                audit_id, user_id, action, resource_type, resource_id,
                changes_json, ip_address, user_agent, status, error_message
            ))
            
            logger.debug(f"Audit log created: {action} on {resource_type}/{resource_id}")
        
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")


# Global service instances
auth_service = AuthService()
rbac_service = RBACService()
rate_limiter = RateLimiter()
audit_logger = AuditLogger()
