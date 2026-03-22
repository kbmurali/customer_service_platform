"""
Security controls: Authentication, RBAC, and access control
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import logging
import hashlib
import secrets
from jose import JWTError, jwt
import bcrypt

from databases.connections import get_mysql
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


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
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


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
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
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
        # Expired tokens are expected (webapp polls /api/system/status every 15s;
        # if the container restarts, old JWTs remain in the browser until re-login).
        # Log at DEBUG to avoid flooding the log with ERROR every 15 seconds.
        err_str = str(e).lower()
        if "expired" in err_str:
            logger.debug("Token expired: %s", e)
        else:
            logger.warning("Token decode failed: %s", e)
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
            expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
            
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
    
    _CACHE_VERSION_KEY = "rbac:cache_version"

    def __init__(self):
        self.mysql = get_mysql()
        self._permission_cache = {}
        self._tool_permission_cache = {}
        self._local_cache_version = 0
        # Lazy Redis connection for cross-process cache invalidation.
        # When any process calls clear_cache(), it increments a Redis key.
        # Other processes (A2A servers) detect the version change on their
        # next permission check and invalidate their local cache.
        self._redis = None

    def _get_redis(self):
        """Lazy Redis connection — only created on first use."""
        if self._redis is None:
            try:
                import redis as _redis
                self._redis = _redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                self._redis.ping()
            except Exception as e:
                logger.warning("RBACService: Redis unavailable for cache sync: %s", e)
                self._redis = None
        return self._redis

    def _check_cache_version(self):
        """
        Compare local cache version with Redis. If Redis version is newer,
        another process called clear_cache() — invalidate local caches.
        """
        r = self._get_redis()
        if r is None:
            return
        try:
            remote_version = int(r.get(self._CACHE_VERSION_KEY) or 0)
            if remote_version > self._local_cache_version:
                self._permission_cache.clear()
                self._tool_permission_cache.clear()
                self._local_cache_version = remote_version
                logger.info(
                    "RBACService: cache invalidated by remote version %d",
                    remote_version,
                )
        except Exception as e:
            logger.debug("RBACService: Redis cache version check failed: %s", e)
    
    def check_permission(
        self,
        user_role: str,
        resource_type: str,
        action: str
    ) -> bool:
        """
        Check if a role has permission for an action on a resource
        """
        # Cross-process cache invalidation check
        self._check_cache_version()

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
        """
        # Cross-process cache invalidation check
        self._check_cache_version()

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

    def get_tool_rate_limit(self, user_role: str, tool_name: str) -> int:
        """
        Get the per-minute rate limit for a specific tool and role.
        
        Args:
            user_role: User's role (CSR_READONLY, CSR_TIER1, CSR_TIER2, CSR_SUPERVISOR)
            tool_name: Name of the tool
        
        Returns:
            Rate limit per minute. Returns 0 if tool is not permitted for the role,
            or a default of 30 if the lookup fails.
        """
        cache_key = f"rate:{user_role}:{tool_name}"
        
        if cache_key in self._tool_permission_cache:
            return self._tool_permission_cache[cache_key]
        
        try:
            query = """
            SELECT rate_limit_per_minute
            FROM tool_permissions
            WHERE role = %s AND tool_name = %s AND is_allowed = TRUE
            """
            result = self.mysql.execute_query(query, (user_role, tool_name))
            
            if result:
                limit = result[0]["rate_limit_per_minute"]
                self._tool_permission_cache[cache_key] = limit
                return limit
            
            # Tool not permitted for this role
            return 0
        
        except Exception as e:
            logger.error(f"Get tool rate limit failed: {e}")
            return 30  # Default fallback

    def clear_cache(self) -> Dict[str, int]:
        """
        Clear all in-memory permission caches and notify other processes.

        After clearing the local caches, increments a Redis version key
        (``rbac:cache_version``).  Every ``check_permission`` and
        ``check_tool_permission`` call in *any* process compares its
        local version against this key and invalidates if stale.

        Call this after making direct changes to the ``tool_permissions``
        or ``role_permissions`` tables in MySQL.

        Returns:
            Dict with the number of entries cleared from each cache.
        """
        perm_count = len(self._permission_cache)
        tool_count = len(self._tool_permission_cache)
        self._permission_cache.clear()
        self._tool_permission_cache.clear()

        # Bump Redis version so A2A servers invalidate their caches
        r = self._get_redis()
        if r:
            try:
                new_version = r.incr(self._CACHE_VERSION_KEY)
                self._local_cache_version = new_version
                logger.info(
                    "RBACService: Redis cache version bumped to %d",
                    new_version,
                )
            except Exception as e:
                logger.warning("RBACService: Redis cache version bump failed: %s", e)

        logger.info(
            "RBACService cache cleared: %d permission entries, %d tool permission entries",
            perm_count,
            tool_count,
        )
        return {
            "permission_cache_cleared":      perm_count,
            "tool_permission_cache_cleared": tool_count,
        }

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch role and active status for a user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            Dict with keys 'role' and 'is_active', or None if not found.
        """
        try:
            result = self.mysql.execute_query(
                "SELECT `role`, is_active FROM users WHERE user_id = %s",
                (user_id,),
            )
            return result[0] if result else None
        except Exception as e:
            logger.error(f"get_user failed for user_id={user_id}: {e}")
            return None

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
            window_start = datetime.now(timezone.utc) - timedelta(minutes=1)
            window_end = datetime.now(timezone.utc)
            
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


def get_rate_limiter() -> RateLimiter:
    """Get the global RateLimiter instance."""
    return rate_limiter
