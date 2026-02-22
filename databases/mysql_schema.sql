-- ============================================================
-- MySQL Relational Database Schema
-- Health Insurance AI Platform
-- Transactional data, authentication, RBAC, metrics,
-- and Control 5 (Human-in-the-Loop / Circuit Breaker)
-- ============================================================

-- ============================================
-- User Authentication and Authorization
-- ============================================

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,  -- CSR_TIER1, CSR_TIER2, CSR_SUPERVISOR, CSR_READONLY
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS user_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_session_token (session_token),
    INDEX idx_expires_at (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Role-Based Access Control (RBAC)
-- ============================================

CREATE TABLE IF NOT EXISTS permissions (
    permission_id VARCHAR(36) PRIMARY KEY,
    permission_name VARCHAR(100) UNIQUE NOT NULL,
    resource_type VARCHAR(50) NOT NULL,  -- MEMBER, CLAIM, PA, POLICY, PROVIDER, ANALYTICS, AGENT
    action VARCHAR(50) NOT NULL,         -- READ, WRITE, UPDATE, DELETE, QUERY, APPROVE, REJECT
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_resource_type (resource_type),
    INDEX idx_action (action)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS role_permissions (
    role VARCHAR(50) NOT NULL,
    permission_id VARCHAR(36) NOT NULL,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (role, permission_id),
    FOREIGN KEY (permission_id) REFERENCES permissions(permission_id) ON DELETE CASCADE,
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS tool_permissions (
    tool_permission_id VARCHAR(36) PRIMARY KEY,
    role VARCHAR(50) NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    is_allowed BOOLEAN DEFAULT FALSE,
    rate_limit_per_minute INT DEFAULT 60,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_role_tool (role, tool_name),
    INDEX idx_role (role),
    INDEX idx_tool_name (tool_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Audit Logs
-- ============================================

CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(36) NOT NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    changes JSON,
    ip_address VARCHAR(45),
    user_agent TEXT,
    status VARCHAR(20) DEFAULT 'SUCCESS',  -- SUCCESS, FAILED, BLOCKED, DENIED
    error_message TEXT,
    INDEX idx_user_id (user_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_resource_type (resource_type),
    INDEX idx_action (action),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Security Events
-- ============================================

CREATE TABLE IF NOT EXISTS security_events (
    event_id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(100) NOT NULL,  -- ACCESS_DENIED, PERMISSION_VIOLATION, CIRCUIT_BREAKER, etc.
    severity VARCHAR(20) NOT NULL,     -- LOW, MEDIUM, HIGH, CRITICAL
    user_id VARCHAR(36),
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details TEXT,
    ip_address VARCHAR(45),
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP NULL,
    resolved_by VARCHAR(36),
    INDEX idx_timestamp (timestamp),
    INDEX idx_event_type (event_type),
    INDEX idx_severity (severity),
    INDEX idx_user_id (user_id),
    INDEX idx_resolved (resolved),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (resolved_by) REFERENCES users(user_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Control 5: Human-in-the-Loop Approval Requests
-- ============================================

CREATE TABLE IF NOT EXISTS approval_requests (
    request_id VARCHAR(255) PRIMARY KEY,
    action_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    tool_name VARCHAR(255) NOT NULL,
    parameters JSON NOT NULL,
    context JSON,
    impact_level VARCHAR(50) NOT NULL,   -- LOW, MEDIUM, HIGH, CRITICAL
    requested_by VARCHAR(255) NOT NULL,
    requested_at DATETIME NOT NULL,
    expires_at DATETIME NOT NULL,
    status VARCHAR(50) NOT NULL,         -- PENDING, APPROVED, DENIED, EXPIRED
    reviewed_by VARCHAR(255),
    reviewed_at DATETIME,
    review_rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_approval_status (status),
    INDEX idx_approval_agent (agent_id),
    INDEX idx_approval_requested_at (requested_at),
    INDEX idx_approval_tool (tool_name),
    INDEX idx_approval_impact (impact_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Control 5: Circuit Breaker Events
-- ============================================

CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    event_id VARCHAR(36) PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,     -- ACTIVATED, DEACTIVATED
    reason TEXT NOT NULL,
    triggered_by VARCHAR(255) NOT NULL,
    triggered_at DATETIME NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_cb_event_type (event_type),
    INDEX idx_cb_triggered_at (triggered_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Agent Metrics and Analytics
-- ============================================

CREATE TABLE IF NOT EXISTS agent_metrics (
    metric_id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,  -- SUPERVISOR, WORKER, TOOL
    execution_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    avg_execution_time_ms DECIMAL(10, 2),
    total_execution_time_ms BIGINT DEFAULT 0,
    tool_call_count INT DEFAULT 0,
    INDEX idx_timestamp (timestamp),
    INDEX idx_agent_name (agent_name),
    INDEX idx_agent_type (agent_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS tool_metrics (
    metric_id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tool_name VARCHAR(100) NOT NULL,
    tool_category VARCHAR(50) NOT NULL,  -- DATABASE, API, COMPUTATION, SEARCH
    call_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    avg_execution_time_ms DECIMAL(10, 2),
    total_execution_time_ms BIGINT DEFAULT 0,
    INDEX idx_timestamp (timestamp),
    INDEX idx_tool_name (tool_name),
    INDEX idx_tool_category (tool_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS session_metrics (
    metric_id VARCHAR(36) PRIMARY KEY,
    date DATE NOT NULL,
    hour INT NOT NULL,  -- 0-23
    total_sessions INT DEFAULT 0,
    active_sessions INT DEFAULT 0,
    completed_sessions INT DEFAULT 0,
    abandoned_sessions INT DEFAULT 0,
    error_sessions INT DEFAULT 0,
    avg_duration_seconds DECIMAL(10, 2),
    avg_interactions_per_session DECIMAL(10, 2),
    UNIQUE KEY unique_date_hour (date, hour),
    INDEX idx_date (date),
    INDEX idx_hour (hour)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Rate Limiting
-- ============================================

CREATE TABLE IF NOT EXISTS rate_limits (
    rate_limit_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,  -- REQUEST, TOOL, API, QUERY
    -- REQUEST = global per-user request gate (request_processor.py)
    -- TOOL    = per-tool rate limit (agents/tools.py via check_rate_limit_for_tool)
    resource_name VARCHAR(100) NOT NULL,
    request_count INT DEFAULT 0,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    limit_per_window INT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT chk_resource_type CHECK (resource_type IN ('REQUEST', 'TOOL', 'API', 'QUERY')),
    INDEX idx_user_id (user_id),
    INDEX idx_window_end (window_end),
    INDEX idx_resource (resource_type, resource_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Configuration and Settings
-- ============================================

CREATE TABLE IF NOT EXISTS system_config (
    config_id VARCHAR(36) PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT,
    config_type VARCHAR(50) NOT NULL,  -- STRING, INT, FLOAT, BOOLEAN, JSON
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_config_key (config_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- LLM and Model Configuration
-- ============================================

CREATE TABLE IF NOT EXISTS llm_configs (
    config_id VARCHAR(36) PRIMARY KEY,
    config_name VARCHAR(100) UNIQUE NOT NULL,
    provider VARCHAR(50) NOT NULL,  -- OPENAI, ANTHROPIC, AWS_BEDROCK, GOOGLE
    model_name VARCHAR(100) NOT NULL,
    api_endpoint VARCHAR(255),
    temperature DECIMAL(3, 2) DEFAULT 0.70,
    max_tokens INT DEFAULT 4096,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_provider (provider),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Observability Integration
-- ============================================

CREATE TABLE IF NOT EXISTS langfuse_traces (
    trace_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36),
    user_id VARCHAR(36),
    trace_name VARCHAR(255),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(50),
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_user_id (user_id),
    INDEX idx_start_time (start_time),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- SEED DATA
-- ============================================================

-- ============================================
-- Permissions (resource_type + action)
-- ============================================

INSERT INTO permissions (permission_id, permission_name, resource_type, action, description) VALUES
    -- MEMBER
    (UUID(), 'member_read',     'MEMBER',    'READ',    'Read member information'),
    (UUID(), 'member_write',    'MEMBER',    'WRITE',   'Create new member records'),
    (UUID(), 'member_update',   'MEMBER',    'UPDATE',  'Update member information'),
    (UUID(), 'member_delete',   'MEMBER',    'DELETE',  'Delete member records'),
    -- CLAIM
    (UUID(), 'claim_read',      'CLAIM',     'READ',    'Read claim information'),
    (UUID(), 'claim_write',     'CLAIM',     'WRITE',   'Create new claims'),
    (UUID(), 'claim_update',    'CLAIM',     'UPDATE',  'Update claim status'),
    (UUID(), 'claim_delete',    'CLAIM',     'DELETE',  'Delete claim records'),
    -- PA (Prior Authorization)
    (UUID(), 'pa_read',         'PA',        'READ',    'Read prior authorization information'),
    (UUID(), 'pa_write',        'PA',        'WRITE',   'Create new prior authorizations'),
    (UUID(), 'pa_update',       'PA',        'UPDATE',  'Update prior authorization status'),
    (UUID(), 'pa_approve',      'PA',        'APPROVE', 'Approve prior authorizations'),
    (UUID(), 'pa_reject',       'PA',        'REJECT',  'Reject prior authorizations'),
    -- POLICY
    (UUID(), 'policy_read',     'POLICY',    'READ',    'Read policy information'),
    -- PROVIDER
    (UUID(), 'provider_read',   'PROVIDER',  'READ',    'Read provider information'),
    -- ANALYTICS
    (UUID(), 'analytics_read',  'ANALYTICS', 'READ',    'View analytics and reports'),
    -- AGENT (used in request_processor.py for query gating)
    (UUID(), 'agent_query',     'AGENT',     'QUERY',   'Query the agent system');

-- ============================================
-- Role Permissions
-- ============================================

-- CSR_READONLY: Read-only access
INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_READONLY', permission_id FROM permissions WHERE action = 'READ';

INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_READONLY', permission_id FROM permissions WHERE permission_name = 'agent_query';

-- CSR_TIER1: Read + Agent query
INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_TIER1', permission_id FROM permissions WHERE permission_name IN (
    'member_read', 'claim_read', 'pa_read', 'policy_read', 'provider_read',
    'analytics_read', 'agent_query'
);

-- CSR_TIER2: Read + Write + Update + Agent query
INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_TIER2', permission_id FROM permissions WHERE permission_name IN (
    'member_read', 'member_update',
    'claim_read', 'claim_write', 'claim_update',
    'pa_read', 'pa_write', 'pa_update',
    'policy_read', 'provider_read',
    'analytics_read', 'agent_query'
);

-- CSR_SUPERVISOR: All permissions
INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_SUPERVISOR', permission_id FROM permissions;

-- ============================================
-- Tool Permissions
-- Synchronized with the 19 tools in agents/tools.py
-- 15 read tools + 4 write tools (Control 5: Human-in-the-Loop gated)
-- ============================================

INSERT INTO tool_permissions (tool_permission_id, role, tool_name, is_allowed, rate_limit_per_minute) VALUES
    -- -------------------------------------------------------
    -- CSR_READONLY: Read-only tools, conservative rate limits
    -- -------------------------------------------------------
    (UUID(), 'CSR_READONLY', 'member_lookup',          TRUE,  20),
    (UUID(), 'CSR_READONLY', 'check_eligibility',      TRUE,  20),
    (UUID(), 'CSR_READONLY', 'coverage_lookup',         TRUE,  20),
    (UUID(), 'CSR_READONLY', 'claim_lookup',            TRUE,  20),
    (UUID(), 'CSR_READONLY', 'claim_status',            TRUE,  20),
    (UUID(), 'CSR_READONLY', 'claim_payment_info',      TRUE,  20),
    (UUID(), 'CSR_READONLY', 'pa_lookup',               TRUE,  20),
    (UUID(), 'CSR_READONLY', 'pa_status',               TRUE,  20),
    (UUID(), 'CSR_READONLY', 'pa_requirements',         TRUE,  20),
    (UUID(), 'CSR_READONLY', 'provider_search',         TRUE,  20),
    (UUID(), 'CSR_READONLY', 'provider_lookup',         TRUE,  20),
    (UUID(), 'CSR_READONLY', 'network_check',           TRUE,  20),
    (UUID(), 'CSR_READONLY', 'search_policy_info',      TRUE,  20),
    (UUID(), 'CSR_READONLY', 'search_medical_codes',    TRUE,  20),
    (UUID(), 'CSR_READONLY', 'search_knowledge_base',   TRUE,  20),
    -- CSR_READONLY: No write tools

    -- -------------------------------------------------------
    -- CSR_TIER1: Standard read tools, moderate rate limits
    -- -------------------------------------------------------
    (UUID(), 'CSR_TIER1', 'member_lookup',          TRUE,  30),
    (UUID(), 'CSR_TIER1', 'check_eligibility',      TRUE,  30),
    (UUID(), 'CSR_TIER1', 'coverage_lookup',         TRUE,  30),
    (UUID(), 'CSR_TIER1', 'claim_lookup',            TRUE,  30),
    (UUID(), 'CSR_TIER1', 'claim_status',            TRUE,  30),
    (UUID(), 'CSR_TIER1', 'claim_payment_info',      TRUE,  30),
    (UUID(), 'CSR_TIER1', 'pa_lookup',               TRUE,  30),
    (UUID(), 'CSR_TIER1', 'pa_status',               TRUE,  30),
    (UUID(), 'CSR_TIER1', 'pa_requirements',         TRUE,  30),
    (UUID(), 'CSR_TIER1', 'provider_search',         TRUE,  20),
    (UUID(), 'CSR_TIER1', 'provider_lookup',         TRUE,  20),
    (UUID(), 'CSR_TIER1', 'network_check',           TRUE,  20),
    (UUID(), 'CSR_TIER1', 'search_policy_info',      TRUE,  30),
    (UUID(), 'CSR_TIER1', 'search_medical_codes',    TRUE,  30),
    (UUID(), 'CSR_TIER1', 'search_knowledge_base',   TRUE,  30),
    -- CSR_TIER1: No write tools

    -- -------------------------------------------------------
    -- CSR_TIER2: All read tools + higher rate limits
    -- -------------------------------------------------------
    (UUID(), 'CSR_TIER2', 'member_lookup',          TRUE,  60),
    (UUID(), 'CSR_TIER2', 'check_eligibility',      TRUE,  60),
    (UUID(), 'CSR_TIER2', 'coverage_lookup',         TRUE,  60),
    (UUID(), 'CSR_TIER2', 'claim_lookup',            TRUE,  60),
    (UUID(), 'CSR_TIER2', 'claim_status',            TRUE,  60),
    (UUID(), 'CSR_TIER2', 'claim_payment_info',      TRUE,  60),
    (UUID(), 'CSR_TIER2', 'pa_lookup',               TRUE,  60),
    (UUID(), 'CSR_TIER2', 'pa_status',               TRUE,  60),
    (UUID(), 'CSR_TIER2', 'pa_requirements',         TRUE,  60),
    (UUID(), 'CSR_TIER2', 'provider_search',         TRUE,  40),
    (UUID(), 'CSR_TIER2', 'provider_lookup',         TRUE,  40),
    (UUID(), 'CSR_TIER2', 'network_check',           TRUE,  40),
    (UUID(), 'CSR_TIER2', 'search_policy_info',      TRUE,  60),
    (UUID(), 'CSR_TIER2', 'search_medical_codes',    TRUE,  60),
    (UUID(), 'CSR_TIER2', 'search_knowledge_base',   TRUE,  60),
    -- CSR_TIER2: Write tools (Human-in-the-Loop gated)
    (UUID(), 'CSR_TIER2', 'update_claim_status',      TRUE,  10),
    (UUID(), 'CSR_TIER2', 'approve_prior_auth',       TRUE,  10),
    (UUID(), 'CSR_TIER2', 'deny_prior_auth',          TRUE,  10),
    (UUID(), 'CSR_TIER2', 'update_member_info',       TRUE,  10),

    -- -------------------------------------------------------
    -- CSR_SUPERVISOR: All tools + highest rate limits
    -- -------------------------------------------------------
    (UUID(), 'CSR_SUPERVISOR', 'member_lookup',          TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'check_eligibility',      TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'coverage_lookup',         TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'claim_lookup',            TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'claim_status',            TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'claim_payment_info',      TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'pa_lookup',               TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'pa_status',               TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'pa_requirements',         TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'provider_search',         TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'provider_lookup',         TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'network_check',           TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'search_policy_info',      TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'search_medical_codes',    TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'search_knowledge_base',   TRUE, 120),
    -- CSR_SUPERVISOR: Write tools (Human-in-the-Loop gated)
    (UUID(), 'CSR_SUPERVISOR', 'update_claim_status',      TRUE,  30),
    (UUID(), 'CSR_SUPERVISOR', 'approve_prior_auth',       TRUE,  30),
    (UUID(), 'CSR_SUPERVISOR', 'deny_prior_auth',          TRUE,  30),
    (UUID(), 'CSR_SUPERVISOR', 'update_member_info',       TRUE,  30);

-- ============================================
-- LLM Configurations
-- ============================================

INSERT INTO llm_configs (config_id, config_name, provider, model_name, api_endpoint, temperature, max_tokens, is_active) VALUES
    (UUID(), 'default_openai', 'OPENAI', 'gpt-4.1-mini',    NULL, 0.70, 4096, TRUE),
    (UUID(), 'fast_model',     'OPENAI', 'gpt-4.1-nano',    NULL, 0.50, 2048, TRUE),
    (UUID(), 'google_model',   'GOOGLE', 'gemini-2.5-flash', NULL, 0.70, 4096, FALSE);

-- ============================================
-- Sample Users
-- ============================================
-- Password: 'testuser' hashed with bcrypt
-- bcrypt hash: $2b$12$LJ3m4ys3Lk0TSwMCPNEJluQMNGxFhGGArSWKhBKFy/B3CehEMxzFi

INSERT INTO users (user_id, username, email, password_hash, role, first_name, last_name, is_active) VALUES
    ('usr-readonly-001', 'jthompson',  'jane.thompson@healthins.example.com',
     '$2b$12$LJ3m4ys3Lk0TSwMCPNEJluQMNGxFhGGArSWKhBKFy/B3CehEMxzFi',
     'CSR_READONLY',   'Jane',    'Thompson',  TRUE),

    ('usr-tier1-001',    'mgarcia',    'maria.garcia@healthins.example.com',
     '$2b$12$LJ3m4ys3Lk0TSwMCPNEJluQMNGxFhGGArSWKhBKFy/B3CehEMxzFi',
     'CSR_TIER1',      'Maria',   'Garcia',    TRUE),

    ('usr-tier1-002',    'dkim',       'david.kim@healthins.example.com',
     '$2b$12$LJ3m4ys3Lk0TSwMCPNEJluQMNGxFhGGArSWKhBKFy/B3CehEMxzFi',
     'CSR_TIER1',      'David',   'Kim',       TRUE),

    ('usr-tier2-001',    'rpatel',     'raj.patel@healthins.example.com',
     '$2b$12$LJ3m4ys3Lk0TSwMCPNEJluQMNGxFhGGArSWKhBKFy/B3CehEMxzFi',
     'CSR_TIER2',      'Raj',     'Patel',     TRUE),

    ('usr-tier2-002',    'swilson',    'sarah.wilson@healthins.example.com',
     '$2b$12$LJ3m4ys3Lk0TSwMCPNEJluQMNGxFhGGArSWKhBKFy/B3CehEMxzFi',
     'CSR_TIER2',      'Sarah',   'Wilson',    TRUE),

    ('usr-super-001',    'jchen',      'james.chen@healthins.example.com',
     '$2b$12$LJ3m4ys3Lk0TSwMCPNEJluQMNGxFhGGArSWKhBKFy/B3CehEMxzFi',
     'CSR_SUPERVISOR', 'James',   'Chen',      TRUE),

    ('usr-super-002',    'lnguyen',    'lisa.nguyen@healthins.example.com',
     '$2b$12$LJ3m4ys3Lk0TSwMCPNEJluQMNGxFhGGArSWKhBKFy/B3CehEMxzFi',
     'CSR_SUPERVISOR', 'Lisa',    'Nguyen',    TRUE);

-- ============================================
-- Performance Indexes
-- ============================================

CREATE INDEX idx_audit_logs_timestamp_user ON audit_logs(timestamp, user_id);
CREATE INDEX idx_security_events_timestamp_severity ON security_events(timestamp, severity);
CREATE INDEX idx_agent_metrics_timestamp_agent ON agent_metrics(timestamp, agent_name);
CREATE INDEX idx_tool_metrics_timestamp_tool ON tool_metrics(timestamp, tool_name);

-- ============================================
-- CONTROL 8: INTER-AGENT COMMUNICATION SECURITY
-- ============================================

-- Nonce tracking for replay protection
-- Nonces are primarily tracked in Redis (sub-ms lookups) but persisted
-- to MySQL for audit trail and forensic analysis.
CREATE TABLE IF NOT EXISTS mcp_nonce_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    nonce           VARCHAR(64)  NOT NULL,
    from_agent      VARCHAR(100) NOT NULL,
    to_agent        VARCHAR(100) NOT NULL,
    tool_name       VARCHAR(100) NOT NULL,
    received_at     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    was_replay      BOOLEAN      DEFAULT FALSE,
    INDEX idx_nonce_log_nonce (nonce),
    INDEX idx_nonce_log_agents (from_agent, to_agent),
    INDEX idx_nonce_log_received (received_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Agent key metadata for key rotation auditing
-- Actual keys are stored in Docker Swarm / K8s secrets, not in the DB.
CREATE TABLE IF NOT EXISTS mcp_agent_keys (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_pair      VARCHAR(200) NOT NULL COMMENT 'Canonical pair e.g. central_supervisor:member_services_team',
    key_version     INT          NOT NULL DEFAULT 1,
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    rotated_at      TIMESTAMP    NULL,
    is_active       BOOLEAN      DEFAULT TRUE,
    UNIQUE KEY uk_agent_pair_version (agent_pair, key_version),
    INDEX idx_agent_keys_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Seed initial agent key metadata for the two remote MCP agent pairs
INSERT INTO mcp_agent_keys (agent_pair, key_version, is_active) VALUES
    ('central_supervisor:claim_services_team',  1, TRUE),
    ('central_supervisor:member_services_team', 1, TRUE);

-- Performance indexes for Control 8
CREATE INDEX idx_nonce_log_timestamp ON mcp_nonce_log(received_at);
