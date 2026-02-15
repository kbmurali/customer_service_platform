-- MySQL Relational Database Schema
-- Transactional data, authentication, and metrics

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
    resource_type VARCHAR(50) NOT NULL,  -- MEMBER, CLAIM, PA, POLICY, PROVIDER
    action VARCHAR(50) NOT NULL,  -- READ, WRITE, UPDATE, DELETE
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
    status VARCHAR(20) DEFAULT 'SUCCESS',  -- SUCCESS, FAILED
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
    event_type VARCHAR(100) NOT NULL,  -- ACCESS_DENIED, PERMISSION_VIOLATION, etc.
    severity VARCHAR(20) NOT NULL,  -- LOW, MEDIUM, HIGH, CRITICAL
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
    resource_type VARCHAR(50) NOT NULL,  -- API, TOOL, QUERY
    resource_name VARCHAR(100) NOT NULL,
    request_count INT DEFAULT 0,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    limit_per_window INT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
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
    provider VARCHAR(50) NOT NULL,  -- OPENAI, ANTHROPIC, AWS_BEDROCK
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

-- ============================================
-- Initial Data: Default Permissions
-- ============================================

INSERT INTO permissions (permission_id, permission_name, resource_type, action, description) VALUES
    (UUID(), 'member_read', 'MEMBER', 'READ', 'Read member information'),
    (UUID(), 'member_write', 'MEMBER', 'WRITE', 'Create new member records'),
    (UUID(), 'member_update', 'MEMBER', 'UPDATE', 'Update member information'),
    (UUID(), 'claim_read', 'CLAIM', 'READ', 'Read claim information'),
    (UUID(), 'claim_write', 'CLAIM', 'WRITE', 'Create new claims'),
    (UUID(), 'claim_update', 'CLAIM', 'UPDATE', 'Update claim status'),
    (UUID(), 'pa_read', 'PA', 'READ', 'Read prior authorization information'),
    (UUID(), 'pa_write', 'PA', 'WRITE', 'Create new prior authorizations'),
    (UUID(), 'pa_update', 'PA', 'UPDATE', 'Update prior authorization status'),
    (UUID(), 'policy_read', 'POLICY', 'READ', 'Read policy information'),
    (UUID(), 'provider_read', 'PROVIDER', 'READ', 'Read provider information'),
    (UUID(), 'analytics_read', 'ANALYTICS', 'READ', 'View analytics and reports');

-- ============================================
-- Initial Data: Role Permissions
-- ============================================

-- CSR Tier 1: Basic read access
INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_TIER1', permission_id FROM permissions WHERE permission_name IN (
    'member_read', 'claim_read', 'pa_read', 'policy_read', 'provider_read'
);

-- CSR Tier 2: Read + Write + Update
INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_TIER2', permission_id FROM permissions WHERE permission_name IN (
    'member_read', 'member_update', 'claim_read', 'claim_write', 'claim_update',
    'pa_read', 'pa_write', 'pa_update', 'policy_read', 'provider_read'
);

-- CSR Supervisor: All permissions
INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_SUPERVISOR', permission_id FROM permissions;

-- CSR Read-only: Only read permissions
INSERT INTO role_permissions (role, permission_id)
SELECT 'CSR_READONLY', permission_id FROM permissions WHERE action = 'READ';

-- ============================================
-- Initial Data: Tool Permissions
-- ============================================

-- Define tool permissions for each role
INSERT INTO tool_permissions (tool_permission_id, role, tool_name, is_allowed, rate_limit_per_minute) VALUES
    -- CSR Tier 1
    (UUID(), 'CSR_TIER1', 'member_lookup', TRUE, 30),
    (UUID(), 'CSR_TIER1', 'claim_status_lookup', TRUE, 30),
    (UUID(), 'CSR_TIER1', 'pa_status_lookup', TRUE, 30),
    (UUID(), 'CSR_TIER1', 'provider_search', TRUE, 20),
    (UUID(), 'CSR_TIER1', 'eligibility_check', TRUE, 30),
    
    -- CSR Tier 2
    (UUID(), 'CSR_TIER2', 'member_lookup', TRUE, 60),
    (UUID(), 'CSR_TIER2', 'claim_status_lookup', TRUE, 60),
    (UUID(), 'CSR_TIER2', 'claim_create', TRUE, 20),
    (UUID(), 'CSR_TIER2', 'claim_update', TRUE, 30),
    (UUID(), 'CSR_TIER2', 'pa_status_lookup', TRUE, 60),
    (UUID(), 'CSR_TIER2', 'pa_create', TRUE, 20),
    (UUID(), 'CSR_TIER2', 'pa_update', TRUE, 30),
    (UUID(), 'CSR_TIER2', 'provider_search', TRUE, 40),
    (UUID(), 'CSR_TIER2', 'eligibility_check', TRUE, 60),
    
    -- CSR Supervisor
    (UUID(), 'CSR_SUPERVISOR', 'member_lookup', TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'claim_status_lookup', TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'claim_create', TRUE, 60),
    (UUID(), 'CSR_SUPERVISOR', 'claim_update', TRUE, 60),
    (UUID(), 'CSR_SUPERVISOR', 'pa_status_lookup', TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'pa_create', TRUE, 60),
    (UUID(), 'CSR_SUPERVISOR', 'pa_update', TRUE, 60),
    (UUID(), 'CSR_SUPERVISOR', 'provider_search', TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'eligibility_check', TRUE, 120),
    (UUID(), 'CSR_SUPERVISOR', 'analytics_query', TRUE, 60),
    (UUID(), 'CSR_SUPERVISOR', 'report_generation', TRUE, 30),
    
    -- CSR Read-only
    (UUID(), 'CSR_READONLY', 'member_lookup', TRUE, 20),
    (UUID(), 'CSR_READONLY', 'claim_status_lookup', TRUE, 20),
    (UUID(), 'CSR_READONLY', 'pa_status_lookup', TRUE, 20),
    (UUID(), 'CSR_READONLY', 'provider_search', TRUE, 20),
    (UUID(), 'CSR_READONLY', 'eligibility_check', TRUE, 20);

-- ============================================
-- Initial Data: LLM Configurations
-- ============================================

INSERT INTO llm_configs (config_id, config_name, provider, model_name, api_endpoint, temperature, max_tokens, is_active) VALUES
    (UUID(), 'default_openai', 'OPENAI', 'gpt-4.1-mini', NULL, 0.70, 4096, TRUE),
    (UUID(), 'fast_model', 'OPENAI', 'gpt-4.1-nano', NULL, 0.50, 2048, TRUE),
    (UUID(), 'google_model', 'GOOGLE', 'gemini-2.5-flash', NULL, 0.70, 4096, FALSE);

-- ============================================
-- Indexes for Performance
-- ============================================

-- Additional indexes for common query patterns
CREATE INDEX idx_audit_logs_timestamp_user ON audit_logs(timestamp, user_id);
CREATE INDEX idx_security_events_timestamp_severity ON security_events(timestamp, severity);
CREATE INDEX idx_agent_metrics_timestamp_agent ON agent_metrics(timestamp, agent_name);
CREATE INDEX idx_tool_metrics_timestamp_tool ON tool_metrics(timestamp, tool_name);
