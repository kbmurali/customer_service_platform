-- MySQL Schema for Feedback and Learning System

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    execution_id VARCHAR(36) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    team_name VARCHAR(100) NOT NULL,
    
    -- User feedback
    rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    feedback_type ENUM('positive', 'negative', 'neutral') NOT NULL,
    comment TEXT,
    tags JSON,
    
    -- Context
    user_query TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    tools_used JSON,
    execution_time FLOAT,
    
    -- Implicit signals
    follow_up_count INT DEFAULT 0,
    escalated BOOLEAN DEFAULT FALSE,
    task_completed BOOLEAN DEFAULT FALSE,
    session_duration FLOAT,
    
    -- Metadata
    user_id VARCHAR(36) NOT NULL,
    user_role VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_agent_name (agent_name),
    INDEX idx_team_name (team_name),
    INDEX idx_rating (rating),
    INDEX idx_feedback_type (feedback_type),
    INDEX idx_created_at (created_at),
    INDEX idx_session_id (session_id),
    INDEX idx_execution_id (execution_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Learning insights table
CREATE TABLE IF NOT EXISTS learning_insights (
    insight_id VARCHAR(36) PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    team_name VARCHAR(100) NOT NULL,
    
    -- Insight details
    insight_type VARCHAR(50) NOT NULL,  -- prompt_issue, tool_misuse, knowledge_gap, success_pattern, error_pattern
    description TEXT NOT NULL,
    pattern TEXT,
    frequency INT DEFAULT 1,
    
    -- Recommendations
    recommended_action VARCHAR(255),
    priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    status ENUM('pending', 'in_progress', 'applied', 'rejected') DEFAULT 'pending',
    
    -- Impact
    affected_queries INT DEFAULT 0,
    potential_improvement FLOAT,
    
    -- Metadata
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP NULL,
    
    -- Indexes
    INDEX idx_agent_name (agent_name),
    INDEX idx_team_name (team_name),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_insight_type (insight_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Prompt versions table
CREATE TABLE IF NOT EXISTS prompt_versions (
    version_id VARCHAR(36) PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    
    -- Prompt content
    prompt_text TEXT NOT NULL,
    version_number INT NOT NULL,
    
    -- Performance metrics
    avg_rating FLOAT,
    success_rate FLOAT,
    avg_execution_time FLOAT,
    total_uses INT DEFAULT 0,
    
    -- A/B testing
    is_active BOOLEAN DEFAULT TRUE,
    traffic_percentage INT DEFAULT 100 CHECK (traffic_percentage BETWEEN 0 AND 100),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    notes TEXT,
    
    -- Indexes
    INDEX idx_agent_name (agent_name),
    INDEX idx_is_active (is_active),
    INDEX idx_version_number (version_number),
    UNIQUE KEY uk_agent_version (agent_name, version_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Feedback patterns table
CREATE TABLE IF NOT EXISTS feedback_patterns (
    pattern_id VARCHAR(36) PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,  -- issue_pattern, success_pattern
    description TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    occurrences INT DEFAULT 1,
    
    -- Pattern details
    common_keywords JSON,
    common_tags JSON,
    avg_rating FLOAT,
    
    -- Related agents
    agent_names JSON,
    
    -- Metadata
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_pattern_type (pattern_type),
    INDEX idx_confidence (confidence),
    INDEX idx_occurrences (occurrences)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent metrics table
CREATE TABLE IF NOT EXISTS agent_metrics (
    metric_id VARCHAR(36) PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    period VARCHAR(20) NOT NULL,  -- day, week, month
    
    -- Core metrics
    avg_rating FLOAT,
    success_rate FLOAT,
    resolution_rate FLOAT,
    avg_response_time FLOAT,
    total_interactions INT DEFAULT 0,
    
    -- Feedback distribution
    positive_count INT DEFAULT 0,
    negative_count INT DEFAULT 0,
    neutral_count INT DEFAULT 0,
    
    -- Trends
    rating_trend FLOAT,
    success_rate_trend FLOAT,
    response_time_trend FLOAT,
    
    -- Metadata
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_agent_name (agent_name),
    INDEX idx_period (period),
    INDEX idx_period_start (period_start),
    UNIQUE KEY uk_agent_period (agent_name, period, period_start)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Fine-tuning datasets table
CREATE TABLE IF NOT EXISTS finetuning_datasets (
    dataset_id VARCHAR(36) PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    
    -- Dataset details
    total_examples INT NOT NULL,
    min_rating INT NOT NULL,
    date_range_start TIMESTAMP NOT NULL,
    date_range_end TIMESTAMP NOT NULL,
    
    -- File info
    filename VARCHAR(255) NOT NULL,
    file_size BIGINT,
    
    -- Status
    status VARCHAR(50) DEFAULT 'prepared',  -- prepared, uploaded, training, completed, failed
    openai_file_id VARCHAR(100),
    finetuning_job_id VARCHAR(100),
    finetuned_model_id VARCHAR(100),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    completed_at TIMESTAMP NULL,
    
    -- Indexes
    INDEX idx_agent_name (agent_name),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Error patterns table
CREATE TABLE IF NOT EXISTS error_patterns (
    pattern_id VARCHAR(36) PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    
    -- Pattern details
    description TEXT NOT NULL,
    error_type VARCHAR(100) NOT NULL,
    frequency INT DEFAULT 1,
    
    -- Prevention
    prevention_strategy TEXT,
    warning_message TEXT,
    
    -- Impact
    affected_queries INT DEFAULT 0,
    avg_rating_impact FLOAT,
    
    -- Metadata
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_occurrence TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_agent_name (agent_name),
    INDEX idx_error_type (error_type),
    INDEX idx_frequency (frequency)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create views for common queries

-- Agent performance summary view
CREATE OR REPLACE VIEW agent_performance_summary AS
SELECT 
    agent_name,
    COUNT(*) as total_feedback,
    AVG(rating) as avg_rating,
    SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) / COUNT(*) * 100 as success_rate,
    SUM(CASE WHEN escalated = FALSE THEN 1 ELSE 0 END) / COUNT(*) * 100 as resolution_rate,
    AVG(execution_time) as avg_response_time,
    SUM(CASE WHEN feedback_type = 'positive' THEN 1 ELSE 0 END) as positive_count,
    SUM(CASE WHEN feedback_type = 'negative' THEN 1 ELSE 0 END) as negative_count,
    MAX(created_at) as last_feedback_at
FROM feedback
GROUP BY agent_name;

-- Recent insights view
CREATE OR REPLACE VIEW recent_insights AS
SELECT 
    i.*,
    COUNT(f.feedback_id) as related_feedback_count
FROM learning_insights i
LEFT JOIN feedback f ON f.agent_name = i.agent_name 
    AND f.created_at >= i.discovered_at
WHERE i.status IN ('pending', 'in_progress')
GROUP BY i.insight_id
ORDER BY i.priority DESC, i.discovered_at DESC;

-- Prompt performance comparison view
CREATE OR REPLACE VIEW prompt_performance_comparison AS
SELECT 
    agent_name,
    version_id,
    version_number,
    avg_rating,
    success_rate,
    avg_execution_time,
    total_uses,
    is_active,
    traffic_percentage,
    created_at
FROM prompt_versions
WHERE is_active = TRUE
ORDER BY agent_name, version_number DESC;
