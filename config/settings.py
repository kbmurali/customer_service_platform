"""
Configuration settings for Health Insurance AI Platform
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Health Insurance AI Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # Neo4j Knowledge Graph
    NEO4J_KG_URI: str = os.getenv("NEO4J_KG_URI", "bolt://neo4j-kg:7687")
    NEO4J_KG_USER: str = os.getenv("NEO4J_KG_USER", "neo4j")
    NEO4J_KG_PASSWORD: str = os.getenv("NEO4J_KG_PASSWORD", "")
    NEO4J_KG_DATABASE: str = "neo4j"
    
    # Neo4j Context Graph
    NEO4J_CG_URI: str = os.getenv("NEO4J_CG_URI", "bolt://neo4j-cg:7687")
    NEO4J_CG_USER: str = os.getenv("NEO4J_CG_USER", "neo4j")
    NEO4J_CG_PASSWORD: str = os.getenv("NEO4J_CG_PASSWORD", "")
    NEO4J_CG_DATABASE: str = "neo4j"
    
    # MySQL
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "mysql")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "health_insurance")
    
    # Chroma Vector DB
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "chroma")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_PERSIST_DIRECTORY: str = "/data/chroma"
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4.1-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 4096
    
    # LangSmith
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = "health-insurance-ai-platform"
    LANGSMITH_TRACING: bool = True
    
    # LangFuse
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://langfuse:3000")
    
    # Prometheus
    PROMETHEUS_PORT: int = 9090
    METRICS_ENABLED: bool = True
    
    # Grafana
    GRAFANA_PORT: int = 3001
    GRAFANA_ADMIN_USER: str = "admin"
    GRAFANA_ADMIN_PASSWORD: str = os.getenv("GRAFANA_ADMIN_PASSWORD", "")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    # Agent Configuration
    AGENT_TIMEOUT_SECONDS: int = 300
    AGENT_MAX_RETRIES: int = 3
    AGENT_RETRY_DELAY_SECONDS: int = 2
    
    # Supervisor Configuration
    SUPERVISOR_MAX_ITERATIONS: int = 50
    SUPERVISOR_TIMEOUT_SECONDS: int = 600
    
    # Tool Configuration
    TOOL_TIMEOUT_SECONDS: int = 30
    TOOL_MAX_RETRIES: int = 2
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[str] = "/var/log/health-insurance-ai/app.log"
    
    # MCP Server
    MCP_ENABLED: bool = True
    MCP_PORT: int = 8001
    
    # A2A Configuration
    A2A_ENABLED: bool = True
    A2A_PORT: int = 8002
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_database_url() -> str:
    """Get MySQL database URL"""
    return (
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
    )


def get_neo4j_kg_config() -> dict:
    """Get Neo4j Knowledge Graph configuration"""
    return {
        "uri": settings.NEO4J_KG_URI,
        "auth": (settings.NEO4J_KG_USER, settings.NEO4J_KG_PASSWORD),
        "database": settings.NEO4J_KG_DATABASE,
    }


def get_neo4j_cg_config() -> dict:
    """Get Neo4j Context Graph configuration"""
    return {
        "uri": settings.NEO4J_CG_URI,
        "auth": (settings.NEO4J_CG_USER, settings.NEO4J_CG_PASSWORD),
        "database": settings.NEO4J_CG_DATABASE,
    }


def get_chroma_config() -> dict:
    """Get Chroma configuration"""
    return {
        "host": settings.CHROMA_HOST,
        "port": settings.CHROMA_PORT,
        "persist_directory": settings.CHROMA_PERSIST_DIRECTORY,
    }
