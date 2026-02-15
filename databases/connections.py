"""
Database connection managers for Neo4j, MySQL, and Chroma
"""
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver
import pymysql
from pymysql.cursors import DictCursor
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import get_settings, get_neo4j_kg_config, get_neo4j_cg_config, get_chroma_config

logger = logging.getLogger(__name__)
settings = get_settings()


class Neo4jKGConnection:
    """Neo4j Knowledge Graph connection manager"""
    
    def __init__(self):
        self.driver: Optional[Driver] = None
        self.config = get_neo4j_kg_config()
    
    def connect(self) -> Driver:
        """Establish connection to Neo4j KG"""
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self.config["uri"],
                    auth=self.config["auth"]
                )
                # Verify connectivity
                self.driver.verify_connectivity()
                logger.info("Connected to Neo4j Knowledge Graph")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j KG: {e}")
                raise
        return self.driver
    
    def close(self):
        """Close Neo4j KG connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Closed Neo4j Knowledge Graph connection")
    
    @contextmanager
    def session(self, database: Optional[str] = None):
        """Context manager for Neo4j session"""
        driver = self.connect()
        db = database or self.config["database"]
        session = driver.session(database=db)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> list:
        """Execute a Cypher query and return results"""
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]


class Neo4jCGConnection:
    """Neo4j Context Graph connection manager"""
    
    def __init__(self):
        self.driver: Optional[Driver] = None
        self.config = get_neo4j_cg_config()
    
    def connect(self) -> Driver:
        """Establish connection to Neo4j CG"""
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self.config["uri"],
                    auth=self.config["auth"]
                )
                # Verify connectivity
                self.driver.verify_connectivity()
                logger.info("Connected to Neo4j Context Graph")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j CG: {e}")
                raise
        return self.driver
    
    def close(self):
        """Close Neo4j CG connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Closed Neo4j Context Graph connection")
    
    @contextmanager
    def session(self, database: Optional[str] = None):
        """Context manager for Neo4j session"""
        driver = self.connect()
        db = database or self.config["database"]
        session = driver.session(database=db)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> list:
        """Execute a Cypher query and return results"""
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]


class MySQLConnection:
    """MySQL connection manager"""
    
    def __init__(self):
        self.connection = None
    
    def connect(self):
        """Establish connection to MySQL"""
        if self.connection is None or not self.connection.open:
            try:
                self.connection = pymysql.connect(
                    host=settings.MYSQL_HOST,
                    port=settings.MYSQL_PORT,
                    user=settings.MYSQL_USER,
                    password=settings.MYSQL_PASSWORD,
                    database=settings.MYSQL_DATABASE,
                    cursorclass=DictCursor,
                    autocommit=False
                )
                logger.info("Connected to MySQL")
            except Exception as e:
                logger.error(f"Failed to connect to MySQL: {e}")
                raise
        return self.connection
    
    def close(self):
        """Close MySQL connection"""
        if self.connection and self.connection.open:
            self.connection.close()
            self.connection = None
            logger.info("Closed MySQL connection")
    
    @contextmanager
    def cursor(self):
        """Context manager for MySQL cursor"""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"MySQL transaction failed: {e}")
            raise
        finally:
            cursor.close()
    
    def execute_query(self, query: str, parameters: Optional[tuple] = None) -> list:
        """Execute a SQL query and return results"""
        with self.cursor() as cursor:
            cursor.execute(query, parameters or ())
            return cursor.fetchall()
    
    def execute_update(self, query: str, parameters: Optional[tuple] = None) -> int:
        """Execute an UPDATE/INSERT/DELETE query and return affected rows"""
        with self.cursor() as cursor:
            affected = cursor.execute(query, parameters or ())
            return affected


class ChromaConnection:
    """Chroma vector database connection manager"""
    
    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.config = get_chroma_config()
    
    def connect(self) -> chromadb.Client:
        """Establish connection to Chroma"""
        if self.client is None:
            try:
                # Use HTTP client for containerized deployment
                self.client = chromadb.HttpClient(
                    host=self.config["host"],
                    port=self.config["port"],
                    settings=ChromaSettings(
                        anonymized_telemetry=False
                    )
                )
                # Test connection
                self.client.heartbeat()
                logger.info("Connected to Chroma vector database")
            except Exception as e:
                logger.error(f"Failed to connect to Chroma: {e}")
                raise
        return self.client
    
    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Get or create a collection"""
        client = self.connect()
        return client.get_or_create_collection(
            name=name,
            metadata=metadata or {}
        )
    
    def get_collection(self, name: str):
        """Get an existing collection"""
        client = self.connect()
        return client.get_collection(name=name)
    
    def list_collections(self) -> list:
        """List all collections"""
        client = self.connect()
        return client.list_collections()


# Global connection instances
neo4j_kg_conn = Neo4jKGConnection()
neo4j_cg_conn = Neo4jCGConnection()
mysql_conn = MySQLConnection()
chroma_conn = ChromaConnection()


def get_neo4j_kg() -> Neo4jKGConnection:
    """Get Neo4j Knowledge Graph connection"""
    return neo4j_kg_conn


def get_neo4j_cg() -> Neo4jCGConnection:
    """Get Neo4j Context Graph connection"""
    return neo4j_cg_conn


def get_mysql() -> MySQLConnection:
    """Get MySQL connection"""
    return mysql_conn


def get_chroma() -> ChromaConnection:
    """Get Chroma connection"""
    return chroma_conn


def close_all_connections():
    """Close all database connections"""
    neo4j_kg_conn.close()
    neo4j_cg_conn.close()
    mysql_conn.close()
    logger.info("All database connections closed")
