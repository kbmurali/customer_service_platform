#!/usr/bin/env python3
"""
test_environment.py : Environment verification script for
Customer Service Intelligence Platform.

Verifies that all required Python packages from requirements.txt
can be imported successfully.

Usage:
    python test_environment.py
"""
import sys
import importlib
from typing import Tuple


# (display_name, actual_import_name)
MODULES_TO_TEST = [
    # Core LangChain stack
    ("langchain", "langchain"),
    ("langchain-core", "langchain_core"),
    ("langchain-community", "langchain_community"),
    ("langsmith", "langsmith"),
    ("langchain-openai", "langchain_openai"),
    ("langchain-anthropic", "langchain_anthropic"),
    ("langchain-aws", "langchain_aws"),
    ("langchain-experimental", "langchain_experimental"),
    ("langchain-mcp-adapters", "langchain_mcp_adapters"),
    # LangGraph stack
    ("langgraph", "langgraph"),
    ("langgraph-checkpoint", "langgraph.checkpoint"),
    ("langgraph-prebuilt", "langgraph.prebuilt"),
    # MCP
    ("mcp", "mcp"),
    # LLM / NLP
    ("transformers", "transformers"),
    ("llmlingua", "llmlingua"),
    # Vector databases
    ("chromadb", "chromadb"),
    # Graph databases
    ("neo4j", "neo4j"),
    # Relational databases
    ("mysql-connector-python", "mysql.connector"),
    ("pymysql", "pymysql"),
    # Redis
    ("redis", "redis"),
    # Web framework
    ("fastapi", "fastapi"),
    ("httpx", "httpx"),
    ("aiohttp", "aiohttp"),
    # Data processing
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("numexpr", "numexpr"),
    # Core
    ("pydantic", "pydantic"),
    ("python-dotenv", "dotenv"),
    # Security - Auth
    ("python-jose", "jose"),
    ("passlib", "passlib"),
    # Security - Input validation
    ("nemoguardrails", "nemoguardrails"),
    # Security - Sanitization
    ("nh3", "nh3"),
    # Security - PII/PHI
    ("presidio-analyzer", "presidio_analyzer"),
    ("presidio-anonymizer", "presidio_anonymizer"),
    # Security - Output validation
    ("guardrails-ai", "guardrails"),
    # Observability
    ("langfuse", "langfuse"),
    ("prometheus-client", "prometheus_client"),
    # Chroma embedding support
    ("pysqlite3-binary", "pysqlite3"),
]


def test_import(display_name: str, import_name: str) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        importlib.import_module(import_name)
        return True, f"  [PASS] {display_name}"
    except ImportError as e:
        return False, f"  [FAIL] {display_name} — {e}"
    except Exception as e:
        return False, f"  [FAIL] {display_name} — unexpected: {e}"


def main():
    """Run environment verification."""
    print("=" * 70)
    print("  Customer Service Intelligence Platform")
    print("  Environment Verification — Module Imports")
    print("=" * 70)

    passed = 0
    failed = 0

    for display_name, import_name in MODULES_TO_TEST:
        success, message = test_import(display_name, import_name)
        print(message)
        if success:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print("\n" + "=" * 70)
    print(f"  Summary: {passed}/{total} modules imported successfully")
    if failed == 0:
        print("  [OK] All imports verified — environment ready.")
    else:
        print(f"  [!!] {failed} module(s) failed — review output above.")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
