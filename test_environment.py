#!/usr/bin/env python3
"""
Environment verification script for Customer Service Intelligence Platform.
"""
import sys
import importlib
from typing import List, Tuple

def test_import(module_name: str) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        importlib.import_module(module_name)
        return True, f"===> {module_name} imported successfully"
    except ImportError as e:
        return False, f"!!!=> {module_name} failed to import: {str(e)}"

def main():
    """Run environment verification tests."""
    print(" Customer Service Intelligence Platform - Environment Verification")
    print("\n" + "=" * 70)

    # Core dependencies to test
    modules_to_test = [
		"langchain",
		"langchain_community",
		"langgraph",
		"langchain_openai",
		"langchain_anthropic",
		"langsmith",
		"langchain_mcp_adapters",
		"mcp",
		"pysqlite3-binary",
		"chromadb",
		"pydantic",
		"dotenv",
		"pandas",
		"numpy",
		"numexpr"
	]
    
    results = []
    
    for module in modules_to_test:
        success, message = test_import(module)
        results.append((success, message))
        print(message)

    # Summary
    successful = sum(1 for success, _ in results if success)
    total = len(results)
    print("\n" + "=" * 70)
    
    print(f" Summary: {successful}/{total} modules imported successfully")
    
    if successful == total:
        print("===> Environment setup complete! Ready to proceed with development.")
        return 0
    else:
        print("!!!=> Some modules failed to import. Please check your  Some modules failed to import. Please check your installation.")
        return 1

if __name__ == "__main__":
	sys.exit(main())
