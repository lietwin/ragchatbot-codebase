#!/usr/bin/env python3
"""
Test script to diagnose the real RAG system issues
"""

import sys
import os
from config import config
from rag_system import RAGSystem


def test_system_initialization():
    """Test if RAG system initializes properly"""
    print("=== Testing RAG System Initialization ===")

    try:
        rag_system = RAGSystem(config)
        print("✓ RAG system initialized successfully")
        return rag_system
    except Exception as e:
        print(f"✗ RAG system initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_vector_store(rag_system):
    """Test vector store functionality"""
    print("\n=== Testing Vector Store ===")

    try:
        # Check if courses exist
        course_titles = rag_system.vector_store.get_existing_course_titles()
        print(f"Found {len(course_titles)} existing courses: {course_titles}")

        if not course_titles:
            print("⚠ Warning: No courses found in vector store")
            return False

        # Test direct vector store search
        print("Testing direct vector store search...")
        search_result = rag_system.vector_store.search("MCP")
        print(
            f"Search result - Documents: {len(search_result.documents)}, Error: {search_result.error}"
        )

        if search_result.error:
            print(f"✗ Vector store search error: {search_result.error}")
            return False
        elif search_result.is_empty():
            print("⚠ Vector store search returned no results")
            return False
        else:
            print("✓ Vector store search successful")
            return True

    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_search_tool(rag_system):
    """Test CourseSearchTool directly"""
    print("\n=== Testing CourseSearchTool ===")

    try:
        # Test search tool directly
        search_tool = rag_system.search_tool
        result = search_tool.execute("What is MCP?")

        if "error" in result.lower() or "failed" in result.lower():
            print(f"✗ Search tool error: {result}")
            return False
        elif "no relevant content" in result.lower():
            print("⚠ Search tool found no content")
            return False
        else:
            print("✓ Search tool executed successfully")
            print(f"Result preview: {result[:200]}...")
            return True

    except Exception as e:
        print(f"✗ Search tool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tool_manager(rag_system):
    """Test ToolManager functionality"""
    print("\n=== Testing ToolManager ===")

    try:
        # Check tool definitions
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        print(f"Found {len(tool_definitions)} tools:")
        for tool_def in tool_definitions:
            print(
                f"  - {tool_def.get('name', 'Unknown')}: {tool_def.get('description', 'No description')}"
            )

        # Test tool execution
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="test query"
        )

        if "Tool 'search_course_content' not found" in result:
            print("✗ Search tool not found in tool manager")
            return False
        else:
            print("✓ Tool manager executed search tool successfully")
            return True

    except Exception as e:
        print(f"✗ Tool manager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ai_generator(rag_system):
    """Test AI generator without API call"""
    print("\n=== Testing AI Generator (Configuration) ===")

    try:
        ai_gen = rag_system.ai_generator
        print(f"API Key configured: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
        print(f"Model: {ai_gen.model}")
        print(
            f"System prompt contains tools: {'search_course_content' in ai_gen.SYSTEM_PROMPT}"
        )

        # Check if API key is properly configured
        if not config.ANTHROPIC_API_KEY:
            print("✗ Anthropic API key not configured")
            return False

        print("✓ AI generator configuration looks correct")
        return True

    except Exception as e:
        print(f"✗ AI generator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_loading(rag_system):
    """Test loading course data"""
    print("\n=== Testing Data Loading ===")

    try:
        docs_path = "../docs"
        if not os.path.exists(docs_path):
            print(f"✗ Docs directory not found: {docs_path}")
            return False

        # List available documents
        doc_files = [
            f for f in os.listdir(docs_path) if f.endswith((".pdf", ".txt", ".docx"))
        ]
        print(f"Found {len(doc_files)} document files: {doc_files}")

        # Check if data is already loaded
        existing_courses = rag_system.vector_store.get_existing_course_titles()
        print(f"Existing courses in DB: {len(existing_courses)}")

        if not existing_courses:
            print("Attempting to load course data...")
            courses_added, chunks_added = rag_system.add_course_folder(docs_path)
            print(f"Added {courses_added} courses, {chunks_added} chunks")

            if courses_added == 0:
                print("⚠ Warning: No courses were added")
                return False

        print("✓ Course data is available")
        return True

    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end_query(rag_system):
    """Test a real end-to-end query"""
    print("\n=== Testing End-to-End Query ===")

    try:
        # Test a simple content query
        print("Testing content query...")
        response, sources = rag_system.query("What is MCP?")

        print(f"Response: {response[:200] if response else 'No response'}...")
        print(f"Sources: {len(sources)} sources returned")

        if not response:
            print("✗ No response received")
            return False
        elif "query failed" in response.lower():
            print("✗ Query failed error")
            return False
        elif "error" in response.lower():
            print(f"⚠ Response contains error: {response}")
            return False
        else:
            print("✓ End-to-end query successful")
            return True

    except Exception as e:
        print(f"✗ End-to-end query failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests"""
    print("RAG System Diagnostic Test")
    print("=" * 50)

    # Initialize system
    rag_system = test_system_initialization()
    if not rag_system:
        print("\n❌ CRITICAL: System initialization failed - cannot continue")
        return

    # Run diagnostic tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Vector Store", test_vector_store),
        ("Search Tool", test_search_tool),
        ("Tool Manager", test_tool_manager),
        ("AI Generator Config", test_ai_generator),
        ("End-to-End Query", test_end_to_end_query),
    ]

    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func(rag_system)

    # Print summary
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{test_name:20} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system should be working correctly.")
    else:
        print("🔍 Some tests failed. Check the details above for specific issues.")

        # Provide specific guidance
        if not results.get("Data Loading"):
            print(
                "\n💡 ISSUE: Data loading failed - course documents may not be loaded"
            )
        if not results.get("Vector Store"):
            print(
                "\n💡 ISSUE: Vector store problems - database may be empty or corrupted"
            )
        if not results.get("AI Generator Config"):
            print("\n💡 ISSUE: AI generator configuration - check API key")


if __name__ == "__main__":
    main()
