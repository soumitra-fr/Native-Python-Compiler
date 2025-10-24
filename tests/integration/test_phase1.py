"""
Phase 1 Integration Test - End-to-End Compiler Pipeline

Tests the complete compilation pipeline:
Python Source → AST → IR → LLVM → Native Binary

Phase: 1 (Integration)
"""

import subprocess
import tempfile
import os
from pathlib import Path

from compiler.backend.codegen import CompilerPipeline


def test_simple_arithmetic():
    """Test 1: Simple arithmetic"""
    print("\n" + "=" * 80)
    print("TEST 1: Simple Arithmetic")
    print("=" * 80)
    
    source = """
def compute(a: int, b: int) -> int:
    return a + b * 2

def main() -> int:
    result: int = compute(5, 10)
    return result
"""
    
    print("Source:")
    print(source)
    
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        pipeline = CompilerPipeline()
        success = pipeline.compile_source(source, output_path, optimize=True, verbose=False)
        
        if success:
            result = subprocess.run([output_path], capture_output=True, timeout=5)
            expected = 5 + 10 * 2  # Should be 25
            if result.returncode == expected:
                print(f"✅ PASSED - Result: {result.returncode} (expected: {expected})")
                return True
            else:
                print(f"❌ FAILED - Result: {result.returncode}, expected: {expected}")
                return False
        else:
            print("❌ FAILED - Compilation failed")
            return False
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_control_flow():
    """Test 2: If/else control flow"""
    print("\n" + "=" * 80)
    print("TEST 2: Control Flow (if/else)")
    print("=" * 80)
    
    source = """
def max_value(a: int, b: int) -> int:
    if a > b:
        return a
    else:
        return b

def main() -> int:
    result: int = max_value(42, 17)
    return result
"""
    
    print("Source: max_value(42, 17)")
    
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        pipeline = CompilerPipeline()
        success = pipeline.compile_source(source, output_path, optimize=True, verbose=False)
        
        if success:
            result = subprocess.run([output_path], capture_output=True, timeout=5)
            expected = 42  # max(42, 17) = 42
            if result.returncode == expected:
                print(f"✅ PASSED - Result: {result.returncode} (expected: {expected})")
                return True
            else:
                print(f"❌ FAILED - Result: {result.returncode}, expected: {expected}")
                return False
        else:
            print("❌ FAILED - Compilation failed")
            return False
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_loops():
    """Test 3: For loops"""
    print("\n" + "=" * 80)
    print("TEST 3: Loops (for with range)")
    print("=" * 80)
    
    source = """
def sum_range(n: int) -> int:
    total: int = 0
    for i in range(n):
        total += i
    return total

def main() -> int:
    result: int = sum_range(10)
    return result
"""
    
    print("Source: sum_range(10)")
    
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        pipeline = CompilerPipeline()
        success = pipeline.compile_source(source, output_path, optimize=True, verbose=False)
        
        if success:
            result = subprocess.run([output_path], capture_output=True, timeout=5)
            expected = sum(range(10))  # 0+1+2+...+9 = 45
            if result.returncode == expected:
                print(f"✅ PASSED - Result: {result.returncode} (expected: {expected})")
                return True
            else:
                print(f"❌ FAILED - Result: {result.returncode}, expected: {expected}")
                return False
        else:
            print("❌ FAILED - Compilation failed")
            return False
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_nested_calls():
    """Test 4: Nested function calls"""
    print("\n" + "=" * 80)
    print("TEST 4: Nested Function Calls")
    print("=" * 80)
    
    source = """
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

def compute(x: int) -> int:
    temp1: int = add(x, 5)
    temp2: int = multiply(temp1, 2)
    return temp2

def main() -> int:
    result: int = compute(10)
    return result
"""
    
    print("Source: compute(10) = (10+5)*2")
    
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        pipeline = CompilerPipeline()
        success = pipeline.compile_source(source, output_path, optimize=True, verbose=False)
        
        if success:
            result = subprocess.run([output_path], capture_output=True, timeout=5)
            expected = (10 + 5) * 2  # 30
            if result.returncode == expected:
                print(f"✅ PASSED - Result: {result.returncode} (expected: {expected})")
                return True
            else:
                print(f"❌ FAILED - Result: {result.returncode}, expected: {expected}")
                return False
        else:
            print("❌ FAILED - Compilation failed")
            return False
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_complex_expression():
    """Test 5: Complex expressions"""
    print("\n" + "=" * 80)
    print("TEST 5: Complex Expressions")
    print("=" * 80)
    
    source = """
def main() -> int:
    a: int = 10
    b: int = 20
    c: int = 5
    result: int = (a + b) * c - a
    return result
"""
    
    print("Source: (10 + 20) * 5 - 10")
    
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        pipeline = CompilerPipeline()
        success = pipeline.compile_source(source, output_path, optimize=True, verbose=False)
        
        if success:
            result = subprocess.run([output_path], capture_output=True, timeout=5)
            expected = (10 + 20) * 5 - 10  # 140
            if result.returncode == expected:
                print(f"✅ PASSED - Result: {result.returncode} (expected: {expected})")
                return True
            else:
                print(f"❌ FAILED - Result: {result.returncode}, expected: {expected}")
                return False
        else:
            print("❌ FAILED - Compilation failed")
            return False
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 1 INTEGRATION TESTS")
    print("Python → AST → IR → LLVM → Native Binary")
    print("=" * 80)
    
    tests = [
        ("Simple Arithmetic", test_simple_arithmetic),
        ("Control Flow", test_control_flow),
        ("Loops", test_loops),
        ("Nested Calls", test_nested_calls),
        ("Complex Expressions", test_complex_expression),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAILED - Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\n✅ Phase 1 Complete: Core Compiler Working!")
        print("   - Python source code ✅")
        print("   - AST parsing ✅")
        print("   - Semantic analysis ✅")
        print("   - IR generation ✅")
        print("   - LLVM IR generation ✅")
        print("   - Native code compilation ✅")
        print("   - Standalone executables ✅")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    print("\n" + "=" * 80)
