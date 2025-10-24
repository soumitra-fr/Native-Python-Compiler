"""
Phase 2 Integration Tests - AI-Powered Compilation Pipeline

Tests the complete AI compilation pipeline:
RuntimeTracer → TypeInference → StrategyAgent → Compilation

Phase: 2 (Integration)
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.compilation_pipeline import AICompilationPipeline


def test_ai_pipeline_simple():
    """Test 1: Basic AI Pipeline Integration"""
    print("\n" + "=" * 80)
    print("TEST 1: AI Pipeline - Simple Compilation")
    print("=" * 80)
    
    source = """
def add(x: int, y: int) -> int:
    return x + y

def main() -> int:
    result: int = add(10, 32)
    return result
"""
    
    print("Source:")
    print(source)
    
    # Write source file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        source_file = f.name
    
    try:
        # Run AI compilation pipeline
        pipeline = AICompilationPipeline(
            enable_profiling=True,
            enable_type_inference=True,
            enable_strategy_agent=True,
            verbose=False
        )
        
        result = pipeline.compile_intelligently(source_file)
        
        if result.success and result.output_path:
            # Make path absolute
            output_path = os.path.abspath(result.output_path)
            
            # Verify output exists
            if os.path.exists(output_path):
                # Try to run it
                run_result = subprocess.run([output_path], capture_output=True, timeout=5)
                expected = 42  # 10 + 32
                
                if run_result.returncode == expected:
                    print(f"✅ PASSED - Pipeline compiled and executed correctly")
                    print(f"   Strategy: {result.strategy.value}")
                    print(f"   Total Time: {result.metrics.total_time_ms:.2f}ms")
                    print(f"   Result: {run_result.returncode} (expected: {expected})")
                    
                    # Cleanup
                    os.remove(output_path)
                    return True
                else:
                    print(f"❌ FAILED - Wrong result: {run_result.returncode}, expected: {expected}")
                    return False
            else:
                print(f"❌ FAILED - Output file not found: {output_path}")
                return False
        else:
            print(f"❌ FAILED - Compilation failed: {result.error_message}")
            return False
            
    finally:
        if os.path.exists(source_file):
            os.remove(source_file)


def test_ai_pipeline_with_optimization():
    """Test 2: AI Pipeline Selects Optimal Strategy"""
    print("\n" + "=" * 80)
    print("TEST 2: AI Pipeline - Strategy Selection")
    print("=" * 80)
    
    # Code with loops that should benefit from optimization
    source = """
def factorial(n: int) -> int:
    result: int = 1
    i: int = 1
    while i <= n:
        result = result * i
        i = i + 1
    return result

def main() -> int:
    return factorial(5)
"""
    
    print("Source:")
    print(source)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        source_file = f.name
    
    try:
        pipeline = AICompilationPipeline(verbose=False)
        result = pipeline.compile_intelligently(source_file)
        
        if result.success and result.output_path:
            output_path = os.path.abspath(result.output_path)
            
            if os.path.exists(output_path):
                run_result = subprocess.run([output_path], capture_output=True, timeout=5)
                expected = 120  # 5! = 120
                
                if run_result.returncode == expected:
                    print(f"✅ PASSED - Strategy agent chose: {result.strategy.value}")
                    print(f"   Expected speedup: {result.strategy_decision.expected_speedup:.1f}x")
                    print(f"   Reasoning: {result.strategy_decision.reasoning}")
                    print(f"   Result: {run_result.returncode} (expected: {expected})")
                    
                    os.remove(output_path)
                    return True
                else:
                    print(f"❌ FAILED - Wrong result: {run_result.returncode}, expected: {expected}")
                    return False
            else:
                print(f"❌ FAILED - Output not found")
                return False
        else:
            print(f"❌ FAILED - {result.error_message}")
            return False
            
    finally:
        if os.path.exists(source_file):
            os.remove(source_file)


def test_ai_pipeline_type_inference():
    """Test 3: AI Pipeline - Type Inference Integration"""
    print("\n" + "=" * 80)
    print("TEST 3: AI Pipeline - Type Inference")
    print("=" * 80)
    
    source = """
def compute(x: int, y: int) -> int:
    total: int = x + y
    double: int = total * 2
    return double

def main() -> int:
    return compute(15, 15)
"""
    
    print("Source:")
    print(source)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        source_file = f.name
    
    try:
        pipeline = AICompilationPipeline(
            enable_type_inference=True,
            verbose=False
        )
        result = pipeline.compile_intelligently(source_file)
        
        if result.success and result.output_path:
            output_path = os.path.abspath(result.output_path)
            
            if os.path.exists(output_path):
                run_result = subprocess.run([output_path], capture_output=True, timeout=5)
                expected = 60  # (15 + 15) * 2 = 60
                
                if run_result.returncode == expected:
                    print(f"✅ PASSED - Type inference worked")
                    print(f"   Inferred {result.metrics.types_inferred} type(s)")
                    print(f"   Result: {run_result.returncode} (expected: {expected})")
                    
                    os.remove(output_path)
                    return True
                else:
                    print(f"❌ FAILED - Wrong result: {run_result.returncode}, expected: {expected}")
                    return False
            else:
                print(f"❌ FAILED - Output not found")
                return False
        else:
            print(f"❌ FAILED - {result.error_message}")
            return False
            
    finally:
        if os.path.exists(source_file):
            os.remove(source_file)


def test_ai_pipeline_metrics():
    """Test 4: AI Pipeline - Metrics Collection"""
    print("\n" + "=" * 80)
    print("TEST 4: AI Pipeline - Metrics Collection")
    print("=" * 80)
    
    source = """
def triple(n: int) -> int:
    return n * 3

def main() -> int:
    return triple(7)
"""
    
    print("Source:")
    print(source)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        source_file = f.name
    
    try:
        pipeline = AICompilationPipeline(verbose=False)
        result = pipeline.compile_intelligently(source_file)
        
        if result.success:
            # Check that metrics were collected
            metrics = result.metrics
            
            checks = [
                ("Total time", metrics.total_time_ms > 0),
                ("Profiling time", metrics.profiling_time_ms >= 0),
                ("Type inference time", metrics.type_inference_time_ms >= 0),
                ("Strategy selection time", metrics.strategy_selection_time_ms >= 0),
                ("Compilation time", metrics.compilation_time_ms > 0),
            ]
            
            all_passed = all(check[1] for check in checks)
            
            if all_passed:
                print(f"✅ PASSED - All metrics collected")
                for name, passed in checks:
                    print(f"   ✓ {name}: collected")
                print(f"   Total pipeline time: {metrics.total_time_ms:.2f}ms")
                
                # Cleanup
                if result.output_path and os.path.exists(result.output_path):
                    os.remove(result.output_path)
                return True
            else:
                print(f"❌ FAILED - Some metrics missing")
                for name, passed in checks:
                    status = "✓" if passed else "✗"
                    print(f"   {status} {name}")
                return False
        else:
            print(f"❌ FAILED - {result.error_message}")
            return False
            
    finally:
        if os.path.exists(source_file):
            os.remove(source_file)


def test_ai_pipeline_all_stages():
    """Test 5: AI Pipeline - All Stages Working"""
    print("\n" + "=" * 80)
    print("TEST 5: AI Pipeline - All Stages Integration")
    print("=" * 80)
    
    source = """
def power(base: int, exp: int) -> int:
    result: int = 1
    i: int = 0
    while i < exp:
        result = result * base
        i = i + 1
    return result

def main() -> int:
    return power(2, 5)
"""
    
    print("Source:")
    print(source)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        source_file = f.name
    
    try:
        # Run with all AI features enabled
        pipeline = AICompilationPipeline(
            enable_profiling=True,
            enable_type_inference=True,
            enable_strategy_agent=True,
            verbose=False
        )
        
        result = pipeline.compile_intelligently(source_file)
        
        if result.success and result.output_path:
            output_path = os.path.abspath(result.output_path)
            
            if os.path.exists(output_path):
                run_result = subprocess.run([output_path], capture_output=True, timeout=5)
                expected = 32  # 2^5 = 32
                
                # Check all components worked
                checks = [
                    ("Compilation", result.success),
                    ("Correct output", run_result.returncode == expected),
                    ("Strategy selected", result.strategy_decision is not None),
                    ("Metrics collected", result.metrics.total_time_ms > 0),
                ]
                
                all_passed = all(check[1] for check in checks)
                
                if all_passed:
                    print(f"✅ PASSED - Full AI pipeline working!")
                    print(f"   Strategy: {result.strategy.value}")
                    print(f"   Expected speedup: {result.strategy_decision.expected_speedup:.1f}x")
                    print(f"   Total time: {result.metrics.total_time_ms:.2f}ms")
                    print(f"   Result: {run_result.returncode} (expected: {expected})")
                    
                    os.remove(output_path)
                    return True
                else:
                    print(f"❌ FAILED - Some components failed")
                    for name, passed in checks:
                        status = "✓" if passed else "✗"
                        print(f"   {status} {name}")
                    return False
            else:
                print(f"❌ FAILED - Output not found")
                return False
        else:
            print(f"❌ FAILED - {result.error_message}")
            return False
            
    finally:
        if os.path.exists(source_file):
            os.remove(source_file)


def main():
    """Run all Phase 2 integration tests"""
    print("\n" + "=" * 80)
    print("PHASE 2 INTEGRATION TESTS - AI COMPILATION PIPELINE")
    print("=" * 80)
    
    tests = [
        test_ai_pipeline_simple,
        test_ai_pipeline_with_optimization,
        test_ai_pipeline_type_inference,
        test_ai_pipeline_metrics,
        test_ai_pipeline_all_stages,
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"❌ Test raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - Test {i}: {test.__doc__.strip()}")
    
    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 ALL PHASE 2 TESTS PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
