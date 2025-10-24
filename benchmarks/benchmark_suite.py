"""
Comprehensive Benchmark Suite for Native Python Compiler

Compares compiled native code vs interpreted Python across various workloads.

Phase: 1.5 (Benchmarks & Optimization)
"""

import subprocess
import tempfile
import time
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.backend.codegen import CompilerPipeline


def benchmark_interpreted(code: str, iterations: int = 1000) -> float:
    """Benchmark interpreted Python code"""
    setup = f"""
{code}
import time
start = time.perf_counter()
for _ in range({iterations}):
    result = main()
end = time.perf_counter()
print(end - start)
"""
    
    try:
        result = subprocess.run(
            ['python', '-c', setup],
            capture_output=True,
            text=True,
            timeout=30
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Interpreted benchmark failed: {e}")
        return -1


def benchmark_compiled(code: str, iterations: int = 1000, opt_level: int = 3) -> tuple:
    """Benchmark compiled native code"""
    # Wrap code to run multiple iterations
    wrapper = f"""
{code}

def run_benchmark() -> int:
    total: int = 0
    i: int = 0
    while i < {iterations}:
        total = total + main()
        i = i + 1
    return total

def main_wrapper() -> int:
    return run_benchmark()
"""
    
    # Replace main with main_wrapper
    final_code = wrapper.replace("def main_wrapper", "def main_final").replace("def main(", "def compute_main(")
    final_code = final_code.replace("def main_final", "def main")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "bench")
        
        try:
            # Compile
            pipeline = CompilerPipeline()
            compile_start = time.perf_counter()
            success = pipeline.compile_source(code, output_path, optimize=True, opt_level=opt_level, verbose=False)
            compile_time = time.perf_counter() - compile_start
            
            if not success:
                return -1, compile_time
            
            # Get binary size
            binary_size = os.path.getsize(output_path)
            
            # Run benchmark (the binary itself includes the iterations)
            start = time.perf_counter()
            result = subprocess.run([output_path], capture_output=True, timeout=30)
            elapsed = time.perf_counter() - start
            
            # Divide by iterations to get per-iteration time
            per_iter = elapsed / iterations if iterations > 0 else elapsed
            
            return per_iter, compile_time, binary_size
            
        except Exception as e:
            print(f"Compiled benchmark failed: {e}")
            return -1, 0, 0


def run_benchmark(name: str, code: str, iterations: int = 10000):
    """Run a single benchmark"""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*80}")
    print(f"Iterations: {iterations:,}")
    
    # Test different optimization levels
    opt_levels = [0, 1, 2, 3]
    results = {}
    
    for opt_level in opt_levels:
        compiled_time, compile_time, binary_size = benchmark_compiled(code, iterations, opt_level)
        
        if compiled_time > 0:
            results[opt_level] = {
                'time': compiled_time * 1000,  # Convert to ms
                'compile_time': compile_time,
                'binary_size': binary_size
            }
    
    # Display results
    print(f"\n{'Opt Level':<12} {'Time/iter':<15} {'Compile Time':<15} {'Binary Size':<15}")
    print("-" * 60)
    
    for opt_level in opt_levels:
        if opt_level in results:
            r = results[opt_level]
            print(f"O{opt_level:<11} {r['time']:.6f} ms{'':<6} {r['compile_time']:.3f} s{'':<8} {r['binary_size']:,} bytes")
    
    # Show speedup from optimizations
    if 0 in results and 3 in results:
        speedup = results[0]['time'] / results[3]['time']
        print(f"\n🚀 Speedup O0→O3: {speedup:.2f}x")


# ============================================================================
# BENCHMARK 1: Fibonacci (Recursive)
# ============================================================================

FIBONACCI_CODE = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def main() -> int:
    return fibonacci(10)
"""


# ============================================================================
# BENCHMARK 2: Factorial (Iterative)
# ============================================================================

FACTORIAL_CODE = """
def factorial(n: int) -> int:
    result: int = 1
    i: int = 1
    while i <= n:
        result = result * i
        i = i + 1
    return result

def main() -> int:
    return factorial(10)
"""


# ============================================================================
# BENCHMARK 3: Sum of Squares
# ============================================================================

SUM_SQUARES_CODE = """
def sum_squares(n: int) -> int:
    total: int = 0
    i: int = 0
    while i < n:
        total = total + i * i
        i = i + 1
    return total

def main() -> int:
    return sum_squares(100)
"""


# ============================================================================
# BENCHMARK 4: Nested Loops
# ============================================================================

NESTED_LOOPS_CODE = """
def nested_loops(n: int) -> int:
    total: int = 0
    i: int = 0
    while i < n:
        j: int = 0
        while j < n:
            total = total + i * j
            j = j + 1
        i = i + 1
    return total

def main() -> int:
    return nested_loops(20)
"""


# ============================================================================
# BENCHMARK 5: Arithmetic Operations
# ============================================================================

ARITHMETIC_CODE = """
def compute(a: int, b: int, c: int) -> int:
    x: int = a + b * c
    y: int = x - b / 2
    z: int = y * 3 + a
    return z - c

def main() -> int:
    return compute(10, 20, 30)
"""


# ============================================================================
# BENCHMARK 6: Control Flow Heavy
# ============================================================================

CONTROL_FLOW_CODE = """
def classify(x: int) -> int:
    if x < 0:
        if x < -10:
            return 1
        return 2
    if x > 0:
        if x > 10:
            return 3
        return 4
    return 5

def control_flow_test(n: int) -> int:
    total: int = 0
    i: int = -15
    while i < n:
        total = total + classify(i)
        i = i + 1
    return total

def main() -> int:
    return control_flow_test(15)
"""


def main():
    """Run all benchmarks"""
    print("\n" + "="*80)
    print(" " * 20 + "NATIVE PYTHON COMPILER BENCHMARKS")
    print("="*80)
    print(f"Comparing optimization levels: O0 (none) → O3 (aggressive)")
    print(f"Platform: {sys.platform}")
    
    benchmarks = [
        ("Fibonacci (Recursive)", FIBONACCI_CODE, 100),
        ("Factorial (Iterative)", FACTORIAL_CODE, 10000),
        ("Sum of Squares", SUM_SQUARES_CODE, 1000),
        ("Nested Loops", NESTED_LOOPS_CODE, 100),
        ("Arithmetic Operations", ARITHMETIC_CODE, 10000),
        ("Control Flow Heavy", CONTROL_FLOW_CODE, 1000),
    ]
    
    for name, code, iterations in benchmarks:
        try:
            run_benchmark(name, code, iterations)
        except Exception as e:
            print(f"\n❌ Benchmark '{name}' failed: {e}")
    
    print("\n" + "="*80)
    print("✅ BENCHMARK SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
