"""
Simple Benchmark - Test Optimization Levels

Tests a single fibonacci function with different optimization levels.
"""

import subprocess
import tempfile
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from compiler.backend.codegen import CompilerPipeline


FIBONACCI_CODE = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def main() -> int:
    return fibonacci(15)
"""


def test_optimization_level(opt_level: int):
    """Test a single optimization level"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "fib")
        
        # Compile
        pipeline = CompilerPipeline()
        compile_start = time.perf_counter()
        success = pipeline.compile_source(
            FIBONACCI_CODE,
            output_path,
            optimize=(opt_level > 0),
            opt_level=opt_level,
            verbose=False
        )
        compile_time = time.perf_counter() - compile_start
        
        if not success:
            print(f"O{opt_level}: Compilation failed")
            return
        
        # Get binary size
        binary_size = os.path.getsize(output_path)
        
        # Run and time it
        times = []
        for _ in range(5):  # Run 5 times and average
            start = time.perf_counter()
            result = subprocess.run([output_path], capture_output=True, timeout=10)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        print(f"O{opt_level}: {avg_time*1000:.3f}ms (compile: {compile_time:.3f}s, size: {binary_size:,} bytes)")
        return avg_time


def main():
    print("="*60)
    print("OPTIMIZATION LEVEL BENCHMARK - Fibonacci(15)")
    print("="*60)
    
    times = {}
    for opt_level in [0, 1, 2, 3]:
        times[opt_level] = test_optimization_level(opt_level)
    
    if 0 in times and 3 in times and times[0] and times[3]:
        speedup = times[0] / times[3]
        print(f"\n🚀 Speedup O0→O3: {speedup:.2f}x")
    
    print("="*60)


if __name__ == "__main__":
    main()
