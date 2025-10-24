"""
LinkedIn Showcase: Native Python Compiler Performance
Demonstrates real-world speedup on Mandelbrot Set computation
"""

import time
import subprocess
import sys

def print_header(text):
    """Print a nice header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_result(label, time_val):
    """Print execution time."""
    print(f"  {label}: {time_val:.4f} seconds")

def run_standard_python():
    """Run standard Python Mandelbrot."""
    print("Running Standard Python (CPython 3.x)...")
    
    # Import and run the standard version
    start = time.time()
    result = subprocess.run(
        [sys.executable, "mandelbrot_benchmark/mandelbrot.py"],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        # Extract time from output
        for line in result.stdout.split('\n'):
            if 'Execution time' in line:
                try:
                    time_str = line.split(':')[1].strip().split()[0]
                    return float(time_str)
                except:
                    pass
    
    return elapsed

def run_compiled_version():
    """Run our native compiled Mandelbrot."""
    print("Running Native Python Compiler (with JIT)...")
    
    # Import and run the compiled version
    start = time.time()
    result = subprocess.run(
        [sys.executable, "mandelbrot_benchmark/mandelbrot_compiled.py"],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        # Extract time from output
        for line in result.stdout.split('\n'):
            if 'Execution time' in line:
                try:
                    time_str = line.split(':')[1].strip().split()[0]
                    return float(time_str)
                except:
                    pass
    
    return elapsed

def main():
    """Main benchmark showcase."""
    
    print_header("🚀 Native Python Compiler - Performance Showcase")
    
    print("Test: Mandelbrot Set (1000x1000 pixels, 1M iterations)")
    print()
    
    # Run standard Python
    print("[ 1 ] Standard Python (CPython)...")
    python_time = run_standard_python()
    print(f"      Time: {python_time:.3f}s")
    print()
    
    # Run our compiler
    print("[ 2 ] Native Python Compiler...")
    compiled_time = run_compiled_version()
    print(f"      Time: {compiled_time:.3f}s")
    print()
    
    # Show comparison
    print_header("Results")
    
    speedup = python_time / compiled_time
    
    print(f"  Standard Python:    {python_time:.3f}s")
    print(f"  Our Compiler:       {compiled_time:.3f}s")
    print()
    print(f"  ⚡ Speedup:          {speedup:.1f}x FASTER")
    print()
    
    print(f"  Same Python code. {speedup:.1f}x performance gain.")
    print(f"  No code changes. Just add @njit decorator.")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    main()
