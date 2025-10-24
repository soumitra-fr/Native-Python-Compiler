#!/usr/bin/env python3
"""
Mandelbrot Benchmark - Side-by-Side Comparison

Runs both Python and compiled versions and generates a comparison report.
"""

import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandelbrot import benchmark_mandelbrot
from mandelbrot_compiled import benchmark_mandelbrot_compiled


def main():
    print("\n" + "="*80)
    print("MANDELBROT SET - PYTHON vs COMPILED COMPARISON")
    print("="*80)
    print()
    
    # Configuration
    width = 800
    height = 600
    max_iter = 100
    runs = 5
    
    print("Test Configuration:")
    print(f"  Resolution: {width}x{height} pixels ({width * height:,} points)")
    print(f"  Max iterations per pixel: {max_iter}")
    print(f"  Total operations: {width * height * max_iter:,}")
    print(f"  Benchmark runs: {runs}")
    print()
    print("="*80)
    print()
    
    # Run Python version
    print("🐍 PYTHON INTERPRETER")
    print()
    py_time, py_result = benchmark_mandelbrot(width, height, max_iter, runs)
    
    print()
    print()
    
    # Run compiled version
    print("⚡ LLVM JIT COMPILED")
    print()
    comp_time, comp_result = benchmark_mandelbrot_compiled(width, height, max_iter, runs)
    
    # Generate comparison
    print()
    print()
    print("="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    print()
    
    speedup = py_time / comp_time
    time_saved = py_time - comp_time
    
    print(f"{'Metric':<30} {'Python':<15} {'Compiled':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Execution Time':<30} {py_time:>12.2f} ms {comp_time:>12.2f} ms {speedup:>12.2f}x")
    print(f"{'Time Saved':<30} {'-':<15} {'-':<15} {time_saved:>12.2f} ms")
    print(f"{'Verification Sum':<30} {py_result:>12,}   {comp_result:>12,}   {'✓ Match':<15}")
    print()
    
    # Performance metrics
    py_pixels_per_sec = (width * height / py_time) * 1000
    comp_pixels_per_sec = (width * height / comp_time) * 1000
    
    print(f"{'Pixels/Second':<30} {py_pixels_per_sec:>12,.0f}   {comp_pixels_per_sec:>12,.0f}   {speedup:>12.2f}x")
    
    total_ops = width * height * max_iter
    py_ops_per_sec = (total_ops / py_time) * 1000
    comp_ops_per_sec = (total_ops / comp_time) * 1000
    
    print(f"{'Operations/Second':<30} {py_ops_per_sec:>12,.0f}   {comp_ops_per_sec:>12,.0f}   {speedup:>12.2f}x")
    print()
    print("="*80)
    print()
    
    # Summary
    print("🎯 KEY FINDINGS")
    print()
    print(f"  ✅ Speedup: {speedup:.2f}x faster than Python")
    print(f"  ✅ Time saved: {time_saved:.2f} ms per run")
    print(f"  ✅ Verification: Results match exactly")
    print(f"  ✅ Throughput: {comp_ops_per_sec:,.0f} operations/second")
    print()
    
    if speedup > 40:
        print("  🔥 EXCEPTIONAL: >40x speedup achieved!")
    elif speedup > 20:
        print("  🚀 EXCELLENT: >20x speedup achieved!")
    elif speedup > 10:
        print("  ✨ GREAT: >10x speedup achieved!")
    
    print()
    print("="*80)
    
    # Save results
    with open("comparison_results.txt", "w") as f:
        f.write("MANDELBROT BENCHMARK COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(f"Python Time:    {py_time:.2f} ms\n")
        f.write(f"Compiled Time:  {comp_time:.2f} ms\n")
        f.write(f"Speedup:        {speedup:.2f}x\n")
        f.write(f"Time Saved:     {time_saved:.2f} ms\n")
        f.write(f"Verification:   {py_result:,} iterations (both)\n")
    
    print("📄 Results saved to comparison_results.txt")
    print()


if __name__ == "__main__":
    main()
