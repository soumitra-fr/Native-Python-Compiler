#!/usr/bin/env python3
"""
Mandelbrot Set Generator - Numeric Computation Benchmark

The Mandelbrot set is a fractal defined by iterating the formula:
    z(n+1) = z(n)^2 + c

For each complex number c, we iterate until:
- |z| > 2 (point escapes to infinity), or
- max iterations reached (point is in the set)

This is a pure computation benchmark:
- No I/O operations
- No external libraries
- Heavy arithmetic (complex number multiplication)
- Tight nested loops
- Type-intensive operations

Perfect for testing compiler optimization on numeric workloads.
"""

import time


def mandelbrot_escape_time(c_real: float, c_imag: float, max_iter: int) -> int:
    """
    Calculate escape time for a point in the complex plane.
    
    Args:
        c_real: Real part of complex number c
        c_imag: Imaginary part of complex number c
        max_iter: Maximum iterations before assuming point is in set
        
    Returns:
        Number of iterations until escape (or max_iter if in set)
    """
    z_real = 0.0
    z_imag = 0.0
    
    for iteration in range(max_iter):
        # Calculate z^2 + c
        # (a + bi)^2 = a^2 - b^2 + 2abi
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        
        # Check if escaped (|z| > 2, or z^2 > 4)
        if z_real_sq + z_imag_sq > 4.0:
            return iteration
        
        # z = z^2 + c
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real
    
    return max_iter


def generate_mandelbrot(width: int, height: int, max_iter: int) -> int:
    """
    Generate Mandelbrot set for a rectangular region.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        max_iter: Maximum iterations per point
        
    Returns:
        Total number of iterations performed (for verification)
    """
    # Define viewing window in complex plane
    x_min = -2.5
    x_max = 1.0
    y_min = -1.25
    y_max = 1.25
    
    # Calculate step sizes
    x_step = (x_max - x_min) / width
    y_step = (y_max - y_min) / height
    
    total_iterations = 0
    
    # For each pixel
    for py in range(height):
        c_imag = y_min + py * y_step
        
        for px in range(width):
            c_real = x_min + px * x_step
            
            # Calculate escape time
            escape_time = mandelbrot_escape_time(c_real, c_imag, max_iter)
            total_iterations += escape_time
    
    return total_iterations


def benchmark_mandelbrot(width: int = 800, height: int = 600, max_iter: int = 100, runs: int = 3):
    """
    Benchmark Mandelbrot generation with timing.
    
    Args:
        width: Image width
        height: Image height
        max_iter: Max iterations per pixel
        runs: Number of benchmark runs
    """
    print("="*80)
    print("MANDELBROT SET BENCHMARK")
    print("="*80)
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height} pixels")
    print(f"  Max iterations: {max_iter}")
    print(f"  Total points: {width * height:,}")
    print(f"  Benchmark runs: {runs}")
    print("="*80)
    print()
    
    times = []
    results = []
    
    # Warmup run
    print("Warmup run...")
    _ = generate_mandelbrot(width, height, max_iter)
    print("  ✓ Complete")
    print()
    
    # Benchmark runs
    print(f"Running {runs} benchmark iterations...")
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        
        start = time.perf_counter()
        total_iters = generate_mandelbrot(width, height, max_iter)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        results.append(total_iters)
        
        print(f"{elapsed_ms:.2f} ms")
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Min time:     {min_time:.2f} ms")
    print(f"Max time:     {max_time:.2f} ms")
    print(f"Verification: {results[0]:,} total iterations")
    print()
    
    # Calculate performance metrics
    total_ops = width * height * max_iter
    ops_per_sec = (total_ops / avg_time) * 1000
    
    print(f"Performance:")
    print(f"  Operations:     {total_ops:,}")
    print(f"  Ops/second:     {ops_per_sec:,.0f}")
    print(f"  Pixels/second:  {(width * height / avg_time) * 1000:,.0f}")
    print("="*80)
    
    return avg_time, results[0]


if __name__ == "__main__":
    # Run benchmark with default settings
    avg_time, total_iters = benchmark_mandelbrot(
        width=800,
        height=600,
        max_iter=100,
        runs=5
    )
    
    print()
    print(f"✅ Benchmark complete!")
    print(f"   Average execution time: {avg_time:.2f} ms")
    print(f"   Verification sum: {total_iters:,} iterations")
