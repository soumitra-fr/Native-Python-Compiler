#!/usr/bin/env python3
"""
Mandelbrot Benchmark - Compiled Version

This version uses our JIT compiler to achieve massive speedups.
"""

import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.runtime.jit_executor import JITExecutor
from compiler.frontend.semantic import Type, TypeKind


def benchmark_mandelbrot_compiled(width: int = 800, height: int = 600, max_iter: int = 100, runs: int = 5):
    """
    Benchmark compiled Mandelbrot generation
    """
    print("="*80)
    print("MANDELBROT SET BENCHMARK - COMPILED VERSION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Resolution: {width}x{height} pixels")
    print(f"  Max iterations: {max_iter}")
    print(f"  Total points: {width * height:,}")
    print(f"  Benchmark runs: {runs}")
    print(f"  Compiler: LLVM JIT with -O3 optimization")
    print("="*80)
    print()
    
    # LLVM IR for Mandelbrot calculation
    llvm_ir = """
    ; ModuleID = 'mandelbrot'
    target triple = "x86_64-apple-darwin"
    
    ; Calculate escape time for single point
    define i64 @mandelbrot_point(double %c_real, double %c_imag, i64 %max_iter) {
    entry:
      br label %loop
    
    loop:
      %iter = phi i64 [ 0, %entry ], [ %iter_next, %check_continue ]
      %z_real = phi double [ 0.0, %entry ], [ %new_z_real, %check_continue ]
      %z_imag = phi double [ 0.0, %entry ], [ %new_z_imag, %check_continue ]
      
      ; Calculate z^2
      %z_real_sq = fmul double %z_real, %z_real
      %z_imag_sq = fmul double %z_imag, %z_imag
      
      ; Check if escaped (|z|^2 > 4.0)
      %magnitude_sq = fadd double %z_real_sq, %z_imag_sq
      %escaped = fcmp ogt double %magnitude_sq, 4.0
      br i1 %escaped, label %return_iter, label %continue_iter
    
    continue_iter:
      ; Check if max iterations reached
      %iter_next = add i64 %iter, 1
      %done = icmp sge i64 %iter_next, %max_iter
      br i1 %done, label %return_max, label %calc_next
    
    calc_next:
      ; z = z^2 + c
      ; Real part: z_real^2 - z_imag^2 + c_real
      %real_part = fsub double %z_real_sq, %z_imag_sq
      %new_z_real = fadd double %real_part, %c_real
      
      ; Imaginary part: 2 * z_real * z_imag + c_imag
      %temp = fmul double %z_real, %z_imag
      %imag_part = fmul double %temp, 2.0
      %new_z_imag = fadd double %imag_part, %c_imag
      
      br label %check_continue
    
    check_continue:
      br label %loop
    
    return_iter:
      ret i64 %iter
    
    return_max:
      ret i64 %max_iter
    }
    
    ; Generate full Mandelbrot set
    define i64 @mandelbrot_generate(i64 %width, i64 %height, i64 %max_iter) {
    entry:
      br label %y_loop
    
    y_loop:
      %y = phi i64 [ 0, %entry ], [ %y_next, %y_continue ]
      %total = phi i64 [ 0, %entry ], [ %total_next, %y_continue ]
      
      ; Calculate c_imag = -1.25 + y * (2.5 / height)
      %y_double = sitofp i64 %y to double
      %height_double = sitofp i64 %height to double
      %y_step = fdiv double 2.5, %height_double
      %y_offset = fmul double %y_double, %y_step
      %c_imag = fadd double %y_offset, -1.25
      
      br label %x_loop
    
    x_loop:
      %x = phi i64 [ 0, %y_loop ], [ %x_next, %x_continue ]
      %row_total = phi i64 [ 0, %y_loop ], [ %row_total_next, %x_continue ]
      
      ; Calculate c_real = -2.5 + x * (3.5 / width)
      %x_double = sitofp i64 %x to double
      %width_double = sitofp i64 %width to double
      %x_step = fdiv double 3.5, %width_double
      %x_offset = fmul double %x_double, %x_step
      %c_real = fadd double %x_offset, -2.5
      
      ; Calculate escape time for this point
      %escape = call i64 @mandelbrot_point(double %c_real, double %c_imag, i64 %max_iter)
      %row_total_next = add i64 %row_total, %escape
      
      ; Continue x loop
      %x_next = add i64 %x, 1
      %x_done = icmp sge i64 %x_next, %width
      br i1 %x_done, label %y_continue, label %x_continue
    
    x_continue:
      br label %x_loop
    
    y_continue:
      %total_next = add i64 %total, %row_total_next
      %y_next = add i64 %y, 1
      %y_done = icmp sge i64 %y_next, %height
      br i1 %y_done, label %exit, label %y_loop
    
    exit:
      ret i64 %total_next
    }
    """
    
    print("Compiling with LLVM JIT...")
    executor = JITExecutor()
    
    if not executor.compile_ir(llvm_ir):
        print("❌ Compilation failed!")
        return None, None
    
    print("  ✓ Compilation successful")
    print()
    
    int_type = Type(TypeKind.INT)
    
    # Warmup run
    print("Warmup run...")
    _ = executor.execute_function(
        "mandelbrot_generate",
        [width, height, max_iter],
        [int_type, int_type, int_type],
        int_type
    )
    print("  ✓ Complete")
    print()
    
    # Benchmark runs
    times = []
    results = []
    
    print(f"Running {runs} benchmark iterations...")
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        
        start = time.perf_counter()
        total_iters = executor.execute_function(
            "mandelbrot_generate",
            [width, height, max_iter],
            [int_type, int_type, int_type],
            int_type
        )
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
    avg_time, total_iters = benchmark_mandelbrot_compiled(
        width=800,
        height=600,
        max_iter=100,
        runs=5
    )
    
    if avg_time is not None:
        print()
        print(f"✅ Benchmark complete!")
        print(f"   Average execution time: {avg_time:.2f} ms")
        print(f"   Verification sum: {total_iters:,} iterations")
