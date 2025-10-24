"""
Numba Integration Layer - Phase 0, Week 2
Automatic JIT compilation wrapper with fallback mechanisms
"""

import inspect
import functools
import warnings
from typing import Callable, Any, Optional, Dict, Set
from dataclasses import dataclass
import time

try:
    from numba import jit, types
    from numba.core.errors import TypingError, NumbaError
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Install with: pip install numba")


@dataclass
class CompilationResult:
    """Result of a compilation attempt"""
    success: bool
    compiled_func: Optional[Callable]
    error_message: Optional[str] = None
    compilation_time: float = 0.0
    speedup_estimate: float = 1.0


class NumbaCompiler:
    """
    Intelligent wrapper around Numba JIT compilation
    Handles type inference, fallback, and performance tracking
    """
    
    def __init__(self):
        self.compiled_functions: Dict[str, CompilationResult] = {}
        self.compilation_failures: Set[str] = set()
        self.performance_data: Dict[str, Dict[str, float]] = {}
        
    def can_compile(self, func: Callable) -> bool:
        """
        Quick heuristic check if function is compilable
        
        Returns:
            True if function looks compilable, False otherwise
        """
        if not NUMBA_AVAILABLE:
            return False
        
        # Get function source
        try:
            source = inspect.getsource(func)
        except:
            return False
        
        # Simple heuristics for compilability
        # These will be replaced by ML model in Phase 0, Week 3
        
        # Check for unsupported features
        unsupported_keywords = [
            'import',      # Dynamic imports
            'open(',       # File I/O
            'print(',      # I/O operations (can be supported with special mode)
            'eval',        # Dynamic code
            'exec',        # Dynamic code
            'globals(',    # Introspection
            'locals(',     # Introspection
            '__dict__',    # Dynamic attributes
            'class ',      # Class definitions (not in simple functions)
        ]
        
        for keyword in unsupported_keywords:
            if keyword in source:
                return False
        
        return True
    
    def compile(
        self,
        func: Callable,
        nopython: bool = True,
        cache: bool = True,
        fastmath: bool = True
    ) -> CompilationResult:
        """
        Attempt to compile a function with Numba
        
        Args:
            func: Function to compile
            nopython: Use nopython mode (faster, more restrictive)
            cache: Cache compiled code
            fastmath: Enable fast math optimizations
            
        Returns:
            CompilationResult with compilation status and compiled function
        """
        func_name = func.__name__
        
        # Check if already tried and failed
        if func_name in self.compilation_failures:
            return CompilationResult(
                success=False,
                compiled_func=None,
                error_message="Previously failed compilation"
            )
        
        # Check if already compiled
        if func_name in self.compiled_functions:
            return self.compiled_functions[func_name]
        
        if not NUMBA_AVAILABLE:
            return CompilationResult(
                success=False,
                compiled_func=None,
                error_message="Numba not available"
            )
        
        # Quick compilability check
        if not self.can_compile(func):
            self.compilation_failures.add(func_name)
            return CompilationResult(
                success=False,
                compiled_func=None,
                error_message="Function contains unsupported features"
            )
        
        # Attempt compilation
        start_time = time.perf_counter()
        
        try:
            # Try nopython mode first (fastest)
            if nopython:
                compiled_func = jit(
                    nopython=True,
                    cache=cache,
                    fastmath=fastmath
                )(func)
            else:
                # Fall back to object mode
                compiled_func = jit(
                    nopython=False,
                    forceobj=True,
                    cache=cache
                )(func)
            
            compilation_time = time.perf_counter() - start_time
            
            result = CompilationResult(
                success=True,
                compiled_func=compiled_func,
                compilation_time=compilation_time
            )
            
            self.compiled_functions[func_name] = result
            return result
            
        except (TypingError, NumbaError) as e:
            # Compilation failed
            self.compilation_failures.add(func_name)
            
            # If nopython failed, try object mode
            if nopython:
                return self.compile(func, nopython=False, cache=cache, fastmath=fastmath)
            
            return CompilationResult(
                success=False,
                compiled_func=None,
                error_message=str(e)
            )
    
    def auto_compile(self, func: Callable) -> Callable:
        """
        Decorator that automatically compiles if possible, otherwise returns original
        
        Usage:
            compiler = NumbaCompiler()
            
            @compiler.auto_compile
            def my_function(x):
                return x * 2
        """
        result = self.compile(func)
        
        if result.success:
            return result.compiled_func
        else:
            # Return original function as fallback
            return func
    
    def benchmark(
        self,
        original_func: Callable,
        compiled_func: Callable,
        *args,
        **kwargs
    ) -> float:
        """
        Benchmark speedup of compiled vs original function
        
        Returns:
            Speedup factor (compiled_time / original_time)
        """
        # Warm-up (trigger JIT compilation)
        try:
            compiled_func(*args, **kwargs)
        except:
            pass
        
        # Benchmark original
        start = time.perf_counter()
        for _ in range(100):
            original_func(*args, **kwargs)
        original_time = time.perf_counter() - start
        
        # Benchmark compiled
        start = time.perf_counter()
        for _ in range(100):
            compiled_func(*args, **kwargs)
        compiled_time = time.perf_counter() - start
        
        if compiled_time > 0:
            speedup = original_time / compiled_time
        else:
            speedup = 1.0
        
        return speedup
    
    def print_stats(self):
        """Print compilation statistics"""
        print("=" * 80)
        print("NUMBA COMPILATION STATISTICS")
        print("=" * 80)
        
        successful = sum(1 for r in self.compiled_functions.values() if r.success)
        failed = len(self.compilation_failures)
        
        print(f"Successful compilations: {successful}")
        print(f"Failed compilations: {failed}")
        
        if successful > 0:
            print("\nSuccessfully compiled functions:")
            for func_name, result in self.compiled_functions.items():
                if result.success:
                    print(f"  - {func_name} (compilation time: {result.compilation_time:.4f}s)")
        
        if failed > 0:
            print("\nFailed compilations:")
            for func_name in self.compilation_failures:
                print(f"  - {func_name}")
        
        print("=" * 80)


# Global compiler instance
_default_compiler = NumbaCompiler()


def auto_compile(func: Callable) -> Callable:
    """Convenience decorator using default compiler"""
    return _default_compiler.auto_compile(func)


def compile_function(func: Callable) -> CompilationResult:
    """Compile a function using default compiler"""
    return _default_compiler.compile(func)


def print_stats():
    """Print stats from default compiler"""
    _default_compiler.print_stats()


if __name__ == "__main__":
    # Example usage and testing
    compiler = NumbaCompiler()
    
    # Test 1: Simple numeric function (should compile)
    @compiler.auto_compile
    def add(a, b):
        return a + b
    
    # Test 2: Loop-heavy function (should compile)
    @compiler.auto_compile
    def matrix_sum(n):
        total = 0
        for i in range(n):
            for j in range(n):
                total += i * j
        return total
    
    # Test 3: Function with unsupported features (should fallback)
    @compiler.auto_compile
    def use_print(x):
        print(f"Value: {x}")
        return x * 2
    
    # Test 4: Recursive function
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    fib_compiled = compiler.auto_compile(fibonacci)
    
    # Run tests
    print("Testing compiled functions...")
    print(f"add(5, 3) = {add(5, 3)}")
    print(f"matrix_sum(100) = {matrix_sum(100)}")
    print(f"use_print(10) = {use_print(10)}")
    print(f"fibonacci(10) = {fib_compiled(10)}")
    
    # Benchmark
    if NUMBA_AVAILABLE:
        print("\nBenchmarking matrix_sum...")
        
        def matrix_sum_py(n):
            total = 0
            for i in range(n):
                for j in range(n):
                    total += i * j
            return total
        
        speedup = compiler.benchmark(matrix_sum_py, matrix_sum, 100)
        print(f"Speedup: {speedup:.2f}x")
    
    # Print statistics
    compiler.print_stats()
