"""
Hot Function Detector - Phase 0, Week 1
Identifies frequently executed functions that are candidates for compilation
"""

import sys
import time
import inspect
import functools
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading


@dataclass
class FunctionProfile:
    """Profile data for a single function"""
    name: str
    module: str
    call_count: int = 0
    total_time: float = 0.0
    self_time: float = 0.0
    avg_time: float = 0.0
    is_hot: bool = False
    
    def update_stats(self):
        """Recalculate derived statistics"""
        if self.call_count > 0:
            self.avg_time = self.total_time / self.call_count


class HotFunctionDetector:
    """
    Lightweight profiler to detect hot functions
    Uses decorator-based instrumentation
    """
    
    def __init__(self, hot_threshold: int = 100, time_threshold: float = 0.1):
        """
        Args:
            hot_threshold: Minimum call count to be considered "hot"
            time_threshold: Minimum total time (seconds) to be considered "hot"
        """
        self.hot_threshold = hot_threshold
        self.time_threshold = time_threshold
        self.profiles: Dict[str, FunctionProfile] = {}
        self.call_stack = []
        self._lock = threading.Lock()
        
    def profile(self, func: Callable) -> Callable:
        """
        Decorator to profile a function
        
        Usage:
            detector = HotFunctionDetector()
            
            @detector.profile
            def my_function(x):
                return x * 2
        """
        func_name = func.__name__
        func_module = func.__module__
        func_id = f"{func_module}.{func_name}"
        
        # Initialize profile if not exists
        if func_id not in self.profiles:
            self.profiles[func_id] = FunctionProfile(
                name=func_name,
                module=func_module
            )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profile = self.profiles[func_id]
            
            # Track call stack for nested timing
            start_time = time.perf_counter()
            self.call_stack.append(func_id)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                
                self.call_stack.pop()
                
                with self._lock:
                    profile.call_count += 1
                    profile.total_time += elapsed
                    profile.self_time += elapsed  # Simplified: not accounting for nested calls
                    profile.update_stats()
        
        return wrapper
    
    def analyze(self) -> List[FunctionProfile]:
        """
        Analyze collected profiles and identify hot functions
        
        Returns:
            List of FunctionProfile objects sorted by hotness
        """
        for profile in self.profiles.values():
            profile.update_stats()
            
            # Mark as hot if meets thresholds
            if (profile.call_count >= self.hot_threshold and 
                profile.total_time >= self.time_threshold):
                profile.is_hot = True
        
        # Sort by total time (most expensive first)
        hot_functions = sorted(
            self.profiles.values(),
            key=lambda p: p.total_time,
            reverse=True
        )
        
        return hot_functions
    
    def get_hot_functions(self) -> List[FunctionProfile]:
        """Get only functions marked as hot"""
        return [p for p in self.analyze() if p.is_hot]
    
    def print_report(self, top_n: int = 10):
        """Print a formatted report of hot functions"""
        functions = self.analyze()
        
        print("=" * 80)
        print("HOT FUNCTION DETECTION REPORT")
        print("=" * 80)
        print(f"{'Function':<40} {'Calls':>10} {'Total(s)':>12} {'Avg(ms)':>12} {'Hot':>6}")
        print("-" * 80)
        
        for i, profile in enumerate(functions[:top_n]):
            hot_marker = "🔥" if profile.is_hot else "  "
            print(
                f"{profile.module}.{profile.name:<40} "
                f"{profile.call_count:>10} "
                f"{profile.total_time:>12.6f} "
                f"{profile.avg_time*1000:>12.6f} "
                f"{hot_marker:>6}"
            )
        
        print("=" * 80)
        hot_count = sum(1 for p in self.profiles.values() if p.is_hot)
        print(f"Total functions tracked: {len(self.profiles)}")
        print(f"Hot functions detected: {hot_count}")
        print("=" * 80)
    
    def reset(self):
        """Reset all profiling data"""
        with self._lock:
            self.profiles.clear()
            self.call_stack.clear()


# Global detector instance for easy use
_default_detector = HotFunctionDetector()


def profile(func: Callable) -> Callable:
    """Convenience decorator using the default detector"""
    return _default_detector.profile(func)


def get_hot_functions() -> List[FunctionProfile]:
    """Get hot functions from the default detector"""
    return _default_detector.get_hot_functions()


def print_report(top_n: int = 10):
    """Print report from the default detector"""
    _default_detector.print_report(top_n)


def reset():
    """Reset the default detector"""
    _default_detector.reset()


if __name__ == "__main__":
    # Example usage
    detector = HotFunctionDetector(hot_threshold=10, time_threshold=0.01)
    
    @detector.profile
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    @detector.profile
    def factorial(n):
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n+1):
            result *= i
        return result
    
    @detector.profile
    def matrix_multiply(size):
        # Simulate matrix multiplication
        total = 0
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    total += i * j * k
        return total
    
    # Run test workload
    print("Running test workload...")
    
    for _ in range(20):
        fibonacci(10)
    
    for _ in range(100):
        factorial(20)
    
    for _ in range(5):
        matrix_multiply(50)
    
    # Print results
    detector.print_report()
    
    # Get hot functions for compilation
    hot_funcs = detector.get_hot_functions()
    print(f"\nFunctions to compile: {[f.name for f in hot_funcs]}")
