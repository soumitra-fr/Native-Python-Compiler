"""
Phase 2.1: Runtime Tracer - Collect Execution Data for ML Training

This module instruments Python code execution to collect:
- Function call frequencies
- Argument types at runtime
- Return value types
- Execution paths
- Hot code regions

Data is used to train AI models in Phase 2.2 and 2.3.

Phase: 2.1 (Runtime Tracer)
"""

import sys
import ast
import inspect
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path


@dataclass
class FunctionCallEvent:
    """Record of a single function call"""
    function_name: str
    arg_types: List[str]
    return_type: str
    execution_time_ms: float
    call_count: int = 1
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionProfile:
    """Complete execution profile for a code module"""
    module_name: str
    total_runtime_ms: float
    function_calls: Dict[str, List[FunctionCallEvent]]
    hot_functions: List[str]  # Functions called most frequently
    type_patterns: Dict[str, Dict[str, int]]  # function -> {type_signature -> count}
    
    def to_dict(self) -> dict:
        return {
            'module_name': self.module_name,
            'total_runtime_ms': self.total_runtime_ms,
            'function_calls': {
                k: [event.to_dict() for event in v]
                for k, v in self.function_calls.items()
            },
            'hot_functions': self.hot_functions,
            'type_patterns': self.type_patterns
        }
    
    def save(self, filepath: str):
        """Save profile to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class RuntimeTracer:
    """
    Traces Python code execution to collect training data
    
    Usage:
        tracer = RuntimeTracer()
        tracer.start()
        # ... run your code ...
        profile = tracer.stop()
        profile.save('profile.json')
    """
    
    def __init__(self):
        self.function_calls: Dict[str, List[FunctionCallEvent]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.type_signatures: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.tracing_enabled = False
        self.traced_module: Optional[str] = None
    
    def _get_type_name(self, obj: Any) -> str:
        """Get type name of an object"""
        if obj is None:
            return 'None'
        return type(obj).__name__
    
    def _make_signature(self, args: tuple, kwargs: dict, return_value: Any) -> str:
        """Create type signature string"""
        arg_types = [self._get_type_name(arg) for arg in args]
        kwarg_types = [f"{k}:{self._get_type_name(v)}" for k, v in kwargs.items()]
        all_args = ', '.join(arg_types + kwarg_types)
        ret_type = self._get_type_name(return_value)
        return f"({all_args}) -> {ret_type}"
    
    def trace_function(self, func):
        """Decorator to trace function calls"""
        def wrapper(*args, **kwargs):
            if not self.tracing_enabled:
                return func(*args, **kwargs)
            
            func_name = func.__name__
            
            # Measure execution time
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            
            # Record call
            arg_types = [self._get_type_name(arg) for arg in args]
            return_type = self._get_type_name(result)
            
            event = FunctionCallEvent(
                function_name=func_name,
                arg_types=arg_types,
                return_type=return_type,
                execution_time_ms=elapsed
            )
            
            self.function_calls[func_name].append(event)
            self.call_counts[func_name] += 1
            
            # Record type signature
            signature = self._make_signature(args, kwargs, result)
            self.type_signatures[func_name][signature] += 1
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    def start(self, module_name: str = "__main__"):
        """Start tracing execution"""
        self.tracing_enabled = True
        self.traced_module = module_name
        self.start_time = time.perf_counter()
        self.function_calls.clear()
        self.call_counts.clear()
        self.type_signatures.clear()
    
    def stop(self) -> ExecutionProfile:
        """Stop tracing and return profile"""
        self.tracing_enabled = False
        self.end_time = time.perf_counter()
        
        total_runtime = (self.end_time - self.start_time) * 1000  # ms
        
        # Find hot functions (top 10 by call count)
        hot_functions = sorted(
            self.call_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        hot_func_names = [name for name, _ in hot_functions]
        
        return ExecutionProfile(
            module_name=self.traced_module or "__main__",
            total_runtime_ms=total_runtime,
            function_calls=dict(self.function_calls),
            hot_functions=hot_func_names,
            type_patterns=dict(self.type_signatures)
        )


class AutoTracer:
    """
    Automatically traces all function calls in a module
    
    Usage:
        with AutoTracer('my_module') as tracer:
            # Run code
            result = my_function()
        
        profile = tracer.get_profile()
        profile.save('profile.json')
    """
    
    def __init__(self, module_name: str = "__main__"):
        self.tracer = RuntimeTracer()
        self.module_name = module_name
        self.original_trace = None
    
    def trace_calls(self, frame, event, arg):
        """Trace function calls"""
        if event == 'call':
            func_name = frame.f_code.co_name
            
            # Get arguments
            args = []
            local_vars = frame.f_locals
            
            # Record call
            if self.tracer.tracing_enabled:
                self.tracer.call_counts[func_name] += 1
        
        return self.trace_calls
    
    def __enter__(self):
        self.tracer.start(self.module_name)
        self.original_trace = sys.gettrace()
        sys.settrace(self.trace_calls)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.settrace(self.original_trace)
        self.profile = self.tracer.stop()
        return False
    
    def get_profile(self) -> ExecutionProfile:
        """Get execution profile"""
        return self.profile


# Example usage
if __name__ == "__main__":
    # Create a tracer
    tracer = RuntimeTracer()
    
    # Decorate functions to trace
    @tracer.trace_function
    def fibonacci(n: int) -> int:
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    @tracer.trace_function
    def factorial(n: int) -> int:
        if n <= 1:
            return 1
        return n * factorial(n - 1)
    
    # Start tracing
    tracer.start("test_module")
    
    # Run code
    fib_result = fibonacci(10)
    fact_result = factorial(10)
    
    # Stop and get profile
    profile = tracer.stop()
    
    # Display results
    print("\n" + "="*80)
    print("RUNTIME TRACE RESULTS")
    print("="*80)
    print(f"Module: {profile.module_name}")
    print(f"Total Runtime: {profile.total_runtime_ms:.2f}ms")
    print(f"\nHot Functions:")
    for func in profile.hot_functions:
        count = len(profile.function_calls.get(func, []))
        print(f"  - {func}: {count} calls")
    
    print(f"\nType Patterns:")
    for func, signatures in profile.type_patterns.items():
        print(f"  {func}:")
        for sig, count in signatures.items():
            print(f"    {sig} × {count}")
    
    # Save to file
    output_dir = Path(__file__).parent.parent / "training_data"
    output_dir.mkdir(exist_ok=True)
    profile_path = output_dir / "example_profile.json"
    profile.save(str(profile_path))
    print(f"\n✅ Profile saved to: {profile_path}")
    print("="*80)
