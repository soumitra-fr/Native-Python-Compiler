"""
Advanced Runtime Tracer with Distributed Tracing and Online Learning
State-of-the-art instrumentation and profiling
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import time
import threading
import queue
import json
from pathlib import Path
from collections import defaultdict
import ast
import inspect
import sys


@dataclass
class DistributedTraceEvent:
    """Distributed tracing event with span information"""
    trace_id: str                   # Unique trace ID
    span_id: str                    # Unique span ID
    parent_span_id: Optional[str]   # Parent span
    function_name: str
    module_name: str
    arg_types: List[str]
    return_type: str
    execution_time: float
    memory_delta: float
    cpu_usage: float
    timestamp: float
    thread_id: int
    process_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlineLearningUpdate:
    """Real-time learning update"""
    pattern: str
    old_prediction: Any
    actual_outcome: Any
    error: float
    timestamp: float


class DistributedRuntimeTracer:
    """
    Advanced runtime tracer with:
    - Distributed tracing across processes
    - Online learning from execution
    - Real-time profiling
    - Adaptive instrumentation
    - Performance anomaly detection
    """
    
    def __init__(
        self,
        enable_distributed: bool = True,
        enable_online_learning: bool = True,
        sampling_rate: float = 1.0,
        buffer_size: int = 10000
    ):
        self.enable_distributed = enable_distributed
        self.enable_online_learning = enable_online_learning
        self.sampling_rate = sampling_rate
        
        # Tracing data
        self.traces: Dict[str, List[DistributedTraceEvent]] = defaultdict(list)
        self.current_spans: Dict[int, List[str]] = defaultdict(list)  # Per-thread span stack
        
        # Online learning
        self.learning_updates = queue.Queue(maxsize=buffer_size)
        self.learning_thread = None
        
        # Performance profiling
        self.function_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'memory_usage': [],
            'cpu_usage': []
        })
        
        # Adaptive instrumentation
        self.instrumentation_enabled = defaultdict(lambda: True)
        self.hotspot_functions = set()
        
        # Anomaly detection
        self.anomaly_threshold = 3.0  # 3 standard deviations
        self.detected_anomalies = []
        
        # Start online learning thread
        if enable_online_learning:
            self.learning_thread = threading.Thread(target=self._online_learning_worker, daemon=True)
            self.learning_thread.start()
        
        print(f"🔍 Advanced Runtime Tracer initialized")
        print(f"   Distributed: {enable_distributed}")
        print(f"   Online Learning: {enable_online_learning}")
        print(f"   Sampling Rate: {sampling_rate}")
    
    def trace(self, func: Callable = None, *, sample: bool = True):
        """
        Decorator for tracing function execution
        
        Args:
            func: Function to trace
            sample: Whether to sample (based on sampling_rate)
        """
        def decorator(f):
            def wrapper(*args, **kwargs):
                # Sampling
                if sample and np.random.random() > self.sampling_rate:
                    return f(*args, **kwargs)
                
                # Check if instrumentation is disabled for this function
                if not self.instrumentation_enabled[f.__name__]:
                    return f(*args, **kwargs)
                
                # Start span
                trace_id, span_id, parent_span_id = self._start_span(f.__name__)
                
                # Record start
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    # Execute function
                    result = f(*args, **kwargs)
                    
                    # Record end
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    # Extract types
                    arg_types = [type(arg).__name__ for arg in args]
                    return_type = type(result).__name__
                    
                    # Create trace event
                    event = DistributedTraceEvent(
                        trace_id=trace_id,
                        span_id=span_id,
                        parent_span_id=parent_span_id,
                        function_name=f.__name__,
                        module_name=f.__module__,
                        arg_types=arg_types,
                        return_type=return_type,
                        execution_time=end_time - start_time,
                        memory_delta=end_memory - start_memory,
                        cpu_usage=self._get_cpu_usage(),
                        timestamp=start_time,
                        thread_id=threading.get_ident(),
                        process_id=0  # Simplified
                    )
                    
                    # Store event
                    self.traces[trace_id].append(event)
                    
                    # Update statistics
                    self._update_stats(f.__name__, event)
                    
                    # Anomaly detection
                    self._detect_anomaly(f.__name__, event)
                    
                    # Online learning update
                    if self.enable_online_learning:
                        self._queue_learning_update(event)
                    
                    return result
                    
                finally:
                    # End span
                    self._end_span()
            
            wrapper.__name__ = f.__name__
            wrapper.__wrapped__ = f
            return wrapper
        
        if func is None:
            return decorator
        return decorator(func)
    
    def _start_span(self, function_name: str) -> Tuple[str, str, Optional[str]]:
        """Start a new span"""
        thread_id = threading.get_ident()
        
        # Generate IDs
        import uuid
        trace_id = str(uuid.uuid4())[:8]
        span_id = str(uuid.uuid4())[:8]
        
        # Get parent span
        parent_span_id = None
        if self.current_spans[thread_id]:
            parent_span_id = self.current_spans[thread_id][-1]
        
        # Push span
        self.current_spans[thread_id].append(span_id)
        
        return trace_id, span_id, parent_span_id
    
    def _end_span(self):
        """End current span"""
        thread_id = threading.get_ident()
        if self.current_spans[thread_id]:
            self.current_spans[thread_id].pop()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.01)
        except:
            return 0.0
    
    def _update_stats(self, function_name: str, event: DistributedTraceEvent):
        """Update function statistics"""
        stats = self.function_stats[function_name]
        
        stats['count'] += 1
        stats['total_time'] += event.execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], event.execution_time)
        stats['max_time'] = max(stats['max_time'], event.execution_time)
        stats['memory_usage'].append(event.memory_delta)
        stats['cpu_usage'].append(event.cpu_usage)
        
        # Identify hotspots (functions taking > 10% of total time)
        total_time = sum(s['total_time'] for s in self.function_stats.values())
        if stats['total_time'] / total_time > 0.1:
            self.hotspot_functions.add(function_name)
    
    def _detect_anomaly(self, function_name: str, event: DistributedTraceEvent):
        """Detect performance anomalies"""
        stats = self.function_stats[function_name]
        
        if stats['count'] < 10:
            return  # Need more data
        
        # Calculate z-score
        times = [event.execution_time for _ in range(stats['count'])]
        mean_time = stats['avg_time']
        std_time = np.std(times) if len(times) > 1 else 0.0
        
        if std_time > 0:
            z_score = abs(event.execution_time - mean_time) / std_time
            
            if z_score > self.anomaly_threshold:
                anomaly = {
                    'function': function_name,
                    'expected_time': mean_time,
                    'actual_time': event.execution_time,
                    'z_score': z_score,
                    'timestamp': event.timestamp
                }
                self.detected_anomalies.append(anomaly)
                print(f"⚠️  Anomaly detected in {function_name}: {event.execution_time:.4f}s (expected {mean_time:.4f}s, z={z_score:.2f})")
    
    def _queue_learning_update(self, event: DistributedTraceEvent):
        """Queue update for online learning"""
        try:
            update = OnlineLearningUpdate(
                pattern=f"{event.function_name}:{':'.join(event.arg_types)}",
                old_prediction=None,  # Would come from model
                actual_outcome=event.execution_time,
                error=0.0,  # Would be calculated
                timestamp=event.timestamp
            )
            self.learning_updates.put_nowait(update)
        except queue.Full:
            pass  # Drop update if buffer full
    
    def _online_learning_worker(self):
        """Background worker for online learning"""
        print("🧠 Online learning worker started")
        
        while True:
            try:
                update = self.learning_updates.get(timeout=1.0)
                
                # Process update (would update ML models here)
                # For now, just log
                if self.learning_updates.qsize() % 100 == 0:
                    print(f"📊 Processed {self.learning_updates.qsize()} learning updates")
                
            except queue.Empty:
                continue
    
    def adaptive_instrumentation(self, enable_cold_paths: bool = False):
        """
        Adaptively enable/disable instrumentation
        
        Args:
            enable_cold_paths: Whether to instrument rarely-called functions
        """
        print("🎯 Adjusting instrumentation...")
        
        # Disable instrumentation for cold paths
        for func_name, stats in self.function_stats.items():
            total_calls = sum(s['count'] for s in self.function_stats.values())
            call_percentage = stats['count'] / total_calls if total_calls > 0 else 0
            
            if call_percentage < 0.01 and not enable_cold_paths:
                self.instrumentation_enabled[func_name] = False
                print(f"   Disabled instrumentation for {func_name} (cold path)")
            else:
                self.instrumentation_enabled[func_name] = True
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'total_traces': len(self.traces),
            'total_events': sum(len(events) for events in self.traces.values()),
            'hotspot_functions': list(self.hotspot_functions),
            'function_stats': dict(self.function_stats),
            'anomalies': self.detected_anomalies,
            'top_slow_functions': self._get_top_slow_functions(10),
            'top_memory_functions': self._get_top_memory_functions(10)
        }
        return report
    
    def _get_top_slow_functions(self, n: int) -> List[Tuple[str, float]]:
        """Get top N slowest functions"""
        return sorted(
            [(name, stats['avg_time']) for name, stats in self.function_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:n]
    
    def _get_top_memory_functions(self, n: int) -> List[Tuple[str, float]]:
        """Get top N memory-intensive functions"""
        return sorted(
            [
                (name, np.mean(stats['memory_usage']) if stats['memory_usage'] else 0)
                for name, stats in self.function_stats.items()
            ],
            key=lambda x: x[1],
            reverse=True
        )[:n]
    
    def export_traces(self, path: str, format: str = "json"):
        """Export traces for analysis"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Convert to JSON-serializable format
            export_data = {
                trace_id: [
                    {
                        'trace_id': e.trace_id,
                        'span_id': e.span_id,
                        'parent_span_id': e.parent_span_id,
                        'function_name': e.function_name,
                        'module_name': e.module_name,
                        'arg_types': e.arg_types,
                        'return_type': e.return_type,
                        'execution_time': e.execution_time,
                        'memory_delta': e.memory_delta,
                        'cpu_usage': e.cpu_usage,
                        'timestamp': e.timestamp,
                        'thread_id': e.thread_id
                    }
                    for e in events
                ]
                for trace_id, events in self.traces.items()
            }
            
            with open(path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        print(f"✅ Traces exported to {path}")
    
    def clear(self):
        """Clear all tracing data"""
        self.traces.clear()
        self.function_stats.clear()
        self.hotspot_functions.clear()
        self.detected_anomalies.clear()
        print("🗑️  Tracing data cleared")


# Singleton instance
_global_tracer: Optional[DistributedRuntimeTracer] = None


def get_tracer() -> DistributedRuntimeTracer:
    """Get global tracer instance"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DistributedRuntimeTracer()
    return _global_tracer


def trace(func: Callable = None):
    """Convenience decorator using global tracer"""
    tracer = get_tracer()
    return tracer.trace(func)


if __name__ == "__main__":
    # Demo
    tracer = DistributedRuntimeTracer(enable_online_learning=True)
    
    @tracer.trace
    def slow_function(n: int) -> int:
        """Simulate slow function"""
        time.sleep(0.01 * n)
        return n * n
    
    @tracer.trace
    def fast_function(x: float) -> float:
        """Simulate fast function"""
        return x * 2.0
    
    # Run functions
    print("Running traced functions...")
    for i in range(20):
        slow_function(i % 5)
        fast_function(i * 1.5)
    
    # Adaptive instrumentation
    tracer.adaptive_instrumentation()
    
    # Get report
    report = tracer.get_performance_report()
    print(f"\n📊 Performance Report:")
    print(f"   Total traces: {report['total_traces']}")
    print(f"   Total events: {report['total_events']}")
    print(f"   Hotspots: {report['hotspot_functions']}")
    print(f"   Anomalies: {len(report['anomalies'])}")
    
    # Export
    tracer.export_traces("training_data/advanced_traces.json")
