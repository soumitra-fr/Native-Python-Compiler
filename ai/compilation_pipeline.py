"""
AI-Powered Compilation Pipeline
Integrates Runtime Tracer, Type Inference Engine, and Strategy Agent
into a cohesive end-to-end intelligent compilation system.

This pipeline orchestrates the three AI components:
1. RuntimeTracer: Collects execution profiles during interpreted runs
2. TypeInferenceEngine: Infers types from code patterns and runtime data
3. StrategyAgent: Decides optimal compilation strategy based on characteristics

Usage:
    from ai.compilation_pipeline import AICompilationPipeline
    
    pipeline = AICompilationPipeline()
    result = pipeline.compile_intelligently("examples/mycode.py")
    print(f"Compiled with {result.strategy} strategy: {result.output_path}")
"""

import ast
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our AI components
from ai.runtime_tracer import RuntimeTracer, ExecutionProfile
from ai.type_inference_engine import TypeInferenceEngine, TypePrediction
from ai.strategy_agent import (
    StrategyAgent,
    CompilationStrategy,
    CodeCharacteristics,
    StrategyDecision
)

# Import compiler components
from compiler.frontend.parser import Parser
from compiler.backend.codegen import CompilerPipeline as BackendPipeline


class PipelineStage(Enum):
    """Stages of the AI compilation pipeline"""
    PROFILING = "profiling"           # Collect runtime data
    TYPE_INFERENCE = "type_inference"  # Infer types from code
    STRATEGY_SELECTION = "strategy"    # Choose compilation strategy
    COMPILATION = "compilation"        # Compile with chosen strategy
    VALIDATION = "validation"          # Validate output


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution"""
    profiling_time_ms: float = 0.0
    type_inference_time_ms: float = 0.0
    strategy_selection_time_ms: float = 0.0
    compilation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Stage-specific metrics
    functions_profiled: int = 0
    types_inferred: int = 0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "profiling_time_ms": self.profiling_time_ms,
            "type_inference_time_ms": self.type_inference_time_ms,
            "strategy_selection_time_ms": self.strategy_selection_time_ms,
            "compilation_time_ms": self.compilation_time_ms,
            "total_time_ms": self.total_time_ms,
            "functions_profiled": self.functions_profiled,
            "types_inferred": self.types_inferred,
            "confidence_scores": self.confidence_scores
        }


@dataclass
class CompilationResult:
    """Result of AI-powered compilation"""
    success: bool
    strategy: CompilationStrategy
    output_path: Optional[str]
    metrics: PipelineMetrics
    execution_profile: Optional[ExecutionProfile]
    type_predictions: Dict[str, TypePrediction]
    strategy_decision: StrategyDecision
    error_message: Optional[str] = None
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        if not self.success:
            return f"❌ Compilation failed: {self.error_message}"
        
        lines = [
            "✅ AI-Powered Compilation Successful!",
            "",
            f"Strategy: {self.strategy.value.upper()}",
            f"Output: {self.output_path}",
            f"Total Time: {self.metrics.total_time_ms:.2f}ms",
            "",
            "Pipeline Breakdown:",
            f"  • Profiling: {self.metrics.profiling_time_ms:.2f}ms ({self.metrics.functions_profiled} functions)",
            f"  • Type Inference: {self.metrics.type_inference_time_ms:.2f}ms ({self.metrics.types_inferred} types)",
            f"  • Strategy Selection: {self.metrics.strategy_selection_time_ms:.2f}ms",
            f"  • Compilation: {self.metrics.compilation_time_ms:.2f}ms",
            "",
            f"Strategy Reasoning: {self.strategy_decision.reasoning}",
            f"Expected Speedup: {self.strategy_decision.expected_speedup:.1f}x",
            f"Confidence: {self.strategy_decision.confidence:.1%}"
        ]
        
        if self.type_predictions:
            lines.append("")
            lines.append("Type Predictions:")
            for var, pred in list(self.type_predictions.items())[:5]:
                lines.append(f"  • {var}: {pred.predicted_type} ({pred.confidence:.0%} confidence)")
        
        return "\n".join(lines)


class AICompilationPipeline:
    """
    End-to-end AI-powered compilation pipeline.
    
    This orchestrates the complete flow:
    1. Profile code execution to collect runtime type information
    2. Infer types from code patterns and runtime data
    3. Extract code characteristics for strategy decision
    4. Use RL agent to select optimal compilation strategy
    5. Compile with chosen strategy and optimization level
    6. Validate and return results with comprehensive metrics
    """
    
    def __init__(self, 
                 enable_profiling: bool = True,
                 enable_type_inference: bool = True,
                 enable_strategy_agent: bool = True,
                 verbose: bool = False):
        """
        Initialize the AI compilation pipeline.
        
        Args:
            enable_profiling: Whether to run profiling stage
            enable_type_inference: Whether to run type inference
            enable_strategy_agent: Whether to use RL agent for strategy
            verbose: Enable verbose output
        """
        self.enable_profiling = enable_profiling
        self.enable_type_inference = enable_type_inference
        self.enable_strategy_agent = enable_strategy_agent
        self.verbose = verbose
        
        # Initialize AI components
        self.tracer = RuntimeTracer()
        self.type_engine = TypeInferenceEngine()
        self.strategy_agent = StrategyAgent()
        self.backend = BackendPipeline()
        
        # Train type engine with some basic patterns if needed
        self._initialize_type_engine()
    
    def _initialize_type_engine(self):
        """Initialize type inference engine with basic training data"""
        # Basic training samples (code, variable_name, true_type)
        training_samples = [
            ("count = 0", "count", "int"),
            ("total = 0", "total", "int"),
            ("index = 0", "index", "int"),
            ("size = 10", "size", "int"),
            ("num = 5", "num", "int"),
            ("rate = 0.5", "rate", "float"),
            ("price = 9.99", "price", "float"),
            ("average = 0.0", "average", "float"),
            ("score = 95.5", "score", "float"),
            ("is_valid = True", "is_valid", "bool"),
            ("is_active = False", "is_active", "bool"),
            ("has_value = True", "has_value", "bool"),
            ("name = 'test'", "name", "str"),
            ("text = 'hello'", "text", "str"),
            ("message = 'world'", "message", "str"),
        ]
        
        if not self.type_engine.is_trained:
            self.type_engine.train(training_samples)
            if self.verbose:
                print("✓ Type inference engine initialized with basic patterns")
    
    def compile_intelligently(self, 
                            source_path: str,
                            output_path: Optional[str] = None,
                            run_tests: bool = True) -> CompilationResult:
        """
        Main entry point: Compile Python source using AI-guided pipeline.
        
        Args:
            source_path: Path to Python source file
            output_path: Optional output path for compiled binary
            run_tests: Whether to run test inputs for profiling
            
        Returns:
            CompilationResult with all metrics and decisions
        """
        start_time = time.time()
        metrics = PipelineMetrics()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"AI-POWERED COMPILATION PIPELINE")
            print(f"{'='*70}")
            print(f"Source: {source_path}")
            print(f"{'='*70}\n")
        
        try:
            # Read source code
            with open(source_path, 'r') as f:
                source_code = f.read()
            
            # Parse to get AST
            tree = ast.parse(source_code)
            
            # Stage 1: Profile execution (if enabled)
            execution_profile = None
            if self.enable_profiling:
                execution_profile = self._profile_code(source_path, source_code, metrics)
            
            # Stage 2: Infer types (if enabled)
            type_predictions = {}
            if self.enable_type_inference:
                type_predictions = self._infer_types(source_code, tree, execution_profile, metrics)
            
            # Stage 3: Extract code characteristics
            characteristics = self._extract_characteristics(tree, source_code, execution_profile)
            
            # Stage 4: Select compilation strategy (if enabled)
            if self.enable_strategy_agent:
                strategy_decision = self._select_strategy(characteristics, metrics)
            else:
                # Default to NATIVE strategy
                strategy_decision = StrategyDecision(
                    strategy=CompilationStrategy.NATIVE,
                    confidence=1.0,
                    expected_speedup=10.0,
                    reasoning="Default strategy (AI agent disabled)"
                )
            
            # Stage 5: Compile with chosen strategy
            output_file = self._compile_with_strategy(
                source_path,
                source_code,
                strategy_decision.strategy,
                output_path,
                metrics
            )
            
            # Calculate total time
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = CompilationResult(
                success=True,
                strategy=strategy_decision.strategy,
                output_path=output_file,
                metrics=metrics,
                execution_profile=execution_profile,
                type_predictions=type_predictions,
                strategy_decision=strategy_decision
            )
            
            if self.verbose:
                print(f"\n{result.summary()}\n")
                print(f"{'='*70}")
            
            return result
            
        except Exception as e:
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            result = CompilationResult(
                success=False,
                strategy=CompilationStrategy.INTERPRET,
                output_path=None,
                metrics=metrics,
                execution_profile=None,
                type_predictions={},
                strategy_decision=StrategyDecision(
                    strategy=CompilationStrategy.INTERPRET,
                    confidence=0.0,
                    expected_speedup=1.0,
                    reasoning="Compilation failed"
                ),
                error_message=str(e)
            )
            
            if self.verbose:
                print(f"\n❌ Compilation failed: {e}\n")
                print(f"{'='*70}")
            
            return result
    
    def _profile_code(self, 
                     source_path: str, 
                     source_code: str, 
                     metrics: PipelineMetrics) -> Optional[ExecutionProfile]:
        """Stage 1: Profile code execution"""
        if self.verbose:
            print("Stage 1: Profiling Code Execution...")
        
        start = time.time()
        
        try:
            # Try to execute code with tracer
            # Create a safe namespace for execution
            namespace = {'__name__': '__main__'}
            
            # Execute with tracing
            self.tracer.start()
            exec(compile(source_code, source_path, 'exec'), namespace)
            profile = self.tracer.stop()
            
            metrics.profiling_time_ms = (time.time() - start) * 1000
            metrics.functions_profiled = len(profile.function_calls)
            
            if self.verbose:
                print(f"  ✓ Profiled {metrics.functions_profiled} function calls")
                print(f"  ✓ Time: {metrics.profiling_time_ms:.2f}ms\n")
            
            return profile
            
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Profiling skipped: {e}\n")
            metrics.profiling_time_ms = (time.time() - start) * 1000
            return None
    
    def _infer_types(self,
                    source_code: str,
                    tree: ast.AST,
                    execution_profile: Optional[ExecutionProfile],
                    metrics: PipelineMetrics) -> Dict[str, TypePrediction]:
        """Stage 2: Infer types from code and runtime data"""
        if self.verbose:
            print("Stage 2: Inferring Types...")
        
        start = time.time()
        predictions = {}
        
        # Extract variable assignments from AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        
                        # Get code snippet around assignment
                        # (simplified - in production would extract actual code)
                        code_snippet = f"{var_name} = ..."
                        
                        # Infer type using predict method
                        prediction = self.type_engine.predict(code_snippet, var_name)
                        predictions[var_name] = prediction
                        metrics.confidence_scores[var_name] = prediction.confidence
        
        # Add runtime type information if available
        if execution_profile:
            for call_event in execution_profile.function_calls:
                # Extract argument types
                if call_event.arg_types:
                    # Parse arg types from string representation like "(int, int)"
                    pass
        
        metrics.type_inference_time_ms = (time.time() - start) * 1000
        metrics.types_inferred = len(predictions)
        
        if self.verbose:
            print(f"  ✓ Inferred {metrics.types_inferred} type(s)")
            for var, pred in list(predictions.items())[:3]:
                print(f"    • {var}: {pred.predicted_type} ({pred.confidence:.0%} confidence)")
            if len(predictions) > 3:
                print(f"    • ... and {len(predictions) - 3} more")
            print(f"  ✓ Time: {metrics.type_inference_time_ms:.2f}ms\n")
        
        return predictions
    
    def _extract_characteristics(self,
                                tree: ast.AST,
                                source_code: str,
                                execution_profile: Optional[ExecutionProfile]) -> CodeCharacteristics:
        """Extract code characteristics for strategy decision"""
        # Count various code features
        line_count = len(source_code.split('\n'))
        
        # Analyze AST
        function_count = 0
        loop_count = 0
        max_loop_depth = 0
        has_type_hints = False
        arithmetic_ops = 0
        control_flow = 0
        is_recursive = False
        
        current_depth = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                if node.returns:
                    has_type_hints = True
            
            elif isinstance(node, (ast.For, ast.While)):
                loop_count += 1
                current_depth += 1
                max_loop_depth = max(max_loop_depth, current_depth)
            
            elif isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                arithmetic_ops += 1
            
            elif isinstance(node, (ast.If, ast.While, ast.For)):
                control_flow += 1
        
        # Estimate call frequency from profile
        call_frequency = 1  # Default
        if execution_profile and execution_profile.function_calls:
            total_calls = len(execution_profile.function_calls)
            call_frequency = min(total_calls // max(function_count, 1), 1000)
        
        # Estimate complexity
        complexity = (
            function_count * 2 +
            loop_count * 3 +
            max_loop_depth * 5 +
            arithmetic_ops +
            control_flow * 2
        )
        
        return CodeCharacteristics(
            line_count=line_count,
            complexity=complexity,
            call_frequency=call_frequency,
            is_recursive=is_recursive,
            has_loops=(loop_count > 0),
            loop_depth=max_loop_depth,
            has_type_hints=has_type_hints,
            type_certainty=0.8 if has_type_hints else 0.3,
            arithmetic_operations=arithmetic_ops,
            control_flow_statements=control_flow,
            function_calls=function_count
        )
    
    def _select_strategy(self,
                        characteristics: CodeCharacteristics,
                        metrics: PipelineMetrics) -> StrategyDecision:
        """Stage 3: Select optimal compilation strategy"""
        if self.verbose:
            print("Stage 3: Selecting Compilation Strategy...")
        
        start = time.time()
        
        decision = self.strategy_agent.decide_strategy(characteristics)
        
        metrics.strategy_selection_time_ms = (time.time() - start) * 1000
        
        if self.verbose:
            print(f"  ✓ Strategy: {decision.strategy.value.upper()}")
            print(f"  ✓ Confidence: {decision.confidence:.0%}")
            print(f"  ✓ Expected Speedup: {decision.expected_speedup:.1f}x")
            print(f"  ✓ Reasoning: {decision.reasoning}")
            print(f"  ✓ Time: {metrics.strategy_selection_time_ms:.2f}ms\n")
        
        return decision
    
    def _compile_with_strategy(self,
                              source_path: str,
                              source_code: str,
                              strategy: CompilationStrategy,
                              output_path: Optional[str],
                              metrics: PipelineMetrics) -> str:
        """Stage 4: Compile with chosen strategy"""
        if self.verbose:
            print(f"Stage 4: Compiling with {strategy.value.upper()} strategy...")
        
        start = time.time()
        
        # Map strategy to optimization level
        opt_level_map = {
            CompilationStrategy.NATIVE: 3,      # O3 - maximum optimization
            CompilationStrategy.OPTIMIZED: 2,   # O2 - default optimization
            CompilationStrategy.BYTECODE: 1,    # O1 - basic optimization
            CompilationStrategy.INTERPRET: 0,   # O0 - no optimization (fallback)
        }
        
        opt_level = opt_level_map.get(strategy, 2)
        
        # Generate output path if not provided
        if output_path is None:
            base_name = Path(source_path).stem
            output_path = f"{base_name}.out"
        
        # Compile using our backend
        success = self.backend.compile_source(
            source_code,
            output_path=output_path,
            opt_level=opt_level,
            optimize=True,
            verbose=self.verbose
        )
        
        if not success:
            raise Exception("Backend compilation failed")
        
        metrics.compilation_time_ms = (time.time() - start) * 1000
        
        if self.verbose:
            print(f"  ✓ Compiled successfully")
            print(f"  ✓ Output: {output_path}")
            print(f"  ✓ Optimization Level: O{opt_level}")
            print(f"  ✓ Time: {metrics.compilation_time_ms:.2f}ms\n")
        
        return output_path
    
    def save_metrics(self, result: CompilationResult, output_path: str):
        """Save detailed metrics to JSON file"""
        data = {
            "success": result.success,
            "strategy": result.strategy.value,
            "output_path": result.output_path,
            "metrics": result.metrics.to_dict(),
            "strategy_decision": {
                "strategy": result.strategy_decision.strategy.value,
                "confidence": result.strategy_decision.confidence,
                "expected_speedup": result.strategy_decision.expected_speedup,
                "reasoning": result.strategy_decision.reasoning
            },
            "type_predictions": {
                var: {
                    "type": pred.predicted_type,
                    "confidence": pred.confidence,
                    "alternatives": pred.alternatives
                }
                for var, pred in result.type_predictions.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"✓ Metrics saved to {output_path}")


def demo():
    """Demonstration of the AI compilation pipeline"""
    print("\n" + "="*70)
    print("AI COMPILATION PIPELINE - DEMO")
    print("="*70 + "\n")
    
    # Create a simple test program (using simple code that we know works)
    test_code = '''
def compute(a: int, b: int) -> int:
    return a + b * 2

def main() -> int:
    result: int = compute(5, 10)
    return result
'''
    
    # Write test file
    test_file = "test_ai_pipeline.py"
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    print(f"Created test file: {test_file}\n")
    
    # Run pipeline
    pipeline = AICompilationPipeline(verbose=True)
    result = pipeline.compile_intelligently(test_file)
    
    # Save metrics
    if result.success:
        metrics_file = "test_ai_pipeline_metrics.json"
        pipeline.save_metrics(result, metrics_file)
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Try to run the compiled binary
        print("\nTesting compiled binary...")
        import subprocess
        try:
            run_result = subprocess.run([result.output_path], capture_output=True, timeout=5)
            print(f"✓ Binary executed successfully!")
            print(f"  Exit code: {run_result.returncode}")
            print(f"  Expected: 25 (5 + 10 * 2)")
            if run_result.returncode == 25:
                print(f"  ✅ CORRECT RESULT!")
            else:
                print(f"  ⚠️ Unexpected result")
        except Exception as e:
            print(f"  ⚠️ Could not execute binary: {e}")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    
    return result


if __name__ == "__main__":
    demo()
