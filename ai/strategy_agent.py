"""
Phase 2.3: AI Strategy Agent - Learn Optimal Compilation Strategies

Uses reinforcement learning to decide the best compilation strategy
for each function based on characteristics.

Strategies:
1. NATIVE - Compile to native code (fastest, highest compile cost)
2. OPTIMIZED - Compile with optimizations (balanced)
3. BYTECODE - Keep as Python bytecode (fast compile, slower runtime)
4. INTERPRET - Pure interpretation (fastest compile, slowest runtime)

Phase: 2.3 (AI Strategy Agent)
"""

import random
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CompilationStrategy(Enum):
    """Available compilation strategies"""
    NATIVE = "native"  # Full native compilation with aggressive opts
    OPTIMIZED = "optimized"  # Native with moderate opts
    BYTECODE = "bytecode"  # Keep as bytecode, optimize bytecode
    INTERPRET = "interpret"  # Pure interpretation


@dataclass
class CodeCharacteristics:
    """
    Characteristics of a function/code block used for decision making
    """
    # Size metrics
    line_count: int
    complexity: int  # Cyclomatic complexity
    
    # Call metrics
    call_frequency: int  # How often is this called
    is_recursive: bool
    has_loops: bool
    loop_depth: int
    
    # Type metrics
    has_type_hints: bool
    type_certainty: float  # 0-1, from AI type inference
    
    # Operation metrics
    arithmetic_operations: int
    control_flow_statements: int
    function_calls: int
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML model"""
        return np.array([
            self.line_count,
            self.complexity,
            self.call_frequency,
            1 if self.is_recursive else 0,
            1 if self.has_loops else 0,
            self.loop_depth,
            1 if self.has_type_hints else 0,
            self.type_certainty,
            self.arithmetic_operations,
            self.control_flow_statements,
            self.function_calls
        ], dtype=np.float32)


@dataclass
class StrategyDecision:
    """Result of strategy decision"""
    strategy: CompilationStrategy
    confidence: float
    expected_speedup: float
    reasoning: str


class StrategyAgent:
    """
    RL-based agent that learns optimal compilation strategies
    
    Uses Q-learning to learn when to apply each strategy.
    Reward: speedup achieved - compilation cost
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = 0.1  # Exploration rate
        
        # Q-table: state -> action -> value
        # State is discretized code characteristics
        # Action is compilation strategy
        self.q_table: Dict[str, Dict[CompilationStrategy, float]] = {}
        
        # Experience replay buffer
        self.experiences: List[Tuple] = []
        
        # Statistics
        self.decisions_made = 0
        self.strategies_used = {s: 0 for s in CompilationStrategy}
    
    def _discretize_features(self, chars: CodeCharacteristics) -> str:
        """Convert characteristics to discrete state"""
        # Bin features into categories
        size = "small" if chars.line_count < 10 else "medium" if chars.line_count < 50 else "large"
        freq = "rare" if chars.call_frequency < 10 else "common" if chars.call_frequency < 100 else "hot"
        loops = "no_loops" if not chars.has_loops else f"depth_{min(chars.loop_depth, 3)}"
        types = "typed" if chars.has_type_hints and chars.type_certainty > 0.8 else "partial" if chars.has_type_hints else "untyped"
        
        return f"{size}|{freq}|{loops}|{types}"
    
    def _get_q_value(self, state: str, action: CompilationStrategy) -> float:
        """Get Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {s: 0.0 for s in CompilationStrategy}
        return self.q_table[state][action]
    
    def _set_q_value(self, state: str, action: CompilationStrategy, value: float):
        """Set Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {s: 0.0 for s in CompilationStrategy}
        self.q_table[state][action] = value
    
    def decide_strategy(
        self,
        characteristics: CodeCharacteristics,
        explore: bool = True
    ) -> StrategyDecision:
        """
        Decide compilation strategy for given code
        
        Args:
            characteristics: Code characteristics
            explore: Whether to explore (vs exploit)
        
        Returns:
            StrategyDecision with chosen strategy
        """
        state = self._discretize_features(characteristics)
        
        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            strategy = random.choice(list(CompilationStrategy))
            reasoning = "Exploring: random choice"
        else:
            # Exploit: choose best action
            q_values = {s: self._get_q_value(state, s) for s in CompilationStrategy}
            strategy = max(q_values, key=q_values.get)
            reasoning = self._generate_reasoning(characteristics, strategy)
        
        # Update statistics
        self.decisions_made += 1
        self.strategies_used[strategy] += 1
        
        # Estimate expected speedup (would be learned from experience)
        expected_speedup = self._estimate_speedup(characteristics, strategy)
        confidence = self._get_q_value(state, strategy) / 100.0  # Normalize to 0-1
        
        return StrategyDecision(
            strategy=strategy,
            confidence=min(confidence, 1.0),
            expected_speedup=expected_speedup,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, chars: CodeCharacteristics, strategy: CompilationStrategy) -> str:
        """Generate human-readable reasoning for decision"""
        reasons = []
        
        if strategy == CompilationStrategy.NATIVE:
            if chars.call_frequency > 100:
                reasons.append("Hot function (>100 calls)")
            if chars.has_loops:
                reasons.append("Contains loops - benefits from native compilation")
            if chars.has_type_hints:
                reasons.append("Has type hints - can optimize well")
        
        elif strategy == CompilationStrategy.OPTIMIZED:
            if 10 <= chars.call_frequency <= 100:
                reasons.append("Moderately called function")
            if chars.line_count < 50:
                reasons.append("Small function - fast compilation")
        
        elif strategy == CompilationStrategy.BYTECODE:
            if chars.call_frequency < 10:
                reasons.append("Rarely called - not worth full compilation")
            if not chars.has_type_hints:
                reasons.append("No type hints - harder to optimize")
        
        elif strategy == CompilationStrategy.INTERPRET:
            if chars.line_count < 5:
                reasons.append("Tiny function - interpretation overhead minimal")
            if chars.call_frequency < 5:
                reasons.append("Very rare function")
        
        return "; ".join(reasons) if reasons else "Default choice"
    
    def _estimate_speedup(self, chars: CodeCharacteristics, strategy: CompilationStrategy) -> float:
        """Estimate expected speedup (simplified heuristic)"""
        base_speedup = {
            CompilationStrategy.NATIVE: 10.0,
            CompilationStrategy.OPTIMIZED: 5.0,
            CompilationStrategy.BYTECODE: 2.0,
            CompilationStrategy.INTERPRET: 1.0
        }[strategy]
        
        # Adjust based on characteristics
        if chars.has_loops:
            base_speedup *= 1.5
        if chars.has_type_hints:
            base_speedup *= 1.2
        if chars.is_recursive:
            base_speedup *= 1.3
        
        return base_speedup
    
    def learn(self, state: CodeCharacteristics, action: CompilationStrategy, reward: float):
        """
        Update Q-values based on observed reward
        
        Args:
            state: Code characteristics
            action: Strategy that was used
            reward: Observed reward (speedup - compile_cost)
        """
        state_key = self._discretize_features(state)
        
        # Get current Q-value
        current_q = self._get_q_value(state_key, action)
        
        # Q-learning update: Q(s,a) = Q(s,a) + α(r + γ*max(Q(s',a')) - Q(s,a))
        # Simplified: no next state since this is terminal
        new_q = current_q + self.learning_rate * (reward - current_q)
        
        self._set_q_value(state_key, action, new_q)
        
        # Store experience
        self.experiences.append((state, action, reward))
    
    def save(self, filepath: str):
        """Save learned Q-table"""
        data = {
            'q_table': {
                state: {s.value: v for s, v in actions.items()}
                for state, actions in self.q_table.items()
            },
            'statistics': {
                'decisions_made': self.decisions_made,
                'strategies_used': {s.value: c for s, c in self.strategies_used.items()}
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load learned Q-table"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct Q-table
        self.q_table = {}
        for state, actions in data['q_table'].items():
            self.q_table[state] = {
                CompilationStrategy(s): v for s, v in actions.items()
            }
        
        # Restore statistics
        stats = data['statistics']
        self.decisions_made = stats['decisions_made']
        self.strategies_used = {
            CompilationStrategy(s): c for s, c in stats['strategies_used'].items()
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("AI COMPILATION STRATEGY AGENT - Demo")
    print("="*80)
    
    # Create agent
    agent = StrategyAgent()
    
    # Test cases
    test_cases = [
        CodeCharacteristics(
            line_count=5, complexity=2, call_frequency=1000,
            is_recursive=False, has_loops=True, loop_depth=2,
            has_type_hints=True, type_certainty=0.9,
            arithmetic_operations=10, control_flow_statements=2, function_calls=1
        ),
        CodeCharacteristics(
            line_count=3, complexity=1, call_frequency=5,
            is_recursive=False, has_loops=False, loop_depth=0,
            has_type_hints=False, type_certainty=0.3,
            arithmetic_operations=2, control_flow_statements=0, function_calls=0
        ),
        CodeCharacteristics(
            line_count=20, complexity=5, call_frequency=500,
            is_recursive=True, has_loops=True, loop_depth=1,
            has_type_hints=True, type_certainty=0.95,
            arithmetic_operations=15, control_flow_statements=5, function_calls=3
        ),
    ]
    
    descriptions = [
        "Hot loop-heavy function with type hints",
        "Tiny rarely-called function without types",
        "Complex recursive function with high call frequency"
    ]
    
    print("\n Strategy Decisions:")
    print("-" * 80)
    
    for i, (chars, desc) in enumerate(zip(test_cases, descriptions), 1):
        decision = agent.decide_strategy(chars, explore=False)
        
        print(f"\nCase {i}: {desc}")
        print(f"  Strategy: {decision.strategy.value.upper()}")
        print(f"  Expected Speedup: {decision.expected_speedup:.1f}x")
        print(f"  Reasoning: {decision.reasoning}")
    
    # Simulate learning
    print("\n" + "-" * 80)
    print("Simulating Learning Process...")
    print("-" * 80)
    
    for i in range(100):
        chars = test_cases[i % len(test_cases)]
        decision = agent.decide_strategy(chars, explore=True)
        
        # Simulate reward (in reality, this would come from actual performance)
        reward = random.gauss(decision.expected_speedup, 1.0)
        agent.learn(chars, decision.strategy, reward)
    
    print(f"\n✅ Trained on {agent.decisions_made} decisions")
    print(f"\nStrategy Distribution:")
    for strategy, count in agent.strategies_used.items():
        pct = count / agent.decisions_made * 100
        print(f"  {strategy.value}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ Strategy agent working!")
    print("="*80)
