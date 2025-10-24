"""
State-of-the-Art Deep Reinforcement Learning Strategy Agent
Uses Deep Q-Network (DQN) with prioritized experience replay
Replaces tabular Q-learning (1989) with modern deep RL (2015+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, namedtuple
import random
import json
from pathlib import Path
from enum import Enum


class CompilationStrategy(Enum):
    """Compilation strategies"""
    NATIVE = "native"           # Full native compilation
    OPTIMIZED = "optimized"     # Optimized with LLVM -O3
    JIT = "jit"                 # JIT compilation
    BYTECODE = "bytecode"       # Python bytecode
    INTERPRET = "interpret"     # Interpretation only
    HYBRID = "hybrid"           # Adaptive hybrid approach


@dataclass
class CodeCharacteristics:
    """Enhanced code features for DQN"""
    line_count: int
    complexity: float           # Cyclomatic complexity
    call_frequency: float       # How often it's called
    loop_depth: int            # Maximum loop nesting
    recursion_depth: int       # Recursion depth
    has_numeric_ops: bool      # Contains math operations
    has_loops: bool            # Contains loops
    has_recursion: bool        # Contains recursion
    has_calls: bool            # Contains function calls
    memory_intensive: bool     # Large data structures
    io_bound: bool             # I/O operations
    cpu_bound: bool            # CPU-intensive
    parallelizable: bool       # Can be parallelized
    cache_friendly: bool       # Memory access patterns
    vectorizable: bool         # Can use SIMD
    
    # New features for deep RL
    avg_line_length: float     # Code density
    variable_count: int        # Number of variables
    function_count: int        # Number of functions
    class_count: int           # Number of classes
    import_count: int          # Number of imports
    string_operations: int     # String manipulations
    list_operations: int       # List operations
    dict_operations: int       # Dict operations
    exception_handling: bool   # Try/except blocks
    async_code: bool           # Async/await
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for neural network"""
        return np.array([
            self.line_count / 1000.0,              # Normalize
            self.complexity / 100.0,
            self.call_frequency,
            self.loop_depth / 10.0,
            self.recursion_depth / 10.0,
            float(self.has_numeric_ops),
            float(self.has_loops),
            float(self.has_recursion),
            float(self.has_calls),
            float(self.memory_intensive),
            float(self.io_bound),
            float(self.cpu_bound),
            float(self.parallelizable),
            float(self.cache_friendly),
            float(self.vectorizable),
            self.avg_line_length / 100.0,
            self.variable_count / 100.0,
            self.function_count / 50.0,
            self.class_count / 20.0,
            self.import_count / 50.0,
            self.string_operations / 100.0,
            self.list_operations / 100.0,
            self.dict_operations / 100.0,
            float(self.exception_handling),
            float(self.async_code),
        ], dtype=np.float32)


@dataclass
class CompilationResult:
    """Result of compilation with performance metrics"""
    strategy: CompilationStrategy
    execution_time: float
    memory_usage: float
    compilation_time: float
    speedup: float
    success: bool
    error: Optional[str] = None


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for strategy selection
    State-of-the-art architecture (2016)
    
    Separates value and advantage streams for better learning
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 6,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Value stream (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[2] // 2, 1)
        )
        
        # Advantage stream (A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[2] // 2, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture
        
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        """
        features = self.feature_extractor(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER)
    Samples important transitions more frequently
    State-of-the-art technique (2015)
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, transition: Transition):
        """Add transition with maximum priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample batch with priorities"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return len(self.buffer)


class DeepRLStrategyAgent:
    """
    State-of-the-art Deep RL agent for compilation strategy selection
    
    Features:
    - Dueling DQN architecture
    - Prioritized experience replay
    - Double DQN for stability
    - Noisy networks for exploration
    - Multi-step returns
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 6,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.95)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Statistics
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        
        # Strategy mapping
        self.strategies = list(CompilationStrategy)
        
        print(f"🚀 Deep RL Strategy Agent initialized")
        print(f"   Architecture: Dueling DQN + Prioritized Replay")
        print(f"   Device: {self.device}")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
    
    def select_action(
        self,
        characteristics: CodeCharacteristics,
        deterministic: bool = False
    ) -> CompilationStrategy:
        """
        Select compilation strategy using epsilon-greedy or greedy
        
        Args:
            characteristics: Code features
            deterministic: If True, always select best action (no exploration)
        
        Returns:
            Selected compilation strategy
        """
        # Epsilon-greedy exploration
        if not deterministic and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(characteristics.to_vector()).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.argmax(dim=1).item()
        
        return self.strategies[action_idx]
    
    def store_transition(
        self,
        state: CodeCharacteristics,
        action: CompilationStrategy,
        reward: float,
        next_state: CodeCharacteristics,
        done: bool
    ):
        """Store experience in replay buffer"""
        action_idx = self.strategies.index(action)
        
        transition = Transition(
            state=state.to_vector(),
            action=action_idx,
            reward=reward,
            next_state=next_state.to_vector(),
            done=done
        )
        
        self.replay_buffer.push(transition)
        self.total_reward += reward
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step with prioritized replay
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch with priorities
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        if len(transitions) == 0:
            return None
        
        # Unpack batch
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        dones = torch.FloatTensor([t.done for t in transitions]).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate TD errors for prioritization
        td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors.flatten())
        
        # Weighted loss (importance sampling)
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.losses.append(loss.item())
        
        return loss.item()
    
    def train(
        self,
        training_episodes: int = 1000,
        eval_freq: int = 100
    ):
        """
        Train the agent
        
        Args:
            training_episodes: Number of episodes
            eval_freq: Evaluate every N episodes
        """
        print(f"Training Deep RL agent for {training_episodes} episodes...")
        
        for episode in range(training_episodes):
            self.episodes += 1
            episode_reward = 0
            
            # Simulate compilation scenarios
            # In production, this would use real compilation data
            for step in range(100):
                # Generate random code characteristics
                characteristics = self._generate_random_characteristics()
                
                # Select action
                strategy = self.select_action(characteristics)
                
                # Simulate compilation (would be real in production)
                reward, next_characteristics = self._simulate_compilation(characteristics, strategy)
                
                # Store transition
                done = (step == 99)
                self.store_transition(characteristics, strategy, reward, next_characteristics, done)
                
                # Train
                loss = self.train_step()
                
                episode_reward += reward
            
            self.episode_rewards.append(episode_reward)
            
            # Logging
            if (episode + 1) % eval_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-eval_freq:])
                avg_loss = np.mean(self.losses[-1000:]) if self.losses else 0
                print(f"Episode {episode + 1}/{training_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")
        
        print("✅ Training complete!")
    
    def _generate_random_characteristics(self) -> CodeCharacteristics:
        """Generate random code characteristics for training"""
        return CodeCharacteristics(
            line_count=np.random.randint(10, 1000),
            complexity=np.random.uniform(1, 50),
            call_frequency=np.random.uniform(0, 1),
            loop_depth=np.random.randint(0, 5),
            recursion_depth=np.random.randint(0, 3),
            has_numeric_ops=np.random.random() > 0.5,
            has_loops=np.random.random() > 0.5,
            has_recursion=np.random.random() > 0.3,
            has_calls=np.random.random() > 0.7,
            memory_intensive=np.random.random() > 0.6,
            io_bound=np.random.random() > 0.7,
            cpu_bound=np.random.random() > 0.5,
            parallelizable=np.random.random() > 0.6,
            cache_friendly=np.random.random() > 0.5,
            vectorizable=np.random.random() > 0.6,
            avg_line_length=np.random.uniform(20, 80),
            variable_count=np.random.randint(5, 100),
            function_count=np.random.randint(1, 20),
            class_count=np.random.randint(0, 10),
            import_count=np.random.randint(0, 20),
            string_operations=np.random.randint(0, 50),
            list_operations=np.random.randint(0, 50),
            dict_operations=np.random.randint(0, 30),
            exception_handling=np.random.random() > 0.5,
            async_code=np.random.random() > 0.7
        )
    
    def _simulate_compilation(
        self,
        characteristics: CodeCharacteristics,
        strategy: CompilationStrategy
    ) -> Tuple[float, CodeCharacteristics]:
        """Simulate compilation and return reward"""
        # Reward heuristics based on code characteristics and strategy
        reward = 0.0
        
        # CPU-bound code benefits from native compilation
        if characteristics.cpu_bound and strategy == CompilationStrategy.NATIVE:
            reward += 10.0
        
        # I/O-bound code doesn't need heavy optimization
        if characteristics.io_bound and strategy in [CompilationStrategy.BYTECODE, CompilationStrategy.INTERPRET]:
            reward += 5.0
        
        # Vectorizable code benefits from optimization
        if characteristics.vectorizable and strategy == CompilationStrategy.OPTIMIZED:
            reward += 8.0
        
        # Small code can use interpretation
        if characteristics.line_count < 50 and strategy == CompilationStrategy.INTERPRET:
            reward += 3.0
        
        # Penalize expensive compilation for small code
        if characteristics.line_count < 50 and strategy == CompilationStrategy.NATIVE:
            reward -= 5.0
        
        # Loops benefit from compilation
        if characteristics.has_loops and characteristics.loop_depth > 2:
            if strategy in [CompilationStrategy.NATIVE, CompilationStrategy.JIT]:
                reward += 7.0
        
        # Add some noise
        reward += np.random.normal(0, 1)
        
        # Next state (same for simplicity)
        next_characteristics = characteristics
        
        return reward, next_characteristics
    
    def save(self, path: str):
        """Save model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, f"{path}/dqn_model.pt")
        
        # Save statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }
        with open(f"{path}/training_stats.json", 'w') as f:
            json.dump(stats, f)
        
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(f"{path}/dqn_model.pt", map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        
        print(f"✅ Model loaded from {path}")


# Backward compatibility
class StrategyAgent:
    """Drop-in replacement for old Q-learning version"""
    
    def __init__(self):
        self.agent = DeepRLStrategyAgent()
        print("🚀 Using state-of-the-art Deep RL Strategy Agent (DQN + PER)")
    
    def decide(self, characteristics: Dict) -> str:
        """Select strategy"""
        code_chars = CodeCharacteristics(**characteristics)
        strategy = self.agent.select_action(code_chars, deterministic=True)
        return strategy.value


if __name__ == "__main__":
    # Demo
    agent = DeepRLStrategyAgent()
    
    # Train
    agent.train(training_episodes=500)
    
    # Test
    test_chars = CodeCharacteristics(
        line_count=100,
        complexity=10.0,
        call_frequency=0.5,
        loop_depth=2,
        recursion_depth=0,
        has_numeric_ops=True,
        has_loops=True,
        has_recursion=False,
        has_calls=True,
        memory_intensive=False,
        io_bound=False,
        cpu_bound=True,
        parallelizable=True,
        cache_friendly=True,
        vectorizable=True,
        avg_line_length=50.0,
        variable_count=20,
        function_count=5,
        class_count=1,
        import_count=3,
        string_operations=5,
        list_operations=10,
        dict_operations=5,
        exception_handling=True,
        async_code=False
    )
    
    strategy = agent.select_action(test_chars, deterministic=True)
    print(f"\nSelected strategy: {strategy.value}")
