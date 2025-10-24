"""
Multi-Agent System for Compilation Optimization
Coordinated agents for different optimization objectives
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class OptimizationObjective(Enum):
    """Different optimization objectives"""
    SPEED = "speed"                     # Maximize execution speed
    MEMORY = "memory"                   # Minimize memory usage
    COMPILATION_TIME = "compile_time"  # Fast compilation
    BINARY_SIZE = "binary_size"        # Minimize binary size
    ENERGY = "energy"                  # Energy efficiency
    BALANCED = "balanced"              # Balance all objectives


@dataclass
class AgentDecision:
    """Decision from a single agent"""
    agent_id: str
    strategy: str
    confidence: float
    estimated_value: float
    reasoning: str


@dataclass
class ConsensusDecision:
    """Final decision after multi-agent consensus"""
    final_strategy: str
    confidence: float
    agent_votes: Dict[str, AgentDecision]
    consensus_method: str
    total_reward_estimate: float


class SpecializedAgent:
    """
    Agent specialized for a specific optimization objective
    """
    
    def __init__(
        self,
        agent_id: str,
        objective: OptimizationObjective,
        policy_network: torch.nn.Module,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.objective = objective
        self.policy = policy_network
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Objective-specific weights
        self.objective_weights = self._get_objective_weights()
        
        # Performance history
        self.decisions = []
        self.rewards = []
    
    def _get_objective_weights(self) -> Dict[str, float]:
        """Get weights for different metrics based on objective"""
        if self.objective == OptimizationObjective.SPEED:
            return {'speed': 1.0, 'memory': 0.1, 'compile_time': 0.1, 'binary_size': 0.0, 'energy': 0.2}
        elif self.objective == OptimizationObjective.MEMORY:
            return {'speed': 0.2, 'memory': 1.0, 'compile_time': 0.1, 'binary_size': 0.3, 'energy': 0.2}
        elif self.objective == OptimizationObjective.COMPILATION_TIME:
            return {'speed': 0.3, 'memory': 0.1, 'compile_time': 1.0, 'binary_size': 0.1, 'energy': 0.1}
        elif self.objective == OptimizationObjective.BINARY_SIZE:
            return {'speed': 0.2, 'memory': 0.4, 'compile_time': 0.1, 'binary_size': 1.0, 'energy': 0.1}
        elif self.objective == OptimizationObjective.ENERGY:
            return {'speed': 0.3, 'memory': 0.3, 'compile_time': 0.2, 'binary_size': 0.2, 'energy': 1.0}
        else:  # BALANCED
            return {'speed': 0.3, 'memory': 0.3, 'compile_time': 0.2, 'binary_size': 0.1, 'energy': 0.1}
    
    def decide(self, state: np.ndarray) -> AgentDecision:
        """Make decision based on state and objective"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
            action = torch.argmax(action_probs, dim=-1).item()
            confidence = action_probs[0, action].item()
        
        # Estimate value based on objective
        estimated_value = self._estimate_value(state, action)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(state, action)
        
        strategies = ['native', 'optimized', 'jit', 'bytecode', 'interpret', 'hybrid']
        
        decision = AgentDecision(
            agent_id=self.agent_id,
            strategy=strategies[action],
            confidence=confidence,
            estimated_value=estimated_value,
            reasoning=reasoning
        )
        
        self.decisions.append(decision)
        
        return decision
    
    def _estimate_value(self, state: np.ndarray, action: int) -> float:
        """Estimate value of action for this objective"""
        # Heuristic value estimation based on state features
        value = 0.0
        
        # CPU-bound code (state[11])
        if state[11] > 0.5:
            if action in [0, 1]:  # native, optimized
                value += self.objective_weights['speed'] * 10
        
        # Memory-intensive (state[9])
        if state[9] > 0.5:
            if action in [3, 4]:  # bytecode, interpret
                value += self.objective_weights['memory'] * 8
        
        # Large codebase
        if state[0] > 0.5:  # line_count
            if action in [2, 3]:  # jit, bytecode
                value += self.objective_weights['compile_time'] * 6
        
        return value
    
    def _generate_reasoning(self, state: np.ndarray, action: int) -> str:
        """Generate human-readable reasoning"""
        strategies = ['native', 'optimized', 'jit', 'bytecode', 'interpret', 'hybrid']
        
        reasons = []
        
        if self.objective == OptimizationObjective.SPEED:
            reasons.append(f"Optimizing for maximum execution speed")
        elif self.objective == OptimizationObjective.MEMORY:
            reasons.append(f"Optimizing for minimal memory usage")
        elif self.objective == OptimizationObjective.COMPILATION_TIME:
            reasons.append(f"Optimizing for fast compilation")
        
        reasons.append(f"Selected {strategies[action]} strategy")
        
        return " | ".join(reasons)
    
    def update_reward(self, reward: float):
        """Update with received reward"""
        self.rewards.append(reward)


class MultiAgentSystem:
    """
    Coordinated multi-agent system for compilation optimization
    
    Features:
    - Multiple specialized agents with different objectives
    - Consensus mechanisms (voting, averaging, hierarchical)
    - Communication between agents
    - Meta-controller for agent selection
    """
    
    def __init__(
        self,
        agents: List[SpecializedAgent],
        consensus_method: str = "weighted_voting",
        device: str = "cpu"
    ):
        self.agents = agents
        self.consensus_method = consensus_method
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Meta-controller for agent selection
        self.meta_controller = self._build_meta_controller()
        
        # Communication channels
        self.message_buffer = {}
        
        # Performance tracking
        self.decisions_history = []
        self.consensus_history = []
        
        print(f"🤖 Multi-Agent System initialized")
        print(f"   Agents: {[a.agent_id for a in agents]}")
        print(f"   Consensus: {consensus_method}")
    
    def _build_meta_controller(self) -> torch.nn.Module:
        """Build meta-controller to select which agents to consult"""
        return torch.nn.Sequential(
            torch.nn.Linear(25, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(self.agents)),
            torch.nn.Sigmoid()  # Weights for each agent
        ).to(self.device)
    
    def decide(
        self,
        state: np.ndarray,
        context: Optional[Dict] = None
    ) -> ConsensusDecision:
        """
        Multi-agent decision making
        
        Args:
            state: Current state
            context: Additional context (user preferences, constraints, etc.)
        
        Returns:
            Consensus decision
        """
        # Get decisions from all agents
        agent_decisions = {}
        for agent in self.agents:
            decision = agent.decide(state)
            agent_decisions[agent.agent_id] = decision
        
        # Get meta-controller weights
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            agent_weights = self.meta_controller(state_tensor)[0].cpu().numpy()
        
        # Apply consensus mechanism
        if self.consensus_method == "weighted_voting":
            consensus = self._weighted_voting(agent_decisions, agent_weights)
        elif self.consensus_method == "highest_confidence":
            consensus = self._highest_confidence(agent_decisions)
        elif self.consensus_method == "pareto_optimal":
            consensus = self._pareto_optimal(agent_decisions)
        else:
            consensus = self._simple_voting(agent_decisions)
        
        self.consensus_history.append(consensus)
        
        return consensus
    
    def _weighted_voting(
        self,
        decisions: Dict[str, AgentDecision],
        weights: np.ndarray
    ) -> ConsensusDecision:
        """Weighted voting based on meta-controller"""
        # Count votes with weights
        strategy_scores = {}
        
        for i, (agent_id, decision) in enumerate(decisions.items()):
            strategy = decision.strategy
            weight = weights[i]
            confidence = decision.confidence
            
            score = weight * confidence * decision.estimated_value
            
            if strategy not in strategy_scores:
                strategy_scores[strategy] = 0.0
            strategy_scores[strategy] += score
        
        # Select best strategy
        final_strategy = max(strategy_scores, key=strategy_scores.get)
        final_confidence = strategy_scores[final_strategy] / sum(weights)
        
        # Estimate total reward
        total_reward_estimate = sum(
            d.estimated_value * weights[i]
            for i, d in enumerate(decisions.values())
        )
        
        return ConsensusDecision(
            final_strategy=final_strategy,
            confidence=final_confidence,
            agent_votes=decisions,
            consensus_method="weighted_voting",
            total_reward_estimate=total_reward_estimate
        )
    
    def _highest_confidence(self, decisions: Dict[str, AgentDecision]) -> ConsensusDecision:
        """Select decision with highest confidence"""
        best_decision = max(decisions.values(), key=lambda d: d.confidence)
        
        return ConsensusDecision(
            final_strategy=best_decision.strategy,
            confidence=best_decision.confidence,
            agent_votes=decisions,
            consensus_method="highest_confidence",
            total_reward_estimate=best_decision.estimated_value
        )
    
    def _simple_voting(self, decisions: Dict[str, AgentDecision]) -> ConsensusDecision:
        """Simple majority voting"""
        votes = {}
        for decision in decisions.values():
            strategy = decision.strategy
            votes[strategy] = votes.get(strategy, 0) + 1
        
        final_strategy = max(votes, key=votes.get)
        confidence = votes[final_strategy] / len(decisions)
        
        avg_reward = np.mean([d.estimated_value for d in decisions.values()])
        
        return ConsensusDecision(
            final_strategy=final_strategy,
            confidence=confidence,
            agent_votes=decisions,
            consensus_method="simple_voting",
            total_reward_estimate=avg_reward
        )
    
    def _pareto_optimal(self, decisions: Dict[str, AgentDecision]) -> ConsensusDecision:
        """Select Pareto-optimal decision"""
        # Find decision that's not dominated by any other
        decision_list = list(decisions.values())
        
        for i, decision in enumerate(decision_list):
            is_dominated = False
            
            for j, other in enumerate(decision_list):
                if i != j:
                    # Check if other dominates decision
                    if (other.confidence >= decision.confidence and
                        other.estimated_value >= decision.estimated_value and
                        (other.confidence > decision.confidence or
                         other.estimated_value > decision.estimated_value)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                return ConsensusDecision(
                    final_strategy=decision.strategy,
                    confidence=decision.confidence,
                    agent_votes=decisions,
                    consensus_method="pareto_optimal",
                    total_reward_estimate=decision.estimated_value
                )
        
        # Fallback to highest value
        return self._highest_confidence(decisions)
    
    def communicate(self, sender_id: str, message: Dict):
        """Enable agent communication"""
        self.message_buffer[sender_id] = message
    
    def get_messages(self, receiver_id: str) -> List[Dict]:
        """Get messages for an agent"""
        return [
            {'sender': sender_id, 'content': msg}
            for sender_id, msg in self.message_buffer.items()
            if sender_id != receiver_id
        ]
    
    def train_meta_controller(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        num_epochs: int = 100
    ):
        """Train meta-controller to learn agent weights"""
        optimizer = torch.optim.Adam(self.meta_controller.parameters(), lr=1e-3)
        
        print(f"Training meta-controller...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for state, optimal_weights in training_data:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                target_weights = torch.FloatTensor(optimal_weights).unsqueeze(0).to(self.device)
                
                predicted_weights = self.meta_controller(state_tensor)
                loss = torch.nn.functional.mse_loss(predicted_weights, target_weights)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(training_data):.4f}")
        
        print("✅ Meta-controller training complete!")
    
    def save(self, path: str):
        """Save multi-agent system"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save meta-controller
        torch.save(self.meta_controller.state_dict(), f"{path}/meta_controller.pt")
        
        # Save agent policies
        for i, agent in enumerate(self.agents):
            torch.save(agent.policy.state_dict(), f"{path}/agent_{i}_policy.pt")
        
        # Save metadata
        metadata = {
            'num_agents': len(self.agents),
            'consensus_method': self.consensus_method,
            'agent_objectives': [agent.objective.value for agent in self.agents]
        }
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"✅ Multi-agent system saved to {path}")


def create_multi_agent_system(device: str = "cpu") -> MultiAgentSystem:
    """Factory function to create multi-agent system"""
    
    # Create simple policies for each agent
    def create_policy():
        return torch.nn.Sequential(
            torch.nn.Linear(25, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 6),
            torch.nn.Softmax(dim=-1)
        )
    
    agents = [
        SpecializedAgent("speed_agent", OptimizationObjective.SPEED, create_policy(), device),
        SpecializedAgent("memory_agent", OptimizationObjective.MEMORY, create_policy(), device),
        SpecializedAgent("compile_agent", OptimizationObjective.COMPILATION_TIME, create_policy(), device),
        SpecializedAgent("balanced_agent", OptimizationObjective.BALANCED, create_policy(), device),
    ]
    
    return MultiAgentSystem(agents, consensus_method="weighted_voting", device=device)


if __name__ == "__main__":
    # Demo
    print("=== Multi-Agent System Demo ===")
    
    mas = create_multi_agent_system()
    
    # Test decision
    test_state = np.random.randn(25).astype(np.float32)
    test_state[11] = 0.9  # CPU-bound
    
    decision = mas.decide(test_state)
    
    print(f"\nConsensus Decision:")
    print(f"  Strategy: {decision.final_strategy}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Method: {decision.consensus_method}")
    print(f"\nAgent Votes:")
    for agent_id, vote in decision.agent_votes.items():
        print(f"  {agent_id}: {vote.strategy} (conf={vote.confidence:.2f}, value={vote.estimated_value:.2f})")
