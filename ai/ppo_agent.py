"""
Advanced Proximal Policy Optimization (PPO) Agent
State-of-the-art policy gradient method (2017)
More stable and sample-efficient than DQN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class PPOMemory:
    """Memory buffer for PPO"""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    
    def __init__(self):
        self.clear()
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_tensors(self, device):
        """Convert to tensors"""
        return (
            torch.FloatTensor(np.array(self.states)).to(device),
            torch.LongTensor(self.actions).to(device),
            torch.FloatTensor(self.rewards).to(device),
            torch.FloatTensor(self.values).to(device),
            torch.FloatTensor(self.log_probs).to(device),
            torch.FloatTensor(self.dones).to(device)
        )


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared feature extraction
    Actor: Policy network (outputs action probabilities)
    Critic: Value network (outputs state value)
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 6,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[1] // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[1] // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            action_probs: Probability distribution over actions
            value: State value estimate
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy"""
        action_probs, value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)))
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Advantages over DQN:
    - More stable training
    - Better sample efficiency
    - Continuous action spaces support
    - Clipped surrogate objective prevents large policy updates
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 6,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Memory
        self.memory = PPOMemory()
        
        # Statistics
        self.episodes = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        
        print(f"🚀 PPO Agent initialized")
        print(f"   Device: {self.device}")
        print(f"   Clip epsilon: {clip_epsilon}")
        print(f"   GAE lambda: {gae_lambda}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in memory"""
        self.memory.store(state, action, reward, value, log_prob, done)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        GAE balances bias and variance in advantage estimation
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        values = torch.cat([values, torch.tensor([next_value]).to(self.device)])
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t + 1]
            else:
                next_value = values[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + (γλ) * δ_{t+1} + ...
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        # Returns: R_t = A_t + V(s_t)
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        PPO update with clipped surrogate objective
        
        Returns:
            Dictionary of losses
        """
        if len(self.memory.states) == 0:
            return {}
        
        # Get data from memory
        states, actions, rewards, old_values, old_log_probs, dones = self.memory.get_tensors(self.device)
        
        # Compute advantages with GAE
        advantages, returns = self.compute_gae(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for epoch in range(self.ppo_epochs):
            # Mini-batch training
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                action_probs, values = self.network(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                
                # Ratio: π_new / π_old
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Record losses
                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
                self.entropy_losses.append(entropy_loss.item())
        
        # Clear memory
        self.memory.clear()
        
        return {
            'actor_loss': np.mean(self.actor_losses[-10:]) if self.actor_losses else 0,
            'critic_loss': np.mean(self.critic_losses[-10:]) if self.critic_losses else 0,
            'entropy': -np.mean(self.entropy_losses[-10:]) if self.entropy_losses else 0
        }
    
    def train(self, num_episodes: int = 1000, steps_per_episode: int = 100):
        """Train PPO agent"""
        print(f"Training PPO for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_reward = 0
            
            # Simulate episode (in production, use real compilation data)
            for step in range(steps_per_episode):
                # Generate state
                state = self._generate_random_state()
                
                # Select action
                action, log_prob, value = self.select_action(state)
                
                # Simulate environment
                reward, next_state = self._simulate_step(state, action)
                done = (step == steps_per_episode - 1)
                
                # Store transition
                self.store_transition(state, action, reward, value, log_prob, done)
                
                episode_reward += reward
                self.total_steps += 1
            
            # Update policy
            losses = self.update()
            
            self.episodes += 1
            self.episode_rewards.append(episode_reward)
            
            # Logging
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Actor Loss: {losses.get('actor_loss', 0):.4f}")
                print(f"  Critic Loss: {losses.get('critic_loss', 0):.4f}")
                print(f"  Entropy: {losses.get('entropy', 0):.4f}")
        
        print("✅ PPO training complete!")
    
    def _generate_random_state(self) -> np.ndarray:
        """Generate random state for training"""
        return np.random.randn(self.state_dim).astype(np.float32)
    
    def _simulate_step(self, state: np.ndarray, action: int) -> Tuple[float, np.ndarray]:
        """Simulate environment step"""
        # Simple reward function
        reward = -abs(state[0] - action) + np.random.normal(0, 0.1)
        next_state = state + np.random.randn(self.state_dim) * 0.1
        return reward, next_state
    
    def save(self, path: str):
        """Save model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes': self.episodes,
            'total_steps': self.total_steps
        }, f"{path}/ppo_model.pt")
        
        print(f"✅ PPO model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(f"{path}/ppo_model.pt", map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episodes = checkpoint['episodes']
        self.total_steps = checkpoint['total_steps']
        
        print(f"✅ PPO model loaded from {path}")


if __name__ == "__main__":
    # Demo
    agent = PPOAgent()
    agent.train(num_episodes=500)
    
    # Test
    test_state = np.random.randn(25).astype(np.float32)
    action, _, _ = agent.select_action(test_state, deterministic=True)
    print(f"Test action: {action}")
