"""
Meta-Learning for Fast Adaptation Across Projects
Model-Agnostic Meta-Learning (MAML) for compilation optimization
Enables few-shot learning on new codebases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Task:
    """A meta-learning task (e.g., optimizing a specific project)"""
    support_set: List[Tuple[np.ndarray, int]]  # Training examples
    query_set: List[Tuple[np.ndarray, int]]     # Test examples
    task_id: str
    metadata: Dict


class MetaNetwork(nn.Module):
    """
    Fast-adaptation network for meta-learning
    Small network that can quickly adapt to new tasks
    """
    
    def __init__(self, input_dim: int = 25, output_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def get_flat_params(self) -> torch.Tensor:
        """Get flattened parameters"""
        return torch.cat([p.view(-1) for p in self.parameters()])
    
    def set_flat_params(self, flat_params: torch.Tensor):
        """Set parameters from flattened tensor"""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p))
            offset += numel
    
    def get_flat_grads(self) -> torch.Tensor:
        """Get flattened gradients"""
        return torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) 
                         for p in self.parameters()])


class MAMLAgent:
    """
    Model-Agnostic Meta-Learning Agent
    
    Learns how to quickly adapt to new compilation tasks with minimal data
    Perfect for adapting to new codebases or programming patterns
    """
    
    def __init__(
        self,
        input_dim: int = 25,
        output_dim: int = 6,
        hidden_dim: int = 128,
        meta_lr: float = 1e-3,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # Meta-model
        self.model = MetaNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # Statistics
        self.meta_train_losses = []
        self.meta_val_losses = []
        self.adaptation_curves = []
        
        print(f"🚀 MAML Agent initialized")
        print(f"   Inner steps: {inner_steps}")
        print(f"   Inner LR: {inner_lr}")
        print(f"   Meta LR: {meta_lr}")
    
    def inner_loop(
        self,
        support_set: List[Tuple[np.ndarray, int]],
        fast_weights: Optional[Dict] = None
    ) -> Dict:
        """
        Inner loop: adapt to a specific task
        
        Args:
            support_set: Training examples for this task
            fast_weights: Current parameters (for meta-update)
        
        Returns:
            Updated fast weights
        """
        if fast_weights is None:
            fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner loop optimization
        for step in range(self.inner_steps):
            # Sample batch
            states, actions = zip(*support_set)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            
            # Forward pass with fast weights
            logits = self._forward_with_weights(states, fast_weights)
            loss = F.cross_entropy(logits, actions)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            # Update fast weights
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
        
        return fast_weights
    
    def _forward_with_weights(self, x: torch.Tensor, weights: Dict) -> torch.Tensor:
        """Forward pass with custom weights"""
        # Manual forward pass with provided weights
        x = F.linear(x, weights['net.0.weight'], weights['net.0.bias'])
        x = F.relu(x)
        x = F.linear(x, weights['net.2.weight'], weights['net.2.bias'])
        x = F.relu(x)
        x = F.linear(x, weights['net.4.weight'], weights['net.4.bias'])
        return x
    
    def meta_train_step(self, tasks: List[Task]) -> float:
        """
        Meta-training step: learn to adapt quickly
        
        Args:
            tasks: Batch of tasks
        
        Returns:
            Meta-loss
        """
        meta_loss = 0.0
        
        for task in tasks:
            # Inner loop: adapt to task
            fast_weights = self.inner_loop(task.support_set)
            
            # Outer loop: evaluate on query set
            query_states, query_actions = zip(*task.query_set)
            query_states = torch.FloatTensor(np.array(query_states)).to(self.device)
            query_actions = torch.LongTensor(query_actions).to(self.device)
            
            # Forward with adapted weights
            logits = self._forward_with_weights(query_states, fast_weights)
            loss = F.cross_entropy(logits, query_actions)
            
            meta_loss += loss
        
        # Average over tasks
        meta_loss /= len(tasks)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt(
        self,
        support_set: List[Tuple[np.ndarray, int]],
        num_steps: Optional[int] = None
    ):
        """
        Adapt to a new task (at test time)
        
        Args:
            support_set: Few examples from new task
            num_steps: Number of adaptation steps (defaults to self.inner_steps)
        """
        if num_steps is None:
            num_steps = self.inner_steps
        
        # Clone model for adaptation
        adapted_model = deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        adaptation_losses = []
        
        for step in range(num_steps):
            states, actions = zip(*support_set)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            
            logits = adapted_model(states)
            loss = F.cross_entropy(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            adaptation_losses.append(loss.item())
        
        self.adaptation_curves.append(adaptation_losses)
        
        return adapted_model
    
    def meta_train(
        self,
        train_tasks: List[Task],
        val_tasks: Optional[List[Task]] = None,
        num_iterations: int = 1000,
        tasks_per_batch: int = 4
    ):
        """
        Meta-train the model
        
        Args:
            train_tasks: List of training tasks
            val_tasks: Optional validation tasks
            num_iterations: Number of meta-training iterations
            tasks_per_batch: Number of tasks per meta-batch
        """
        print(f"Meta-training on {len(train_tasks)} tasks...")
        
        for iteration in range(num_iterations):
            # Sample task batch
            task_batch = np.random.choice(train_tasks, size=tasks_per_batch, replace=False)
            
            # Meta-training step
            train_loss = self.meta_train_step(task_batch)
            self.meta_train_losses.append(train_loss)
            
            # Validation
            if val_tasks and (iteration + 1) % 100 == 0:
                val_loss = self._meta_validate(val_tasks)
                self.meta_val_losses.append(val_loss)
                
                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
            elif (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Train Loss: {train_loss:.4f}")
        
        print("✅ Meta-training complete!")
    
    def _meta_validate(self, val_tasks: List[Task]) -> float:
        """Validate on held-out tasks"""
        val_loss = 0.0
        
        with torch.no_grad():
            for task in val_tasks:
                # Adapt on support set
                fast_weights = self.inner_loop(task.support_set)
                
                # Evaluate on query set
                query_states, query_actions = zip(*task.query_set)
                query_states = torch.FloatTensor(np.array(query_states)).to(self.device)
                query_actions = torch.LongTensor(query_actions).to(self.device)
                
                logits = self._forward_with_weights(query_states, fast_weights)
                loss = F.cross_entropy(logits, query_actions)
                
                val_loss += loss.item()
        
        return val_loss / len(val_tasks)
    
    def save(self, path: str):
        """Save meta-model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model': self.model.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'meta_train_losses': self.meta_train_losses,
            'meta_val_losses': self.meta_val_losses
        }, f"{path}/maml_model.pt")
        
        print(f"✅ MAML model saved to {path}")
    
    def load(self, path: str):
        """Load meta-model"""
        checkpoint = torch.load(f"{path}/maml_model.pt", map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.meta_train_losses = checkpoint['meta_train_losses']
        self.meta_val_losses = checkpoint['meta_val_losses']
        
        print(f"✅ MAML model loaded from {path}")


class TransferLearningAgent:
    """
    Transfer learning for compilation optimization
    Pre-train on large corpus, fine-tune on specific projects
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int = 6,
        freeze_base: bool = False,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.base_model = base_model.to(self.device)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Add task-specific head
        if hasattr(base_model, 'net'):
            in_features = list(base_model.net.children())[-1].in_features
        else:
            in_features = 128
        
        self.task_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.base_model.parameters()) + list(self.task_head.parameters()),
            lr=1e-4
        )
        
        print(f"🚀 Transfer Learning Agent initialized")
        print(f"   Base frozen: {freeze_base}")
    
    def fine_tune(
        self,
        train_data: List[Tuple[np.ndarray, int]],
        num_epochs: int = 100,
        batch_size: int = 32
    ):
        """Fine-tune on new task"""
        print(f"Fine-tuning on {len(train_data)} examples...")
        
        for epoch in range(num_epochs):
            # Shuffle data
            np.random.shuffle(train_data)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                states, actions = zip(*batch)
                
                states = torch.FloatTensor(np.array(states)).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                
                # Forward
                features = self.base_model(states)
                logits = self.task_head(features)
                loss = F.cross_entropy(logits, actions)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        print("✅ Fine-tuning complete!")
    
    def predict(self, state: np.ndarray) -> int:
        """Predict action"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.base_model(state)
            logits = self.task_head(features)
            action = torch.argmax(logits, dim=-1)
        
        return action.item()


def generate_synthetic_tasks(num_tasks: int = 100) -> List[Task]:
    """Generate synthetic tasks for meta-learning demo"""
    tasks = []
    
    for i in range(num_tasks):
        # Generate task
        task_bias = np.random.randn(25)
        
        support_set = []
        query_set = []
        
        for j in range(10):  # 10 examples per task
            state = np.random.randn(25) + task_bias * 0.5
            action = np.random.randint(0, 6)
            
            if j < 5:
                support_set.append((state.astype(np.float32), action))
            else:
                query_set.append((state.astype(np.float32), action))
        
        task = Task(
            support_set=support_set,
            query_set=query_set,
            task_id=f"task_{i}",
            metadata={}
        )
        tasks.append(task)
    
    return tasks


if __name__ == "__main__":
    # Demo MAML
    print("=== MAML Demo ===")
    maml_agent = MAMLAgent()
    
    # Generate tasks
    train_tasks = generate_synthetic_tasks(80)
    val_tasks = generate_synthetic_tasks(20)
    
    # Meta-train
    maml_agent.meta_train(train_tasks, val_tasks, num_iterations=500)
    
    # Fast adaptation on new task
    new_task = generate_synthetic_tasks(1)[0]
    adapted_model = maml_agent.adapt(new_task.support_set, num_steps=10)
    
    print(f"✅ Adapted to new task with {len(new_task.support_set)} examples")
