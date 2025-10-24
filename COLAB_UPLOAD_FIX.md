# 🚀 COLAB TRAINING - COPY/PASTE VERSION

If you're having trouble uploading the notebook, follow these steps:

## Method 1: Direct Upload (Recommended)

1. Go to: https://colab.research.google.com/
2. In the top menu: **File → Upload notebook** (NOT the sidebar Upload)
3. Select: training_colab.ipynb
4. Done!

## Method 2: Copy/Paste Content

If upload still fails:

1. Go to: https://colab.research.google.com/
2. File → New notebook
3. Copy content from training_colab.ipynb (see below)
4. Paste into Colab cells

### Cell 1 (Markdown):
```
# 🚀 State-of-the-Art AI Compiler Training

**Free GPU Training** | Tesla T4 | ~2 hours

Upload to Google Colab and run all cells.
```

### Cell 2 (Code):
```python
# Check GPU
!nvidia-smi

import torch
print(f"\n✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 3 (Code):
```python
# Clone repo (UPDATE YOUR_USERNAME!)
!git clone https://github.com/YOUR_USERNAME/Native-Python-Compiler.git
%cd Native-Python-Compiler
```

### Cell 4 (Code):
```python
# Install dependencies
!pip install -q torch transformers scikit-learn numpy pandas tqdm huggingface_hub accelerate sentencepiece protobuf
print("✅ Dependencies installed")
```

### Cell 5 (Code):
```python
# Train Type Inference
from ai.transformer_type_inference import TransformerTypeInferenceEngine
import time

# Generate training data (simplified version)
training_data = []
for i in range(1000):
    patterns = [
        (f"x = {i}", "x", "int"),
        (f"y = {i}.5", "y", "float"),
        (f"s = 'str{i}'", "s", "str"),
        (f"lst = [{i}, {i+1}]", "lst", "list"),
    ]
    pattern = patterns[i % len(patterns)]
    training_data.append({'code': pattern[0], 'variable': pattern[1], 'type': pattern[2]})

val_data = training_data[:200]

print(f"Training Type Inference...")
engine = TransformerTypeInferenceEngine(device="cuda")

start = time.time()
engine.train(
    training_data=training_data,
    validation_data=val_data,
    epochs=10,
    batch_size=32,
    learning_rate=2e-5,
    output_dir="./ai/models/type_inference_gpu"
)
print(f"✅ Done in {(time.time() - start)/60:.1f} min")
```

### Cell 6 (Code):
```python
# Train DQN
from ai.deep_rl_strategy import DeepRLStrategyAgent

print("Training DQN...")
dqn = DeepRLStrategyAgent(device="cuda", batch_size=64)

start = time.time()
dqn.train(training_episodes=1000, eval_freq=100)
dqn.save("./ai/models/dqn_gpu")
print(f"✅ Done in {(time.time() - start)/60:.1f} min")
```

### Cell 7 (Code):
```python
# Train PPO
from ai.ppo_agent import PPOAgent

print("Training PPO...")
ppo = PPOAgent(device="cuda", batch_size=64)

start = time.time()
ppo.train(num_episodes=1000, steps_per_episode=100)
ppo.save("./ai/models/ppo_gpu")
print(f"✅ Done in {(time.time() - start)/60:.1f} min")
```

### Cell 8 (Code):
```python
# Train MAML
from ai.meta_learning import MAMLAgent, generate_synthetic_tasks

print("Training MAML...")
maml = MAMLAgent(device="cuda")
train_tasks = generate_synthetic_tasks(100)
val_tasks = generate_synthetic_tasks(20)

start = time.time()
maml.meta_train(train_tasks, val_tasks, num_iterations=500, tasks_per_batch=8)
maml.save("./ai/models/maml_gpu")
print(f"✅ Done in {(time.time() - start)/60:.1f} min")
```

### Cell 9 (Code):
```python
# Package and download
!zip -r trained_models.zip ai/models/
print("\n✅ All models trained and packaged!")
print("\n📥 Download trained_models.zip from Files panel")

from google.colab import files
files.download('trained_models.zip')
```

## Method 3: Use GitHub (Cleanest)

1. Push your repo to GitHub:
```bash
cd /Users/soumitra11/Desktop/Arc_prac-main/Native-Python-Compiler
git add training_colab.ipynb
git commit -m "Add training notebook"
git push
```

2. In Colab:
   - File → Open notebook
   - Select "GitHub" tab
   - Enter your repo URL
   - Select training_colab.ipynb

## Why Upload Failed

The error "Unable to read file" happens because:
- You clicked "Upload" in the "Open notebook" dialog
- That's for opening existing Colab files from Drive/GitHub
- Not for uploading from local computer

**Solution**: Use **File menu → Upload notebook** instead!
