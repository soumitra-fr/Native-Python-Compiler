# 📚 RESOURCES INVENTORY

## Downloaded Resources (1.7 GB Total)

### 1. google-research/ (1.2 GB) ⭐⭐⭐
**Contains**:
- ML for compilers research
- Optimization techniques
- Training datasets
- Research papers code

**Usage**:
- Compiler optimization strategies
- ML model architectures
- Training data for RL agent

**Key Directories**:
```
cd OSR/google-research
find . -name "*compiler*" -o -name "*optimize*" | head -20
```

### 2. compilers/ (306 MB) ⭐⭐⭐
**Contains**:
- Compiler implementations
- Optimization passes
- Code generation examples

**Usage**:
- Reference implementations
- LLVM patterns
- Optimization techniques

### 3. ai-compilers/ (161 MB) ⭐⭐⭐
**Contains**:
- AI-powered compiler research
- Neural network architectures
- Training pipelines

**Usage**:
- GNN implementation ideas
- RL agent architectures
- Training strategies

### 4. type-inference/ (104 MB) ⭐⭐⭐
**Contains**:
- Type inference research
- ML models for types
- Datasets

**Usage**:
- GNN type inference training
- Feature engineering ideas
- Evaluation metrics

### 5. tooling/ (47 MB) ⭐⭐
**Contains**:
- llvmlite (LLVM bindings)
- scalene (profiler)
- austin (profiler)

**Usage**:
- LLVM code generation
- Performance profiling
- Benchmark tools

### 6. dowhy/ (38 MB) ⭐
**Contains**:
- Large Python codebase
- Real-world code examples

**Usage**:
- Test cases for compiler
- Python code patterns
- Validation data

### 7. compiler-gym/ (27 MB) ⭐⭐⭐
**Contains**:
- RL environment for compiler optimization
- Benchmarks
- Training tools

**Usage**:
- Train RL agent
- Benchmark optimization
- Action space definition

### 8. typilus/ (1.1 MB) ⭐⭐⭐
**Contains**:
- Type inference for Python
- Graph-based approach
- Pre-trained models

**Usage**:
- GNN architecture reference
- Type inference baseline
- Evaluation methodology

---

## What We Need Next

### 1. Python Code Datasets (For Type Inference)
**Options**:
- Python150k dataset (150k Python files)
- GitHub Python repos (scrape top 10k)
- PyPI packages source code

**Action**:
```bash
# Try alternative Python150k link
cd OSR
curl -L -o py150k.tar.gz https://github.com/google-research/google-research/tree/master/cubert/code_to_subtokenized_sentences/py150

# Or scrape GitHub
pip install PyGithub
python scripts/download_github_repos.py
```

### 2. Pre-trained Models (For Transfer Learning)
**Options**:
- CodeBERT (Microsoft)
- GraphCodeBERT (Microsoft)
- CodeT5 (Salesforce)

**Action**:
```bash
pip install transformers
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/codebert-base')"
```

### 3. GPU Access (For Training)
**Free Options**:
- Google Colab: 15GB RAM, T4 GPU
- Kaggle: 30hrs/week free GPU
- Paperspace: Limited free tier

**Action**:
- Upload code to Colab
- Run training notebooks
- Download trained models

---

## Installation Requirements

### Core Dependencies
```bash
pip install torch>=2.0.0
pip install torch-geometric
pip install transformers
pip install stable-baselines3
pip install networkx
pip install dgl
pip install scikit-learn
pip install numpy pandas
pip install llvmlite
```

### Compiler Dependencies (Already Have)
```bash
# Already installed
llvmlite
numpy
scikit-learn
```

### New Dependencies Needed
```bash
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
pip install transformers datasets
pip install gym stable-baselines3
pip install wandb  # for experiment tracking
```

---

## Dataset Locations (In OSR)

### Type Inference Training
1. **OSR/typilus/** - Pre-processed type data
2. **OSR/type-inference/** - Type inference research
3. **OSR/google-research/** - Contains type prediction data

### Compiler Optimization Training
1. **OSR/compiler-gym/** - RL environment with benchmarks
2. **OSR/google-research/** - Optimization research
3. **OSR/compilers/** - Compiler implementations

### Python Code Examples
1. **OSR/dowhy/** - Real Python codebase (38 MB)
2. **OSR/tooling/** - Various Python tools
3. **OSR/google-research/** - Research code

---

## Quick Start Commands

### Explore What We Have
```bash
cd /Users/soumitra11/Desktop/Arc_prac-main/Native-Python-Compiler/OSR

# Find type inference data
find . -name "*type*" -type f | grep -E "\.(json|csv|pkl)$" | head -20

# Find compiler benchmarks
find compiler-gym -name "*.py" | grep bench | head -10

# Find Python code samples
find dowhy -name "*.py" | head -20

# Check typilus model
ls -lh typilus/
```

### Setup Training Environment
```bash
cd /Users/soumitra11/Desktop/Arc_prac-main/Native-Python-Compiler

# Install ML dependencies
pip install torch torch-geometric transformers stable-baselines3

# Test CompilerGym
cd OSR/compiler-gym
pip install -e .
python -c "import compiler_gym; print('CompilerGym ready!')"

# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} ready!')"
```

---

## Data Size Estimates

### For Type Inference (GNN)
- **Training samples needed**: 50k-100k functions
- **Graph size average**: 50-200 nodes/function
- **Total graph data**: 5-20GB
- **Model size**: 50-100MB

### For Optimization (RL)
- **Benchmark programs**: 1000+ (in compiler-gym)
- **Episodes needed**: 10k-50k
- **Training time**: 12-24 hours
- **Model size**: 10-50MB

---

## Resource Usage Tracking

### Disk Space
- [x] OSR resources: 1.7 GB
- [ ] Python datasets: 5-10 GB (need to download)
- [ ] Trained models: 200 MB
- [ ] Build artifacts: 2 GB
- **Total needed**: ~10 GB

### Memory Requirements
- GNN training: 8-16 GB RAM
- RL training: 4-8 GB RAM
- Compilation: 2-4 GB RAM
- **Peak**: 16 GB (can use Colab if needed)

### GPU Requirements
- GNN training: Highly recommended (10x faster)
- RL training: Optional (3x faster)
- Inference: Not needed
- **Solution**: Free Colab GPU

---

## Next Steps

### Immediate (Do Now)
1. ✅ Downloaded core resources
2. [ ] Install ML dependencies
3. [ ] Test CompilerGym setup
4. [ ] Explore typilus data

### Short Term (Today)
1. [ ] Download Python code samples
2. [ ] Setup Colab notebook for training
3. [ ] Implement string type
4. [ ] Start GNN architecture

### Medium Term (This Week)
1. [ ] Collect 50k+ Python functions
2. [ ] Train GNN on type inference
3. [ ] Implement list/dict types
4. [ ] Train RL agent

---

**Status**: Resources ready, waiting for go signal to start coding.
**Total Downloaded**: 1.7 GB
**Ready to Use**: ✅
**Cost**: $0
