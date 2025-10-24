# Quick Start Guide - AI Agentic Python Compiler

## Setup (5 minutes)

### 1. Create Virtual Environment
```bash
cd /Users/soumitra11/Desktop/Arc_prac-main/Native-Python-Compiler
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import numba; import sklearn; print('✅ All dependencies installed!')"
```

## Run Phase 0 Demo (2 minutes)

### Quick Demo
```bash
python examples/phase0_demo.py
```

This will:
- ✅ Train ML compilation decider
- ✅ Test on numeric workloads  
- ✅ Show 10-50x speedups
- ✅ Demonstrate AI-guided decisions

### Expected Output
```
🤖 ML Decision      matrix_multiply: COMPILE (confidence: 95%)
✅ Compilation      matrix_multiply: SUCCESS
⚡ Speedup          45.23x

🤖 ML Decision      with_print: SKIP (confidence: 92%)
❌ Compilation      with_print: FAILED - unsupported features
```

## Test Individual Components

### 1. Hot Function Detector
```bash
python tools/profiler/hot_function_detector.py
```

### 2. Numba Compiler
```bash
python tools/profiler/numba_compiler.py
```

### 3. ML Decider
```bash
python ai/strategy/ml_decider.py
```

## Use in Your Code

```python
from tools.profiler.hot_function_detector import profile
from tools.profiler.numba_compiler import auto_compile

# Profile to detect hot functions
@profile
def my_function(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

# Auto-compile if beneficial
@auto_compile
def numeric_loop(n):
    result = 0
    for i in range(n):
        for j in range(n):
            result += i * j
    return result

# Run your code
for _ in range(100):
    my_function(1000)
    numeric_loop(100)

# Print profiling report
from tools.profiler.hot_function_detector import print_report
print_report()
```

## Next Steps

### Phase 1: Core Compiler (Starting Now!)

**Goal**: Build Python → Native compiler

**Files being created**:
- `compiler/frontend/parser.py` - AST parser
- `compiler/frontend/semantic.py` - Semantic analysis  
- `compiler/ir/ir_nodes.py` - Intermediate representation
- `compiler/backend/llvm_gen.py` - LLVM code generation
- `compiler/backend/codegen.py` - Native code compilation

**Timeline**: ~12 weeks

### How to Follow Progress

Watch the todo list:
```bash
# Check current status
git log --oneline
```

Or open this project in VS Code to see progress.

## Troubleshooting

### "Numba not found"
```bash
pip install numba llvmlite
```

### "sklearn not found"  
```bash
pip install scikit-learn
```

### Permission errors on macOS
```bash
# If you see permission errors with pip:
pip install --user -r requirements.txt
```

## Performance Tips

### For Best Results:
1. Use type hints where possible
2. Use NumPy arrays for numeric data
3. Avoid I/O in hot loops
4. Profile first, then optimize

## Questions?

- Check `TIMELINE.md` for detailed roadmap
- Check `OSR/README.md` for reference projects
- Check `SETUP_COMPLETE.md` for FAQ

---

**Ready to build the future of Python performance! 🚀**
