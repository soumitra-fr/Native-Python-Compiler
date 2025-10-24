# Open Source Research (OSR)

This directory contains curated open-source projects that serve as references, inspiration, and learning resources for building the AI Agentic Python Compiler.

---

## **Directory Structure**

```
OSR/
├── compilers/          # Existing Python compilers
├── ai-compilers/       # AI/ML-based compiler research
├── type-inference/     # Type inference and checking tools
├── tooling/            # LLVM bindings and profiling tools
└── papers/             # Research papers and documentation
```

---

## **1. Compilers (`compilers/`)**

Existing Python compiler implementations to study and learn from.

### **PyPy** 
**Repository**: `pypy/`  
**URL**: https://github.com/pypy/pypy  
**Language**: Python (RPython)  
**License**: MIT

**What it is**:
- Alternative Python interpreter with JIT (Just-In-Time) compiler
- 4-7x faster than CPython on average
- Uses tracing JIT compilation

**What to study**:
- `/pypy/jit/metainterp/` - Tracing JIT implementation
- `/rpython/` - RPython toolchain (compiler for Python subset)
- `/rpython/rtyper/` - Type inference system
- `/pypy/module/` - How standard library is implemented

**Key learnings**:
- How to do tracing JIT compilation
- RPython type system design
- Optimization strategies for dynamic languages
- Runtime design patterns

**Relevance to our project**: 
PyPy demonstrates that aggressive JIT compilation can achieve significant speedups. We'll borrow ideas on trace selection, guard insertion, and deoptimization.

---

### **Numba**
**Repository**: `numba/`  
**URL**: https://github.com/numba/numba  
**Language**: Python  
**License**: BSD 2-Clause

**What it is**:
- JIT compiler for Python using LLVM
- Specializes in numerical and array-oriented code
- Decorator-based: `@jit` for optimization

**What to study**:
- `/numba/core/typing/` - Type inference engine
- `/numba/core/ir.py` - Intermediate representation
- `/numba/targets/` - Code generation for different targets
- `/numba/core/compiler.py` - Compilation pipeline

**Key learnings**:
- How to use llvmlite effectively
- Type inference from bytecode
- Function specialization strategies
- When to compile vs fallback

**Relevance to our project**:
Numba is the closest existing project to our goals. We'll leverage llvmlite similarly and study their type inference, but add AI agents for smarter decisions.

---

### **Cinder**
**Repository**: `cinder/`  
**URL**: https://github.com/facebookincubator/cinder  
**Language**: C++  
**License**: Python Software Foundation License

**What it is**:
- Instagram's performance-oriented CPython fork
- JIT compiler, immortal objects, strict modules
- Production-ready (runs Instagram.com)

**What to study**:
- `/Jit/` - JIT compiler implementation
- `/StrictModules/` - Static Python subset
- `/RuntimeTests/` - Performance tests
- `/CinderDoc/` - Architecture documentation

**Key learnings**:
- Production JIT design
- Gradual typing enforcement
- Integration with existing CPython
- Real-world performance optimization

**Relevance to our project**:
Cinder shows a practical approach to speeding up Python in production. Their "strict modules" align with our compilable subset strategy.

---

### **Pyjion**
**Repository**: `pyjion/`  
**URL**: https://github.com/tonybaloney/pyjion  
**Language**: C++  
**License**: MIT

**What it is**:
- JIT compiler for CPython using .NET's CoreCLR
- Transparent integration (no code changes needed)
- Cross-platform

**What to study**:
- `/src/pyjion/` - JIT integration with CPython
- `/src/pyjion/absint.cpp` - Abstract interpretation
- `/Tests/` - Test cases for JIT behavior

**Key learnings**:
- How to plug into CPython's eval loop
- Type specialization without annotations
- Performance profiling strategies

**Relevance to our project**:
Pyjion demonstrates seamless JIT integration. We may want optional transparent mode in later phases.

---

## **2. AI Compilers (`ai-compilers/`)**

Research projects applying ML/AI to compiler optimization.

### **CompilerGym**
**Repository**: `CompilerGym/`  
**URL**: https://github.com/facebookresearch/CompilerGym  
**Language**: Python, C++  
**License**: MIT

**What it is**:
- OpenAI Gym-style RL environment for compiler optimization
- Provides LLVM optimization pass selection as RL problem
- Benchmarks and baseline models included

**What to study**:
- `/compiler_gym/envs/llvm/` - LLVM RL environment
- `/examples/` - RL agents for optimization
- `/compiler_gym/service/` - Communication protocol

**Key learnings**:
- How to formulate compiler optimization as RL problem
- State/action space design
- Reward function engineering
- Benchmark methodology

**Relevance to our project**:
**CRITICAL RESOURCE**. We'll use CompilerGym to train our optimization agent. Provides ready-made environment and benchmarks.

---

### **MLGO (ML-Guided Compiler Optimizations)**
**Repository**: `ml-compiler-opt/`  
**URL**: https://github.com/google/ml-compiler-opt  
**Language**: Python, C++  
**License**: Apache 2.0

**What it is**:
- Google's ML infrastructure for LLVM
- RL for inlining decisions
- Production deployment at Google scale

**What to study**:
- `/compiler_opt/rl/` - RL training pipeline
- `/compiler_opt/rl/inlining/` - Inlining agent
- `/docs/` - Architecture and design docs

**Key learnings**:
- Production ML for compilers
- Inlining heuristics as RL
- Large-scale training infrastructure
- Deployment strategies

**Relevance to our project**:
MLGO proves that RL-based optimization works at scale. We'll adapt their approaches for our optimization agent.

---

### **TVM (Tensor Virtual Machine)**
**Repository**: `tvm/`  
**URL**: https://github.com/apache/tvm  
**Language**: C++, Python  
**License**: Apache 2.0

**What it is**:
- End-to-end ML compiler stack
- Automatic optimization for diverse hardware
- AutoTVM: ML-based autotuning

**What to study**:
- `/python/tvm/auto_scheduler/` - Automatic scheduling
- `/src/relay/` - High-level IR for ML
- `/python/tvm/autotvm/` - Tuning framework

**Key learnings**:
- Multi-level IR design
- Hardware-aware optimization
- Search-based optimization
- Performance modeling

**Relevance to our project**:
TVM's auto-scheduling demonstrates AI-guided code generation. We'll adapt ideas for general Python, not just ML models.

---

### **Halide**
**Repository**: `Halide/`  
**URL**: https://github.com/halide/Halide  
**Language**: C++  
**License**: MIT

**What it is**:
- DSL for image processing and numerical code
- Separates algorithm from schedule
- Auto-tuning framework

**What to study**:
- `/src/Func.cpp` - Algorithm/schedule separation
- `/src/AutoSchedule.cpp` - Automatic scheduling
- `/apps/` - Real-world applications

**Key learnings**:
- Schedule space exploration
- Auto-tuning methodologies
- Performance models

**Relevance to our project**:
Halide's auto-scheduler uses learned cost models. Similar ideas can guide our loop optimization agent.

---

## **3. Type Inference (`type-inference/`)**

Tools for Python type inference and checking.

### **Pyright**
**Repository**: `pyright/`  
**URL**: https://github.com/microsoft/pyright  
**Language**: TypeScript  
**License**: MIT

**What it is**:
- Fast Python type checker
- Used by VS Code Pylance
- Excellent type inference

**What to study**:
- `/packages/pyright-internal/src/analyzer/` - Type inference engine
- `/packages/pyright-internal/src/analyzer/typeEvaluator.ts` - Type evaluation
- `/packages/pyright-internal/src/analyzer/symbolUtils.ts` - Symbol resolution

**Key learnings**:
- Type inference algorithms
- Handling gradual typing
- Error reporting
- IDE integration

**Relevance to our project**:
Pyright's inference engine is state-of-the-art. We'll study their algorithms and enhance with runtime traces + ML.

---

### **Pyre**
**Repository**: `pyre-check/`  
**URL**: https://github.com/facebook/pyre-check  
**Language**: OCaml  
**License**: MIT

**What it is**:
- Performant type checker for Python
- Used at Meta (Facebook/Instagram)
- Incremental analysis

**What to study**:
- `/source/analysis/` - Type inference and checking
- `/source/interprocedural/` - Call graph analysis
- `/documentation/` - Design docs

**Key learnings**:
- Scalable type analysis
- Incremental checking strategies
- Integration with CI/CD

**Relevance to our project**:
Pyre handles massive codebases efficiently. We'll borrow ideas for scalable analysis.

---

### **MonkeyType**
**Repository**: `MonkeyType/`  
**URL**: https://github.com/Instagram/MonkeyType  
**Language**: Python  
**License**: BSD 3-Clause

**What it is**:
- Runtime type collector
- Generates type stubs from execution traces
- Used at Instagram

**What to study**:
- `/monkeytype/tracing.py` - Trace collection
- `/monkeytype/typing.py` - Type inference from traces
- `/monkeytype/db/` - Trace storage

**Key learnings**:
- Low-overhead tracing
- Type inference from runtime data
- Stub generation

**Relevance to our project**:
**CRITICAL RESOURCE**. MonkeyType's tracing approach is exactly what we need for Phase 2.1. We'll adapt and extend it.

---

## **4. Tooling (`tooling/`)**

LLVM bindings and profiling tools.

### **llvmlite**
**Repository**: `llvmlite/`  
**URL**: https://github.com/numba/llvmlite  
**Language**: Python, C++  
**License**: BSD 2-Clause

**What it is**:
- Lightweight LLVM Python binding
- Used by Numba
- Focuses on code generation

**What to study**:
- `/llvmlite/ir/` - IR builder API
- `/llvmlite/binding/` - LLVM C API wrappers
- `/examples/` - Usage examples

**Key learnings**:
- LLVM IR construction from Python
- Module, function, basic block structure
- Optimization pass management
- JIT execution

**Relevance to our project**:
**PRIMARY TOOL**. llvmlite is our main LLVM interface. We'll use it extensively in Phase 1.3.

---

### **py-spy**
**Repository**: `py-spy/`  
**URL**: https://github.com/benfred/py-spy  
**Language**: Rust  
**License**: MIT

**What it is**:
- Sampling profiler for Python
- No code changes needed
- Low overhead (~2-5%)

**What to study**:
- `/src/python_spy/` - Python internals reading
- `/src/sampler.rs` - Sampling mechanism
- `/src/flamegraph.rs` - Visualization

**Key learnings**:
- Low-overhead profiling techniques
- Sampling strategies
- Stack unwinding

**Relevance to our project**:
We'll use py-spy in Phase 0 and Phase 2.1 for hot path detection.

---

### **scalene**
**Repository**: `scalene/`  
**URL**: https://github.com/plasma-umass/scalene  
**Language**: Python, C++  
**License**: Apache 2.0

**What it is**:
- High-performance CPU + GPU + memory profiler
- Line-level profiling
- Identifies optimization opportunities

**What to study**:
- `/scalene/` - Profiler implementation
- `/scalene/adaptive_sampling.py` - Adaptive sampling
- `/scalene/replacement_malloc.cpp` - Memory profiling

**Key learnings**:
- Multi-dimensional profiling (CPU, GPU, memory)
- Adaptive sampling
- Integration with Python runtime

**Relevance to our project**:
scalene's adaptive sampling could improve our tracing efficiency.

---

### **austin**
**Repository**: `austin/`  
**URL**: https://github.com/P403n1x87/austin  
**Language**: C  
**License**: GPL 3.0

**What it is**:
- Frame stack sampler for CPython
- No instrumentation needed
- Ultra low overhead

**What to study**:
- `/src/py_*.c` - Python version-specific code
- `/src/austin.c` - Main profiling logic

**Key learnings**:
- CPython internals
- Cross-platform profiling
- Minimal overhead techniques

**Relevance to our project**:
austin demonstrates extreme low-overhead profiling. Useful for production monitoring.

---

## **5. Papers (`papers/`)**

Key research papers to read (PDFs to be added):

### **Must-Read Papers**

1. **"Tracing the Meta-Level: PyPy's Tracing JIT Compiler"** (2009)
   - Foundation of PyPy's approach
   
2. **"TypeWriter: Neural Type Prediction with Search-based Validation"** (DeepMind, 2019)
   - ML for type inference
   
3. **"The Case for Learned Index Structures"** (Kraska et al., 2017)
   - ML replacing algorithms (inspiration)
   
4. **"Learning to Optimize Halide with Tree Search"** (Adams et al., 2019)
   - RL for compiler optimization
   
5. **"CompilerGym: Robust, Performant Compiler Optimization Environments"** (Meta, 2021)
   - RL environments for compilers

6. **"MLGO: a Machine Learning Guided Compiler Optimizations Framework"** (Google, 2021)
   - Production ML in LLVM

### **Download Links**
```bash
# TODO: Add paper download scripts
cd papers/
wget <paper_urls>
```

---

## **How to Use This Repository**

### **For Learning**

**Week 1-2: Understand Existing Compilers**
```bash
# Study PyPy's tracing JIT
cd compilers/pypy/pypy/jit/metainterp/
# Read: pyjitpl.py, optimize.py

# Study Numba's architecture
cd ../../../numba/numba/core/
# Read: compiler.py, typing/typeof.py
```

**Week 3-4: Learn LLVM Basics**
```bash
cd tooling/llvmlite/examples/
python -m pip install llvmlite
python simple_module.py  # Run examples
```

**Week 5-6: Explore AI for Compilers**
```bash
cd ai-compilers/CompilerGym/
pip install -e .
python examples/example_unrolling_service.py
```

### **For Development**

**Phase 0: Reference Numba + py-spy**
```bash
# Study Numba's JIT decorator
cd compilers/numba/numba/core/decorators.py

# Understand py-spy profiling
cd tooling/py-spy/
cargo build --release
```

**Phase 1: Reference llvmlite + Numba IR**
```bash
# LLVM IR generation
cd tooling/llvmlite/llvmlite/ir/

# Numba's IR design
cd ../../compilers/numba/numba/core/ir.py
```

**Phase 2: Reference MonkeyType + CompilerGym**
```bash
# Runtime tracing
cd type-inference/MonkeyType/monkeytype/

# RL environment setup
cd ../../ai-compilers/CompilerGym/
```

---

## **Build Instructions**

Some projects require compilation. Here's how to build them:

### **PyPy**
```bash
cd compilers/pypy/
python rpython/bin/rpython -Ojit pypy/goal/targetpypystandalone.py
# Warning: This takes hours! Only do if you want to modify PyPy
```

### **llvmlite**
```bash
cd tooling/llvmlite/
python setup.py build
python -m pip install -e .
```

### **CompilerGym**
```bash
cd ai-compilers/CompilerGym/
pip install cmake
pip install -e .
```

### **py-spy**
```bash
cd tooling/py-spy/
cargo build --release
./target/release/py-spy --help
```

---

## **Quick Reference Cheat Sheet**

| **Need to learn...** | **Check...** |
|----------------------|--------------|
| Tracing JIT | PyPy `/pypy/jit/` |
| Type inference | Pyright `/analyzer/`, MonkeyType |
| LLVM IR generation | llvmlite `/ir/`, Numba `/targets/` |
| RL for optimization | CompilerGym, MLGO |
| Profiling | py-spy, scalene |
| Hardware-aware opt | TVM `/auto_scheduler/` |
| Production JIT | Cinder `/Jit/` |

---

## **Contributing to OSR**

Found a useful resource? Add it!

1. Clone the repository to `OSR/<category>/`
2. Update this README with:
   - What it is
   - What to study
   - Key learnings
   - Relevance to our project
3. Add any build instructions

---

## **Resource Statistics**

- **Total Projects**: 12
- **Total Stars**: 100,000+ (combined)
- **Languages**: Python, C++, Rust, OCaml, TypeScript
- **Lines of Code**: ~5 million (combined)

---

## **Next Steps**

1. **Read** the key files mentioned in each section
2. **Run** the examples to understand behavior
3. **Experiment** with modifications
4. **Apply** learnings to our compiler implementation

---

**Remember**: We're standing on the shoulders of giants. These projects represent decades of compiler research and engineering. Learn from them, respect their licenses, and build something even better. 🚀
