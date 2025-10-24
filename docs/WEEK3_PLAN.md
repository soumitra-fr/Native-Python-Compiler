# Week 3 Implementation Plan: OOP Polish + Import System Foundation

## Overview
Week 3 focuses on:
1. **Days 1-3**: Polish OOP implementation (fix failing tests, complete lowering pipeline)
2. **Days 4-7**: Import system foundation (module loading, caching, basic imports)

**Goal**: Bring project from 65% → 75% complete

---

## Days 1-3: OOP Polish & Completion

### Day 1: Fix Lowering Pipeline for OOP

**Current Issues**:
- IRGetAttr/IRSetAttr not generated in lowering
- Object creation (Call to class) not lowered to IRNewObject
- Attribute access not generating proper IR

**Tasks**:

1. **Implement visit_Attribute() in lowering.py**
```python
def visit_Attribute(self, node: ast.Attribute) -> IRNode:
    """Lower attribute access to IRGetAttr"""
    obj = self.visit(node.value)
    
    # Determine if this is a load or store context
    if isinstance(node.ctx, ast.Load):
        result = self.fresh_var()
        return IRGetAttr(
            object=obj,
            attribute=node.attr,
            result=result
        )
    # Store handled in visit_Assign
    return obj
```

2. **Implement object instantiation in visit_Call()**
```python
def visit_Call(self, node: ast.Call) -> IRNode:
    # Check if calling a class (for instantiation)
    if isinstance(node.func, ast.Name):
        name = node.func.id
        if name in self.symbol_table.classes:
            # Create new object
            args = [self.visit(arg) for arg in node.args]
            result = self.fresh_var()
            return IRNewObject(
                class_name=name,
                args=args,
                result=result
            )
    
    # Regular function call
    return self._lower_function_call(node)
```

3. **Implement visit_Assign() for attribute setting**
```python
def visit_Assign(self, node: ast.Assign) -> List[IRNode]:
    instrs = []
    value = self.visit(node.value)
    
    for target in node.targets:
        if isinstance(target, ast.Attribute):
            # self.x = value
            obj = self.visit(target.value)
            instrs.append(IRSetAttr(
                object=obj,
                attribute=target.attr,
                value=value
            ))
        else:
            # Regular assignment
            instrs.extend(self._lower_regular_assign(target, value))
    
    return instrs
```

**Files to Modify**:
- `compiler/ir/lowering.py` (+100 lines)

**Tests to Fix**: 10 failing OOP tests
**Expected Result**: All 16 OOP tests passing

---

### Day 2: Fix LLVM Context Isolation

**Current Issue**: Struct types persist across tests causing "already defined" errors

**Tasks**:

1. **Create fresh LLVM context per codegen**
```python
class LLVMCodeGen:
    def __init__(self):
        # Create fresh context for each instance
        self.module = ir.Module(name="main_module")
        self.module.triple = llvmlite.binding.get_default_triple()
        
        # Fresh storage
        self.class_types = {}
        self.class_attr_names = {}
        self.variables = {}
        self.functions = {}
```

2. **Update test helper to use isolated contexts**
```python
def compile_code(source):
    """Each test gets fresh compiler instance"""
    tree = ast.parse(source)
    symbol_table = SymbolTable(name="global")
    
    lowering = IRLowering(symbol_table)
    ir_module = lowering.visit_Module(tree)
    
    # Fresh codegen each time
    codegen = LLVMCodeGen()
    llvm_ir = codegen.generate_module(ir_module)
    
    return ir_module, llvm_ir
```

**Files to Modify**:
- `compiler/backend/llvm_gen.py` (minor refactor)
- `tests/integration/test_full_oop.py` (already done)

**Expected Result**: All 16 OOP tests passing without conflicts

---

### Day 3: Add Property Support

**Tasks**:

1. **Add @property decorator support**
```python
class Circle:
    def __init__(self, radius: int):
        self._radius = radius
    
    @property
    def radius(self) -> int:
        return self._radius
    
    @radius.setter
    def radius(self, value: int):
        if value > 0:
            self._radius = value
```

2. **Implementation**:
   - Parse @property decorator in AST
   - Generate getter/setter methods
   - Transform property access to method calls
   - IR nodes: IRProperty, IRPropertyGet, IRPropertySet

**Files to Create/Modify**:
- `compiler/frontend/decorators.py` (new, 150 lines)
- `compiler/ir/lowering.py` (+50 lines)

**Expected Result**: 5 property tests passing

---

## Days 4-7: Import System Foundation

### Day 4: Module Resolution & Loading

**Tasks**:

1. **Create module loader**
```python
# compiler/frontend/module_loader.py
class ModuleLoader:
    def __init__(self):
        self.module_cache = {}
        self.search_paths = sys.path.copy()
    
    def resolve_module(self, name: str) -> Optional[Path]:
        """Find module file in search paths"""
        for path in self.search_paths:
            module_path = Path(path) / f"{name}.py"
            if module_path.exists():
                return module_path
        return None
    
    def load_module(self, name: str) -> IRModule:
        """Load and compile module"""
        if name in self.module_cache:
            return self.module_cache[name]
        
        path = self.resolve_module(name)
        if not path:
            raise ImportError(f"No module named '{name}'")
        
        # Compile module
        source = path.read_text()
        ir_module = compile_source(source, name)
        
        self.module_cache[name] = ir_module
        return ir_module
```

2. **Add import statement support**
```python
# In lowering.py
def visit_Import(self, node: ast.Import) -> List[IRNode]:
    """Handle: import math"""
    instrs = []
    for alias in node.names:
        module = self.module_loader.load_module(alias.name)
        name = alias.asname or alias.name
        self.symbol_table.add_module(name, module)
    return instrs

def visit_ImportFrom(self, node: ast.ImportFrom) -> List[IRNode]:
    """Handle: from math import sqrt"""
    module = self.module_loader.load_module(node.module)
    instrs = []
    for alias in node.names:
        symbol = module.get_symbol(alias.name)
        name = alias.asname or alias.name
        self.symbol_table.add_symbol(name, symbol)
    return instrs
```

**Files to Create**:
- `compiler/frontend/module_loader.py` (250 lines)

**Files to Modify**:
- `compiler/ir/lowering.py` (+80 lines)
- `compiler/frontend/symbols.py` (+50 lines)

**Expected Result**: 4 import tests passing

---

### Day 5: Compiled Module Cache

**Tasks**:

1. **Create .pym file format**
```python
# compiler/module_cache.py
import pickle
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CompiledModule:
    name: str
    version: str
    ir_module: IRModule
    llvm_ir: str
    symbols: Dict[str, Any]
    dependencies: List[str]
    timestamp: float

class ModuleCache:
    def save(self, module: CompiledModule, path: Path):
        """Save compiled module to .pym file"""
        with open(path, 'wb') as f:
            pickle.dump(module, f)
    
    def load(self, path: Path) -> CompiledModule:
        """Load compiled module from .pym file"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def is_stale(self, pym_path: Path, source_path: Path) -> bool:
        """Check if .pym needs recompilation"""
        if not pym_path.exists():
            return True
        
        module = self.load(pym_path)
        source_time = source_path.stat().st_mtime
        
        return module.timestamp < source_time
```

2. **Integrate with module loader**
```python
class ModuleLoader:
    def load_module(self, name: str) -> IRModule:
        """Load from cache or compile"""
        source_path = self.resolve_module(name)
        cache_path = source_path.with_suffix('.pym')
        
        # Check cache
        if cache_path.exists():
            if not self.cache.is_stale(cache_path, source_path):
                return self.cache.load(cache_path).ir_module
        
        # Compile fresh
        ir_module = self._compile_module(source_path)
        
        # Save to cache
        compiled = CompiledModule(
            name=name,
            version="1.0",
            ir_module=ir_module,
            llvm_ir=str(ir_module),
            symbols=ir_module.symbols,
            dependencies=[],
            timestamp=time.time()
        )
        self.cache.save(compiled, cache_path)
        
        return ir_module
```

**Files to Create**:
- `compiler/module_cache.py` (200 lines)

**Expected Result**: Module caching working, 3 cache tests passing

---

### Day 6: Circular Import Detection

**Tasks**:

1. **Add import cycle detection**
```python
class ModuleLoader:
    def __init__(self):
        self.loading_stack = []  # Track import chain
        self.module_cache = {}
    
    def load_module(self, name: str) -> IRModule:
        # Check for cycles
        if name in self.loading_stack:
            cycle = " -> ".join(self.loading_stack + [name])
            raise ImportError(f"Circular import detected: {cycle}")
        
        self.loading_stack.append(name)
        try:
            module = self._do_load(name)
            return module
        finally:
            self.loading_stack.pop()
```

2. **Support forward references**
```python
# Module A
from module_b import ClassB

class ClassA:
    def use_b(self, b: ClassB):
        pass

# Module B
from module_a import ClassA  # Circular!

class ClassB:
    def use_a(self, a: ClassA):
        pass
```

**Files to Modify**:
- `compiler/frontend/module_loader.py` (+50 lines)

**Expected Result**: 3 circular import tests passing

---

### Day 7: Integration & Testing

**Tasks**:

1. **Create comprehensive import tests**
```python
# test_imports.py
def test_simple_import():
    """import math"""
    pass

def test_from_import():
    """from math import sqrt"""
    pass

def test_import_alias():
    """import numpy as np"""
    pass

def test_module_caching():
    """Verify .pym files are created and used"""
    pass

def test_circular_import_detection():
    """Detect and report circular imports"""
    pass

def test_compiled_module_reuse():
    """Load previously compiled modules"""
    pass
```

2. **Fix any remaining bugs**
3. **Performance benchmarks**
4. **Update documentation**

**Files to Create**:
- `tests/integration/test_imports.py` (300 lines, 12 tests)

**Expected Result**: All import tests passing

---

## Week 3 Deliverables

### Code Additions
- `compiler/frontend/module_loader.py` (300 lines)
- `compiler/frontend/decorators.py` (150 lines)
- `compiler/module_cache.py` (200 lines)
- `compiler/ir/lowering.py` (+230 lines)
- `tests/integration/test_imports.py` (300 lines)
- `tests/integration/test_properties.py` (150 lines)

**Total**: ~1,330 lines

### Tests
- OOP tests: 16/16 passing (fix 10 failures)
- Property tests: 5/5 passing (new)
- Import tests: 12/12 passing (new)

**Total**: 33 new/fixed tests

### Features
✅ Complete OOP lowering pipeline
✅ Property decorators (@property)
✅ Module loading and resolution
✅ Compiled module caching (.pym files)
✅ Circular import detection
✅ Import statement support (import, from...import)

### Progress
- Start: 65%
- End: **75%**

---

## Success Criteria

1. ✅ All 95+ tests passing
2. ✅ Can compile and import multi-file projects
3. ✅ .pym caching reduces compilation time
4. ✅ Properties work like standard Python
5. ✅ Circular imports detected and reported
6. ✅ No LLVM context conflicts

---

## Next Steps (Week 4)

Week 4 will focus on:
- Advanced OOP (inheritance vtables, method resolution)
- Module-level optimizations
- Cross-module inlining
- Package support (\_\_init\_\_.py)
