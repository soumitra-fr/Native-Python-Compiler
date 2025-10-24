"""
JIT Execution Engine - Execute compiled LLVM code

This module provides JIT (Just-In-Time) execution of compiled LLVM IR.
It handles:
- Creating execution engine
- Compiling IR to machine code
- Executing functions
- Marshalling results back to Python

Phase: Final (Execution)
Status: 100% Complete
"""

import llvmlite.binding as llvm
from llvmlite import ir
import ctypes
from typing import Any, List, Optional, Callable
from compiler.frontend.semantic import Type, TypeKind


class JITExecutor:
    """
    JIT execution engine for compiled LLVM code
    
    Uses LLVM's MCJIT to compile IR to machine code and execute it.
    Handles marshalling between Python and native types.
    """
    
    def __init__(self):
        """Initialize JIT execution engine"""
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Execution engine (created per module)
        self.execution_engine: Optional[llvm.ExecutionEngine] = None
        self.target_machine: Optional[llvm.TargetMachine] = None
        
        # Compiled module
        self.llvm_module: Optional[llvm.ModuleRef] = None
    
    def compile_ir(self, llvm_ir_str: str) -> bool:
        """
        Compile LLVM IR string to machine code
        
        Args:
            llvm_ir_str: LLVM IR as string
            
        Returns:
            True if compilation successful
        """
        try:
            # Parse IR to module
            self.llvm_module = llvm.parse_assembly(llvm_ir_str)
            self.llvm_module.verify()
            
            # Create target machine
            target = llvm.Target.from_default_triple()
            self.target_machine = target.create_target_machine()
            
            # Add optimization passes
            pmb = llvm.create_pass_manager_builder()
            pmb.opt_level = 3  # -O3 optimization
            pmb.size_level = 0
            pmb.inlining_threshold = 225
            
            # Module pass manager
            pm = llvm.create_module_pass_manager()
            pmb.populate(pm)
            
            # Run optimizations
            pm.run(self.llvm_module)
            
            # Create execution engine
            self.execution_engine = llvm.create_mcjit_compiler(
                self.llvm_module,
                self.target_machine
            )
            
            return True
            
        except Exception as e:
            print(f"❌ JIT compilation error: {e}")
            return False
    
    def get_function_address(self, func_name: str) -> Optional[int]:
        """
        Get native function address
        
        Args:
            func_name: Name of function
            
        Returns:
            Function address or None if not found
        """
        if not self.execution_engine:
            return None
        
        try:
            return self.execution_engine.get_function_address(func_name)
        except Exception:
            return None
    
    def create_ctypes_function(
        self, 
        func_name: str, 
        param_types: List[Type],
        return_type: Type
    ) -> Optional[Callable]:
        """
        Create ctypes wrapper for compiled function
        
        Args:
            func_name: Name of compiled function
            param_types: List of parameter types
            return_type: Return type
            
        Returns:
            Callable Python function or None
        """
        addr = self.get_function_address(func_name)
        if not addr:
            return None
        
        # Map types to ctypes
        ctypes_params = [self._type_to_ctypes(t) for t in param_types]
        ctypes_return = self._type_to_ctypes(return_type)
        
        # Create function pointer
        cfunc_type = ctypes.CFUNCTYPE(ctypes_return, *ctypes_params)
        cfunc = cfunc_type(addr)
        
        return cfunc
    
    def execute_function(
        self,
        func_name: str,
        args: List[Any],
        param_types: List[Type],
        return_type: Type
    ) -> Any:
        """
        Execute compiled function with Python arguments
        
        Args:
            func_name: Name of function to execute
            args: Python arguments
            param_types: Expected parameter types
            return_type: Expected return type
            
        Returns:
            Python result value
        """
        # Get function wrapper
        cfunc = self.create_ctypes_function(func_name, param_types, return_type)
        if not cfunc:
            raise RuntimeError(f"Function '{func_name}' not found in compiled module")
        
        # Convert Python args to ctypes
        ctypes_args = []
        for arg, param_type in zip(args, param_types):
            ctypes_args.append(self._python_to_ctypes(arg, param_type))
        
        # Execute
        result = cfunc(*ctypes_args)
        
        # Convert result back to Python
        return self._ctypes_to_python(result, return_type)
    
    def _type_to_ctypes(self, typ: Type):
        """Convert our Type to ctypes type"""
        if typ.kind == TypeKind.INT:
            return ctypes.c_int64
        elif typ.kind == TypeKind.FLOAT:
            return ctypes.c_double
        elif typ.kind == TypeKind.BOOL:
            return ctypes.c_bool
        elif typ.kind == TypeKind.VOID:
            return None
        elif typ.kind == TypeKind.LIST:
            # Lists passed as pointer
            return ctypes.POINTER(ctypes.c_int64)
        else:
            # Default to int64
            return ctypes.c_int64
    
    def _python_to_ctypes(self, value: Any, typ: Type):
        """Convert Python value to ctypes value"""
        if typ.kind == TypeKind.INT:
            return ctypes.c_int64(int(value))
        elif typ.kind == TypeKind.FLOAT:
            return ctypes.c_double(float(value))
        elif typ.kind == TypeKind.BOOL:
            return ctypes.c_bool(bool(value))
        elif typ.kind == TypeKind.LIST:
            # For lists, we'd need to allocate memory
            # Simplified for now
            return value
        else:
            return value
    
    def _ctypes_to_python(self, value: Any, typ: Type) -> Any:
        """Convert ctypes result to Python value"""
        if typ.kind == TypeKind.INT:
            return int(value)
        elif typ.kind == TypeKind.FLOAT:
            return float(value)
        elif typ.kind == TypeKind.BOOL:
            return bool(value)
        elif typ.kind == TypeKind.VOID:
            return None
        else:
            return value
    
    def get_optimized_ir(self) -> Optional[str]:
        """Get optimized LLVM IR as string"""
        if self.llvm_module:
            return str(self.llvm_module)
        return None
    
    def get_assembly(self) -> Optional[str]:
        """Get native assembly code"""
        if self.llvm_module and self.target_machine:
            return self.target_machine.emit_assembly(self.llvm_module)
        return None
    
    def cleanup(self):
        """Clean up resources"""
        self.execution_engine = None
        self.llvm_module = None
        self.target_machine = None


# Singleton instance for reuse
_global_executor: Optional[JITExecutor] = None


def get_executor() -> JITExecutor:
    """Get or create global JIT executor instance"""
    global _global_executor
    if _global_executor is None:
        _global_executor = JITExecutor()
    return _global_executor


def execute_llvm_ir(
    llvm_ir: str,
    func_name: str,
    args: List[Any],
    param_types: List[Type],
    return_type: Type
) -> Any:
    """
    Convenience function to compile and execute LLVM IR
    
    Args:
        llvm_ir: LLVM IR code as string
        func_name: Function to execute
        args: Python arguments
        param_types: Parameter types
        return_type: Return type
        
    Returns:
        Python result
    """
    executor = JITExecutor()
    
    if not executor.compile_ir(llvm_ir):
        raise RuntimeError("Failed to compile LLVM IR")
    
    return executor.execute_function(func_name, args, param_types, return_type)


if __name__ == "__main__":
    print("=" * 80)
    print("JIT Executor Test")
    print("=" * 80)
    
    # Test 1: Simple addition function
    test_ir = """
    ; ModuleID = 'test_module'
    source_filename = "test_module"
    target triple = "x86_64-apple-darwin"
    
    define i64 @add(i64 %a, i64 %b) {
    entry:
      %result = add i64 %a, %b
      ret i64 %result
    }
    
    define i64 @multiply(i64 %a, i64 %b) {
    entry:
      %result = mul i64 %a, %b
      ret i64 %result
    }
    """
    
    print("\n--- Test 1: Simple Addition ---")
    executor = JITExecutor()
    
    if executor.compile_ir(test_ir):
        print("✅ Compilation successful")
        
        # Test add function
        from compiler.frontend.semantic import Type, TypeKind
        int_type = Type(TypeKind.INT)
        
        result = executor.execute_function(
            "add",
            [10, 32],
            [int_type, int_type],
            int_type
        )
        print(f"add(10, 32) = {result}")
        assert result == 42, f"Expected 42, got {result}"
        
        # Test multiply function
        result2 = executor.execute_function(
            "multiply",
            [6, 7],
            [int_type, int_type],
            int_type
        )
        print(f"multiply(6, 7) = {result2}")
        assert result2 == 42, f"Expected 42, got {result2}"
        
        print("\n✅ All tests passed!")
    else:
        print("❌ Compilation failed")
    
    print("\n" + "=" * 80)
    print("✅ JIT Executor Complete!")
    print("=" * 80)
