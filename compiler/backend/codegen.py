"""
Native Code Generation - Compile LLVM IR to native machine code

This module takes LLVM IR and produces native executable binaries
that can run standalone without Python runtime.

Phase: 1.3 (Backend)
"""

import llvmlite.binding as llvm
import tempfile
import subprocess
import os
from pathlib import Path
from typing import Optional


class NativeCodeGen:
    """
    Compiles LLVM IR to native machine code
    
    Produces standalone executable binaries from LLVM IR.
    """
    
    def __init__(self):
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Get target machine
        target = llvm.Target.from_default_triple()
        self.target_machine = target.create_target_machine()
    
    def compile_to_object(self, llvm_ir: str, output_path: str, optimize: bool = True, opt_level: int = 3):
        """
        Compile LLVM IR to object file
        
        Args:
            llvm_ir: LLVM IR as string
            output_path: Path to output .o file
            optimize: Whether to optimize
            opt_level: Optimization level (0=none, 1=less, 2=default, 3=aggressive)
        """
        # Parse LLVM IR
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        
        # Optimize if requested
        if optimize:
            pmb = llvm.create_pass_manager_builder()
            pmb.opt_level = opt_level  # 0-3: none, less, default, aggressive
            pmb.size_level = 0  # 0=no size optimization, 1=optimize for size
            
            # Enable specific optimizations
            pmb.inlining_threshold = 225  # Inline functions aggressively
            pmb.loop_vectorize = True  # Enable loop vectorization
            pmb.slp_vectorize = True  # Enable superword-level parallelism vectorization
            
            # Create and populate pass manager
            pm = llvm.create_module_pass_manager()
            pmb.populate(pm)
            pm.run(mod)
        
        # Generate object code
        obj_code = self.target_machine.emit_object(mod)
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(obj_code)
    
    def compile_to_assembly(self, llvm_ir: str, output_path: str):
        """
        Compile LLVM IR to assembly (.s file)
        
        Args:
            llvm_ir: LLVM IR as string
            output_path: Path to output .s file
        """
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        
        # Generate assembly
        asm_code = self.target_machine.emit_assembly(mod)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(asm_code)
    
    def link_executable(
        self,
        object_files: list,
        output_path: str,
        runtime_lib: Optional[str] = None
    ):
        """
        Link object files into executable
        
        Args:
            object_files: List of .o file paths
            output_path: Path to output executable
            runtime_lib: Optional runtime library path
        """
        # Build linker command
        # Use clang for linking (handles system libraries automatically)
        cmd = ['clang', '-o', output_path] + object_files
        
        if runtime_lib:
            cmd.append(runtime_lib)
        
        # Add system libraries
        cmd.extend(['-lm'])  # Math library
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Make executable
            os.chmod(output_path, 0o755)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Linking failed: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Error: clang not found. Please install clang.")
            return False
    
    def compile_to_executable(
        self,
        llvm_ir: str,
        output_path: str,
        optimize: bool = True,
        opt_level: int = 3,
        keep_intermediate: bool = False
    ) -> bool:
        """
        Compile LLVM IR directly to executable
        
        Args:
            llvm_ir: LLVM IR as string
            output_path: Path to output executable
            optimize: Whether to optimize
            opt_level: Optimization level (0-3)
            keep_intermediate: Keep .o files
            
        Returns:
            True if successful, False otherwise
        """
        # Create temporary object file
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
            obj_path = tmp.name
        
        try:
            # Compile to object
            self.compile_to_object(llvm_ir, obj_path, optimize, opt_level)
            
            # Link to executable
            success = self.link_executable([obj_path], output_path)
            
            # Clean up unless requested to keep
            if not keep_intermediate and os.path.exists(obj_path):
                os.remove(obj_path)
            
            return success
            
        except Exception as e:
            print(f"Compilation failed: {e}")
            if os.path.exists(obj_path):
                os.remove(obj_path)
            return False


class CompilerPipeline:
    """
    Complete compilation pipeline: Python -> AST -> IR -> LLVM -> Native
    
    Orchestrates the entire compilation process.
    """
    
    def __init__(self):
        from compiler.frontend.parser import Parser
        from compiler.frontend.semantic import analyze
        from compiler.frontend.symbols import SymbolTableBuilder
        from compiler.ir.lowering import IRLowering
        from compiler.backend.llvm_gen import LLVMCodeGen
        
        self.parser = Parser()
        self.symbol_builder = SymbolTableBuilder()
        self.native_gen = NativeCodeGen()
    
    def compile_source(
        self,
        source: str,
        output_path: str,
        optimize: bool = True,
        opt_level: int = 3,
        verbose: bool = False
    ) -> bool:
        """
        Compile Python source to native executable
        
        Args:
            source: Python source code
            output_path: Path to output executable
            optimize: Enable optimizations
            opt_level: Optimization level (0=none, 1=less, 2=default, 3=aggressive)
            verbose: Print compilation stages
            output_path: Path to output executable
            optimize: Enable optimizations
            verbose: Print compilation stages
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Parse
            if verbose:
                print("[ 1/5 ] Parsing...")
            
            parse_result = self.parser.parse_source(source)
            if not parse_result.success:
                print("Parse errors:")
                for error in parse_result.errors:
                    print(f"  Line {error.line}: {error.message}")
                return False
            
            # Step 2: Semantic analysis
            if verbose:
                print("[ 2/5 ] Semantic analysis...")
            
            from compiler.frontend.semantic import analyze
            semantic_result = analyze(parse_result.ast_tree)
            if not semantic_result.success:
                print("Semantic errors:")
                for error in semantic_result.errors:
                    print(f"  Line {error.line}: {error.message}")
                return False
            
            # Step 3: Lower to IR
            if verbose:
                print("[ 3/5 ] Lowering to IR...")
            
            from compiler.ir.lowering import IRLowering
            lowering = IRLowering(self.symbol_builder.global_table)
            ir_module = lowering.visit_Module(parse_result.ast_tree)
            
            if verbose:
                print("\n--- Generated IR ---")
                print(ir_module)
                print("--- End IR ---\n")
            
            # Step 4: Generate LLVM IR
            if verbose:
                print("[ 4/5 ] Generating LLVM IR...")
            
            from compiler.backend.llvm_gen import LLVMCodeGen
            llvm_gen = LLVMCodeGen()
            llvm_ir = llvm_gen.generate_module(ir_module)
            
            if verbose:
                print("\n--- LLVM IR ---")
                print(llvm_ir)
                print("--- End LLVM IR ---\n")
            
            # Step 5: Compile to native
            if verbose:
                opt_levels = ["O0 (no optimization)", "O1 (less)", "O2 (default)", "O3 (aggressive)"]
                opt_desc = opt_levels[opt_level] if 0 <= opt_level <= 3 else f"O{opt_level}"
                print(f"[ 5/5 ] Compiling to native code... ({opt_desc})")
            
            success = self.native_gen.compile_to_executable(
                llvm_ir,
                output_path,
                optimize=optimize,
                opt_level=opt_level
            )
            
            if success and verbose:
                print(f"\n✅ Compiled successfully: {output_path}")
                # Get file size
                size = os.path.getsize(output_path)
                print(f"   Binary size: {size} bytes ({size/1024:.1f} KB)")
                if optimize:
                    print(f"   Optimization: {opt_levels[opt_level] if 0 <= opt_level <= 3 else f'O{opt_level}'}")
            
            return success
            
        except Exception as e:
            print(f"Compilation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compile_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        optimize: bool = True,
        verbose: bool = False
    ) -> bool:
        """
        Compile Python file to native executable
        
        Args:
            input_path: Path to .py file
            output_path: Path to output executable (default: input without .py)
            optimize: Enable optimizations
            verbose: Print compilation stages
            
        Returns:
            True if successful
        """
        # Read source
        with open(input_path, 'r') as f:
            source = f.read()
        
        # Default output path
        if output_path is None:
            output_path = str(Path(input_path).with_suffix(''))
        
        if verbose:
            print(f"Compiling {input_path} -> {output_path}")
            print("=" * 60)
        
        return self.compile_source(source, output_path, optimize, verbose)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("NATIVE CODE GENERATION - Phase 1.3")
    print("=" * 80)
    
    # Create a simple test program
    test_source = """
def add(x: int, y: int) -> int:
    return x + y

def main() -> int:
    result: int = add(10, 32)
    return result
"""
    
    print("\n--- Test Program ---")
    print(test_source)
    print("--- End Program ---\n")
    
    # Compile it
    print("Compiling to native executable...")
    
    pipeline = CompilerPipeline()
    
    # Create output path in temp directory
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as tmp:
        output_path = tmp.name
    
    success = pipeline.compile_source(
        test_source,
        output_path,
        optimize=True,
        verbose=True
    )
    
    if success:
        print(f"\n✅ Compilation successful!")
        print(f"Executable: {output_path}")
        
        # Try to run it
        print("\nTrying to execute...")
        try:
            result = subprocess.run([output_path], capture_output=True, timeout=5)
            print(f"Exit code: {result.returncode}")
            if result.stdout:
                print(f"Output: {result.stdout.decode()}")
            if result.stderr:
                print(f"Errors: {result.stderr.decode()}")
        except Exception as e:
            print(f"Execution failed: {e}")
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    else:
        print("\n❌ Compilation failed!")
    
    print("\n" + "=" * 80)
    print("✅ Phase 1.3 Native Code Generation Complete!")
    print("=" * 80)
    print("\n🎉 Phase 1.3 (Backend) COMPLETE!")
    print("Next: Phase 1.4 - Runtime Library")
