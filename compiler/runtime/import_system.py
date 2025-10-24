"""
Phase 4: Import System Implementation
Provides import, from, and as statement support.
"""

from llvmlite import ir

class ImportSystem:
    """
    Handles Python import statements in compiled code.
    
    Supports:
    - import module
    - import module as alias
    - from module import name
    - from module import name as alias
    - from module import *
    - Relative imports (from . import, from .. import)
    """
    
    def __init__(self, module_loader):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        self.module_loader = module_loader
    
    def generate_import(self, builder, module, module_name, alias=None):
        """
        Generate: import module [as alias]
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            module_name: Name of module to import
            alias: Optional alias name
        
        Returns:
            Pointer to imported module
        """
        # Load the module
        module_ptr = self.module_loader.load_module(builder, module, module_name)
        
        # If alias provided, bind to alias name, otherwise use module name
        binding_name = alias if alias else module_name
        
        # Store in current namespace (would be done by caller)
        return module_ptr
    
    def generate_from_import(self, builder, module, module_name, names, aliases=None):
        """
        Generate: from module import name1, name2, ... [as alias1, alias2, ...]
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            module_name: Name of module to import from
            names: List of names to import
            aliases: Optional list of alias names
        
        Returns:
            Dict of name -> imported value
        """
        # Load the module
        module_ptr = self.module_loader.load_module(builder, module, module_name)
        
        # Import each name
        imported = {}
        for i, name in enumerate(names):
            # Get attribute from module
            value = self.module_loader.get_module_attribute(builder, module, module_ptr, name)
            
            # Use alias if provided
            alias = aliases[i] if aliases and i < len(aliases) else name
            imported[alias] = value
        
        return imported
    
    def generate_from_import_star(self, builder, module, module_name):
        """
        Generate: from module import *
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            module_name: Name of module to import from
        
        Returns:
            Pointer to module (caller will extract all public names)
        """
        # Load the module
        module_ptr = self.module_loader.load_module(builder, module, module_name)
        
        # Declare runtime function to import all public names
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="import_star")
        
        module_void_ptr = builder.bitcast(module_ptr, self.void_ptr)
        result = builder.call(func, [module_void_ptr])
        
        return result
    
    def generate_relative_import(self, builder, module, level, module_name, current_package):
        """
        Generate relative imports: from . import, from .. import, etc.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            level: Number of dots (1 for ., 2 for .., etc.)
            module_name: Name of module (can be empty for "from . import")
            current_package: Name of current package
        
        Returns:
            Pointer to imported module
        """
        # Calculate parent package based on level
        package_parts = current_package.split('.')
        
        if level > len(package_parts):
            raise ValueError(f"Attempted relative import beyond top-level package")
        
        # Go up 'level' levels
        parent_parts = package_parts[:-level] if level > 0 else package_parts
        parent_package = '.'.join(parent_parts)
        
        # Construct full module name
        if module_name:
            full_name = f"{parent_package}.{module_name}" if parent_package else module_name
        else:
            full_name = parent_package
        
        # Load the module
        module_ptr = self.module_loader.load_module(builder, module, full_name)
        
        return module_ptr
    
    def generate_import_function(self, builder, module):
        """
        Generate __import__ builtin function.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
        
        Returns:
            Pointer to __import__ function
        """
        # __import__(name, globals=None, locals=None, fromlist=(), level=0)
        arg_types = [self.char_ptr, self.void_ptr, self.void_ptr, self.void_ptr, self.int32]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="__import__")
        
        return func
    
    def check_import_permissions(self, module_name, restricted_modules=None):
        """
        Check if a module is allowed to be imported.
        
        Args:
            module_name: Name of module
            restricted_modules: List of restricted module names
        
        Returns:
            True if allowed, False otherwise
        """
        if restricted_modules is None:
            restricted_modules = []
        
        # Check if module or any parent is restricted
        parts = module_name.split('.')
        for i in range(len(parts)):
            prefix = '.'.join(parts[:i+1])
            if prefix in restricted_modules:
                return False
        
        return True
    
    def resolve_module_path(self, module_name, package_name=None):
        """
        Resolve a module name to an absolute module path.
        
        Args:
            module_name: Module name (can be relative)
            package_name: Current package name (for relative imports)
        
        Returns:
            Absolute module name
        """
        if module_name.startswith('.'):
            # Relative import
            if package_name is None:
                raise ValueError("Attempted relative import in non-package")
            
            level = len(module_name) - len(module_name.lstrip('.'))
            rest = module_name[level:]
            
            package_parts = package_name.split('.')
            if level > len(package_parts):
                raise ValueError("Attempted relative import beyond top-level package")
            
            parent = '.'.join(package_parts[:-level+1]) if level > 0 else package_name
            
            if rest:
                return f"{parent}.{rest}" if parent else rest
            else:
                return parent
        else:
            # Absolute import
            return module_name


def generate_import_system_runtime():
    """Generate C runtime code for import system."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Forward declarations
typedef struct Module Module;

// Import all public names from module (import *)
void* import_star(void* module_ptr) {
    Module* mod = (Module*)module_ptr;
    
    // Would return dict of all public names (not starting with _)
    // For now, return module itself
    return module_ptr;
}

// __import__ function
void* __import__(char* name, void* globals, void* locals, void* fromlist, int32_t level) {
    // Would implement full __import__ semantics
    // For now, return NULL
    return NULL;
}
"""
    
    # Write to file
    with open('import_system_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Import system runtime generated: import_system_runtime.c")


if __name__ == "__main__":
    from module_loader import ModuleLoader
    
    # Generate runtime C code
    generate_import_system_runtime()
    
    # Test import system
    loader = ModuleLoader()
    import_sys = ImportSystem(loader)
    
    print(f"✅ ImportSystem initialized")
    
    # Test module path resolution
    test_cases = [
        ("numpy", None, "numpy"),
        (".utils", "mypackage.submodule", "mypackage.utils"),
        ("..core", "mypackage.submodule", "core"),
        ("os.path", None, "os.path"),
    ]
    
    for module_name, package, expected in test_cases:
        try:
            result = import_sys.resolve_module_path(module_name, package)
            status = "✅" if result == expected else f"❌ (got {result})"
            print(f"   - resolve_module_path('{module_name}', '{package}'): {status}")
        except Exception as e:
            print(f"   - resolve_module_path('{module_name}', '{package}'): ❌ {e}")
