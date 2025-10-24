"""
Phase 3: Inheritance Implementation
Provides inheritance, MRO (Method Resolution Order), and super() support.
"""

from llvmlite import ir

class InheritanceType:
    """
    Handles class inheritance and method resolution order (MRO).
    
    Python uses C3 linearization for MRO.
    This implementation provides basic inheritance and super() support.
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
    
    def compute_mro(self, class_name, bases, all_classes):
        """
        Compute Method Resolution Order using C3 linearization.
        
        Args:
            class_name: Name of the class
            bases: List of base class names
            all_classes: Dict mapping class names to their bases
        
        Returns:
            List of class names in MRO order
        """
        if not bases:
            return [class_name]
        
        # Simplified MRO for single inheritance
        if len(bases) == 1:
            base_mro = self.compute_mro(bases[0], all_classes.get(bases[0], []), all_classes)
            return [class_name] + base_mro
        
        # For multiple inheritance, use simplified C3 linearization
        mro = [class_name]
        base_mros = [self.compute_mro(base, all_classes.get(base, []), all_classes) for base in bases]
        
        # Merge base MROs
        while any(base_mros):
            for mro_list in base_mros:
                if not mro_list:
                    continue
                
                candidate = mro_list[0]
                
                # Check if candidate appears in tail of other lists
                if not any(candidate in other[1:] for other in base_mros if other):
                    mro.append(candidate)
                    # Remove candidate from all lists
                    for other in base_mros:
                        if other and other[0] == candidate:
                            other.pop(0)
                    break
            else:
                # No valid candidate found - inconsistent hierarchy
                break
        
        return mro
    
    def generate_mro_lookup(self, builder, module, class_ptr, mro_order):
        """
        Generate LLVM IR for MRO-based method lookup.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            class_ptr: Pointer to Class structure
            mro_order: List of class names in MRO order
        
        Returns:
            Function that looks up methods using MRO
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="mro_lookup_method")
        
        return func
    
    def generate_super_call(self, builder, module, instance_ptr, current_class, method_name, args):
        """
        Generate LLVM IR for super() method calls.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
            current_class: Current class in MRO
            method_name: Name of method to call
            args: List of arguments
        
        Returns:
            Result of super() method call
        """
        # Declare runtime super function
        arg_types = [self.void_ptr, self.void_ptr, self.char_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="super_call_method")
        
        # Create method name string
        name_str = self._create_string_literal(builder, module, method_name)
        
        # Create args array
        args_array = self._create_pointer_array(builder, module, args) if args else ir.Constant(self.void_ptr, None)
        
        # Call runtime function
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        class_void_ptr = builder.bitcast(current_class, self.void_ptr)
        num_args = ir.Constant(self.int32, len(args) if args else 0)
        
        result = builder.call(func, [instance_void_ptr, class_void_ptr, name_str, num_args, args_array])
        
        return result
    
    def check_isinstance(self, builder, module, instance_ptr, class_ptr):
        """
        Generate isinstance() check using MRO.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
            class_ptr: Pointer to Class to check
        
        Returns:
            i1 (boolean) result
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.IntType(1), [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="isinstance_check")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        class_void_ptr = builder.bitcast(class_ptr, self.void_ptr)
        
        result = builder.call(func, [instance_void_ptr, class_void_ptr])
        
        return result
    
    def check_issubclass(self, builder, module, subclass_ptr, superclass_ptr):
        """
        Generate issubclass() check using MRO.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            subclass_ptr: Pointer to potential subclass
            superclass_ptr: Pointer to potential superclass
        
        Returns:
            i1 (boolean) result
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.IntType(1), [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="issubclass_check")
        
        subclass_void_ptr = builder.bitcast(subclass_ptr, self.void_ptr)
        superclass_void_ptr = builder.bitcast(superclass_ptr, self.void_ptr)
        
        result = builder.call(func, [subclass_void_ptr, superclass_void_ptr])
        
        return result
    
    def generate_multiple_inheritance_lookup(self, builder, module, class_ptr, attr_name):
        """
        Handle attribute lookup for multiple inheritance.
        Uses diamond problem resolution.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            class_ptr: Pointer to Class
            attr_name: Name of attribute to look up
        
        Returns:
            Attribute value (void*)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="multiple_inheritance_lookup")
        
        name_str = self._create_string_literal(builder, module, attr_name)
        class_void_ptr = builder.bitcast(class_ptr, self.void_ptr)
        
        result = builder.call(func, [class_void_ptr, name_str])
        
        return result
    
    # Helper methods
    
    def _create_string_literal(self, builder, module, string_value):
        """Create a string literal in LLVM IR."""
        string_bytes = (string_value + '\0').encode('utf-8')
        string_const = ir.Constant(ir.ArrayType(self.int8, len(string_bytes)),
                                   bytearray(string_bytes))
        global_str = ir.GlobalVariable(module, string_const.type, 
                                       name=module.get_unique_name("str"))
        global_str.initializer = string_const
        global_str.global_constant = True
        return builder.bitcast(global_str, self.char_ptr)
    
    def _create_pointer_array(self, builder, module, pointers):
        """Create an array of void pointers."""
        if not pointers:
            return ir.Constant(self.void_ptr, None)
        
        array_type = ir.ArrayType(self.void_ptr, len(pointers))
        array_ptr = builder.alloca(array_type)
        
        for i, ptr in enumerate(pointers):
            elem_ptr = builder.gep(array_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, i)])
            ptr_void = builder.bitcast(ptr, self.void_ptr)
            builder.store(ptr_void, elem_ptr)
        
        return builder.bitcast(array_ptr, self.void_ptr)


def generate_inheritance_runtime():
    """Generate C runtime code for inheritance operations."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Forward declarations
typedef struct Class Class;
typedef struct Instance Instance;

struct Class {
    int64_t refcount;
    char* name;
    Class** bases;
    int32_t num_bases;
    void** methods;
    int32_t num_methods;
    char** method_names;
    void** attributes;
    char** attr_names;
    int32_t num_attrs;
};

struct Instance {
    int64_t refcount;
    Class* cls;
    void** attributes;
    char** attr_names;
    int32_t num_attrs;
};

// MRO-based method lookup
void* mro_lookup_method(void* class_ptr, char* method_name) {
    Class* cls = (Class*)class_ptr;
    
    // Search in current class
    for (int32_t i = 0; i < cls->num_methods; i++) {
        if (strcmp(cls->method_names[i], method_name) == 0) {
            return cls->methods[i];
        }
    }
    
    // Search in base classes (left-to-right, depth-first)
    for (int32_t i = 0; i < cls->num_bases; i++) {
        void* method = mro_lookup_method(cls->bases[i], method_name);
        if (method) {
            return method;
        }
    }
    
    return NULL;
}

// Super call - find method in parent class
void* super_call_method(void* instance_ptr, void* current_class_ptr, 
                       char* method_name, int32_t num_args, void* args_array) {
    Instance* inst = (Instance*)instance_ptr;
    Class* current_cls = (Class*)current_class_ptr;
    
    // Find current class in instance's class hierarchy
    // Then look in next class in MRO
    
    // Simplified: just look in first base
    if (current_cls->num_bases > 0) {
        return mro_lookup_method(current_cls->bases[0], method_name);
    }
    
    return NULL;
}

// isinstance check
int32_t isinstance_check(void* instance_ptr, void* class_ptr) {
    Instance* inst = (Instance*)instance_ptr;
    Class* target_cls = (Class*)class_ptr;
    Class* inst_cls = inst->cls;
    
    // Check if instance's class is the target class
    if (inst_cls == target_cls) {
        return 1;
    }
    
    // Check if target class is in instance's class MRO
    for (int32_t i = 0; i < inst_cls->num_bases; i++) {
        if (isinstance_check(instance_ptr, inst_cls->bases[i])) {
            return 1;
        }
    }
    
    return 0;
}

// issubclass check
int32_t issubclass_check(void* subclass_ptr, void* superclass_ptr) {
    Class* sub = (Class*)subclass_ptr;
    Class* super = (Class*)superclass_ptr;
    
    // Check if subclass is superclass
    if (sub == super) {
        return 1;
    }
    
    // Check if superclass is in subclass's bases
    for (int32_t i = 0; i < sub->num_bases; i++) {
        if (issubclass_check(sub->bases[i], superclass_ptr)) {
            return 1;
        }
    }
    
    return 0;
}

// Multiple inheritance attribute lookup
void* multiple_inheritance_lookup(void* class_ptr, char* attr_name) {
    Class* cls = (Class*)class_ptr;
    
    // Search in current class
    for (int32_t i = 0; i < cls->num_attrs; i++) {
        if (strcmp(cls->attr_names[i], attr_name) == 0) {
            return cls->attributes[i];
        }
    }
    
    // Search in base classes (C3 linearization order)
    // Simplified: left-to-right
    for (int32_t i = 0; i < cls->num_bases; i++) {
        void* attr = multiple_inheritance_lookup(cls->bases[i], attr_name);
        if (attr) {
            return attr;
        }
    }
    
    return NULL;
}
"""
    
    # Write to file
    with open('inheritance_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Inheritance runtime generated: inheritance_runtime.c")


if __name__ == "__main__":
    # Generate runtime C code
    generate_inheritance_runtime()
    
    # Test MRO computation
    inheritance = InheritanceType()
    
    # Example: class D(B, C) where B(A), C(A)
    all_classes = {
        'A': [],
        'B': ['A'],
        'C': ['A'],
        'D': ['B', 'C']
    }
    
    mro_d = inheritance.compute_mro('D', ['B', 'C'], all_classes)
    print(f"✅ InheritanceType initialized")
    print(f"   - MRO for D(B, C): {mro_d}")  # Should be [D, B, C, A]
