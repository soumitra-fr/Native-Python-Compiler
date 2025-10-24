"""
Phase 3: Property and Descriptor Implementation
Provides property objects and descriptor protocol support.
"""

from llvmlite import ir

class PropertyType:
    """
    Handles Python properties and descriptors.
    
    Descriptors are objects that define __get__, __set__, and/or __delete__.
    Properties are a common use case for descriptors.
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Property structure
        self.property_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # fget (getter function)
            self.void_ptr,        # fset (setter function)
            self.void_ptr,        # fdel (deleter function)
            self.char_ptr,        # doc (documentation string)
        ])
    
    def create_property(self, builder, fget=None, fset=None, fdel=None, doc=None):
        """
        Create a property object.
        
        Args:
            builder: LLVM IR builder
            fget: Getter function pointer (optional)
            fset: Setter function pointer (optional)
            fdel: Deleter function pointer (optional)
            doc: Documentation string (optional)
        
        Returns:
            Pointer to Property structure
        """
        # Allocate property structure
        prop_ptr = builder.alloca(self.property_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(prop_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set getter
        fget_ptr = builder.gep(prop_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        if fget:
            fget_void_ptr = builder.bitcast(fget, self.void_ptr)
            builder.store(fget_void_ptr, fget_ptr)
        else:
            builder.store(ir.Constant(self.void_ptr, None), fget_ptr)
        
        # Set setter
        fset_ptr = builder.gep(prop_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
        if fset:
            fset_void_ptr = builder.bitcast(fset, self.void_ptr)
            builder.store(fset_void_ptr, fset_ptr)
        else:
            builder.store(ir.Constant(self.void_ptr, None), fset_ptr)
        
        # Set deleter
        fdel_ptr = builder.gep(prop_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 3)])
        if fdel:
            fdel_void_ptr = builder.bitcast(fdel, self.void_ptr)
            builder.store(fdel_void_ptr, fdel_ptr)
        else:
            builder.store(ir.Constant(self.void_ptr, None), fdel_ptr)
        
        # Set doc string
        doc_ptr = builder.gep(prop_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 4)])
        if doc:
            builder.store(doc, doc_ptr)
        else:
            builder.store(ir.Constant(self.char_ptr, None), doc_ptr)
        
        return prop_ptr
    
    def property_get(self, builder, module, property_ptr, instance_ptr):
        """
        Call property getter (__get__).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            property_ptr: Pointer to Property
            instance_ptr: Pointer to Instance (self)
        
        Returns:
            Value returned by getter
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="property_get")
        
        property_void_ptr = builder.bitcast(property_ptr, self.void_ptr)
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        
        result = builder.call(func, [property_void_ptr, instance_void_ptr])
        
        return result
    
    def property_set(self, builder, module, property_ptr, instance_ptr, value):
        """
        Call property setter (__set__).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            property_ptr: Pointer to Property
            instance_ptr: Pointer to Instance (self)
            value: Value to set
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="property_set")
        
        property_void_ptr = builder.bitcast(property_ptr, self.void_ptr)
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        value_void_ptr = builder.bitcast(value, self.void_ptr)
        
        builder.call(func, [property_void_ptr, instance_void_ptr, value_void_ptr])
    
    def property_delete(self, builder, module, property_ptr, instance_ptr):
        """
        Call property deleter (__delete__).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            property_ptr: Pointer to Property
            instance_ptr: Pointer to Instance (self)
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="property_delete")
        
        property_void_ptr = builder.bitcast(property_ptr, self.void_ptr)
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        
        builder.call(func, [property_void_ptr, instance_void_ptr])
    
    def create_descriptor(self, builder, get_func, set_func, delete_func):
        """
        Create a descriptor object (implements descriptor protocol).
        
        Args:
            builder: LLVM IR builder
            get_func: __get__ function pointer
            set_func: __set__ function pointer
            delete_func: __delete__ function pointer
        
        Returns:
            Pointer to descriptor (same as property)
        """
        return self.create_property(builder, fget=get_func, fset=set_func, fdel=delete_func)
    
    def descriptor_get(self, builder, module, descriptor_ptr, instance_ptr, owner_ptr):
        """
        Call descriptor __get__ method.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            descriptor_ptr: Pointer to Descriptor
            instance_ptr: Pointer to Instance (can be None for class access)
            owner_ptr: Pointer to owner Class
        
        Returns:
            Value returned by __get__
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="descriptor_get")
        
        descriptor_void_ptr = builder.bitcast(descriptor_ptr, self.void_ptr)
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr) if instance_ptr else ir.Constant(self.void_ptr, None)
        owner_void_ptr = builder.bitcast(owner_ptr, self.void_ptr)
        
        result = builder.call(func, [descriptor_void_ptr, instance_void_ptr, owner_void_ptr])
        
        return result
    
    def descriptor_set(self, builder, module, descriptor_ptr, instance_ptr, value):
        """
        Call descriptor __set__ method.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            descriptor_ptr: Pointer to Descriptor
            instance_ptr: Pointer to Instance
            value: Value to set
        """
        # Same as property_set
        self.property_set(builder, module, descriptor_ptr, instance_ptr, value)
    
    def descriptor_delete(self, builder, module, descriptor_ptr, instance_ptr):
        """
        Call descriptor __delete__ method.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            descriptor_ptr: Pointer to Descriptor
            instance_ptr: Pointer to Instance
        """
        # Same as property_delete
        self.property_delete(builder, module, descriptor_ptr, instance_ptr)
    
    def incref(self, builder, module, property_ptr):
        """Increment reference count."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="property_incref")
        property_void_ptr = builder.bitcast(property_ptr, self.void_ptr)
        builder.call(func, [property_void_ptr])
    
    def decref(self, builder, module, property_ptr):
        """Decrement reference count and free if zero."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="property_decref")
        property_void_ptr = builder.bitcast(property_ptr, self.void_ptr)
        builder.call(func, [property_void_ptr])


def generate_property_runtime():
    """Generate C runtime code for properties and descriptors."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Property structure
typedef struct Property {
    int64_t refcount;
    void* fget;        // Getter function
    void* fset;        // Setter function
    void* fdel;        // Deleter function
    char* doc;         // Documentation
} Property;

// Reference counting
void property_incref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)++;
    }
}

void property_decref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)--;
        if (*refcount == 0) {
            Property* prop = (Property*)obj;
            if (prop->doc) {
                free(prop->doc);
            }
            free(obj);
        }
    }
}

// Property get
void* property_get(void* property_ptr, void* instance_ptr) {
    Property* prop = (Property*)property_ptr;
    
    if (prop->fget) {
        // Would call getter function with instance
        // For now, return NULL
        return NULL;
    }
    
    // AttributeError: unreadable attribute
    return NULL;
}

// Property set
void property_set(void* property_ptr, void* instance_ptr, void* value) {
    Property* prop = (Property*)property_ptr;
    
    if (prop->fset) {
        // Would call setter function with instance and value
        // For now, no-op
        return;
    }
    
    // AttributeError: can't set attribute
}

// Property delete
void property_delete(void* property_ptr, void* instance_ptr) {
    Property* prop = (Property*)property_ptr;
    
    if (prop->fdel) {
        // Would call deleter function with instance
        // For now, no-op
        return;
    }
    
    // AttributeError: can't delete attribute
}

// Descriptor get (extends property_get with owner parameter)
void* descriptor_get(void* descriptor_ptr, void* instance_ptr, void* owner_ptr) {
    // If instance is NULL, return descriptor itself (class access)
    if (instance_ptr == NULL) {
        return descriptor_ptr;
    }
    
    // Otherwise, call getter
    return property_get(descriptor_ptr, instance_ptr);
}
"""
    
    # Write to file
    with open('property_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Property runtime generated: property_runtime.c")


if __name__ == "__main__":
    # Generate runtime C code
    generate_property_runtime()
    
    # Test property type
    prop_type = PropertyType()
    print(f"✅ PropertyType initialized")
    print(f"   - Property structure: {prop_type.property_type}")
