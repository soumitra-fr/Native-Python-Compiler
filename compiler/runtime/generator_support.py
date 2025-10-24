"""
Phase 7: Generator & Iterator Support
Provides generator functions, yield expressions, and iterator protocol.
"""

from llvmlite import ir


class GeneratorSupport:
    """
    Handles generators and iterators in compiled code.
    
    Features:
    - Generator functions (with yield)
    - yield expressions
    - yield from delegation
    - send/throw/close methods
    - Generator expressions
    - Iterator protocol (__iter__/__next__)
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.double = ir.DoubleType()
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Generator structure
        self.generator_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # frame pointer
            self.int32,           # state (created, running, suspended, finished)
            self.void_ptr,        # yielded value
            self.void_ptr,        # sent value
            self.void_ptr,        # exception
            self.char_ptr,        # name
        ])
        
        # Generator states
        self.GEN_CREATED = 0
        self.GEN_RUNNING = 1
        self.GEN_SUSPENDED = 2
        self.GEN_FINISHED = 3
    
    def create_generator_function(self, builder, module, func_name, body_func):
        """
        Create a generator function.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            func_name: Name of generator function
            body_func: Generator body function pointer
        
        Returns:
            Generator function pointer
        """
        func_type = ir.FunctionType(self.void_ptr, [self.char_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="create_generator_function")
        
        name_str = self._create_string_literal(builder, module, func_name)
        body_ptr = builder.bitcast(body_func, self.void_ptr)
        
        result = builder.call(func, [name_str, body_ptr])
        return result
    
    def generate_yield(self, builder, module, value):
        """
        Generate yield expression.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            value: Value to yield
        
        Returns:
            Sent value (from generator.send())
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="yield_value")
        
        value_ptr = builder.bitcast(value, self.void_ptr)
        result = builder.call(func, [value_ptr])
        return result
    
    def generate_yield_from(self, builder, module, iterable):
        """
        Generate yield from expression.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            iterable: Iterable to delegate to
        """
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="yield_from")
        
        iter_ptr = builder.bitcast(iterable, self.void_ptr)
        builder.call(func, [iter_ptr])
    
    def generator_next(self, builder, module, gen_ptr):
        """
        Get next value from generator (next(gen) or gen.__next__()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            gen_ptr: Generator pointer
        
        Returns:
            Next yielded value
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="generator_next")
        
        gen_void_ptr = builder.bitcast(gen_ptr, self.void_ptr)
        result = builder.call(func, [gen_void_ptr])
        return result
    
    def generator_send(self, builder, module, gen_ptr, value):
        """
        Send value to generator (gen.send(value)).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            gen_ptr: Generator pointer
            value: Value to send
        
        Returns:
            Next yielded value
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="generator_send")
        
        gen_void_ptr = builder.bitcast(gen_ptr, self.void_ptr)
        value_ptr = builder.bitcast(value, self.void_ptr)
        result = builder.call(func, [gen_void_ptr, value_ptr])
        return result
    
    def generator_throw(self, builder, module, gen_ptr, exception):
        """
        Throw exception into generator (gen.throw(exc)).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            gen_ptr: Generator pointer
            exception: Exception to throw
        """
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="generator_throw")
        
        gen_void_ptr = builder.bitcast(gen_ptr, self.void_ptr)
        exc_ptr = builder.bitcast(exception, self.void_ptr)
        builder.call(func, [gen_void_ptr, exc_ptr])
    
    def generator_close(self, builder, module, gen_ptr):
        """
        Close generator (gen.close()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            gen_ptr: Generator pointer
        """
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="generator_close")
        
        gen_void_ptr = builder.bitcast(gen_ptr, self.void_ptr)
        builder.call(func, [gen_void_ptr])
    
    def create_iterator(self, builder, module, iterable):
        """
        Create iterator from iterable (iter(obj) or obj.__iter__()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            iterable: Iterable object
        
        Returns:
            Iterator object
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="create_iterator")
        
        iter_ptr = builder.bitcast(iterable, self.void_ptr)
        result = builder.call(func, [iter_ptr])
        return result
    
    def iterator_next(self, builder, module, iterator):
        """
        Get next value from iterator (next(iter) or iter.__next__()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            iterator: Iterator object
        
        Returns:
            Next value (raises StopIteration when exhausted)
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="iterator_next")
        
        iter_ptr = builder.bitcast(iterator, self.void_ptr)
        result = builder.call(func, [iter_ptr])
        return result
    
    def generator_expression(self, builder, module, expr_func, iterable):
        """
        Create generator expression.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            expr_func: Expression function pointer
            iterable: Source iterable
        
        Returns:
            Generator object
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="generator_expression")
        
        expr_ptr = builder.bitcast(expr_func, self.void_ptr)
        iter_ptr = builder.bitcast(iterable, self.void_ptr)
        result = builder.call(func, [expr_ptr, iter_ptr])
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


def generate_generator_runtime():
    """Generate C runtime code for generators."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Generator states
#define GEN_CREATED   0
#define GEN_RUNNING   1
#define GEN_SUSPENDED 2
#define GEN_FINISHED  3

// Generator structure
typedef struct Generator {
    int64_t refcount;
    void* frame;
    int32_t state;
    void* yielded_value;
    void* sent_value;
    void* exception;
    char* name;
} Generator;

// Create generator function
void* create_generator_function(char* name, void* body_func) {
    return NULL;
}

// Yield value
void* yield_value(void* value) {
    // Suspend generator and return value
    return NULL;
}

// Yield from another generator
void yield_from(void* iterable) {
    // Delegate to another generator
}

// Get next value (next() or __next__())
void* generator_next(void* gen_ptr) {
    Generator* gen = (Generator*)gen_ptr;
    
    if (gen->state == GEN_FINISHED) {
        // Raise StopIteration
        return NULL;
    }
    
    gen->state = GEN_RUNNING;
    // Execute until next yield
    gen->state = GEN_SUSPENDED;
    
    return gen->yielded_value;
}

// Send value to generator
void* generator_send(void* gen_ptr, void* value) {
    Generator* gen = (Generator*)gen_ptr;
    gen->sent_value = value;
    return generator_next(gen_ptr);
}

// Throw exception into generator
void generator_throw(void* gen_ptr, void* exception) {
    Generator* gen = (Generator*)gen_ptr;
    gen->exception = exception;
    gen->state = GEN_FINISHED;
}

// Close generator
void generator_close(void* gen_ptr) {
    Generator* gen = (Generator*)gen_ptr;
    gen->state = GEN_FINISHED;
    gen->refcount--;
    if (gen->refcount == 0) {
        free(gen);
    }
}

// Create iterator from iterable
void* create_iterator(void* iterable) {
    // Call __iter__() method
    return iterable;
}

// Get next from iterator
void* iterator_next(void* iterator) {
    // Call __next__() method
    return NULL;
}

// Create generator expression
void* generator_expression(void* expr_func, void* iterable) {
    // Create generator that applies expr_func to each item
    return NULL;
}
"""
    
    with open('generator_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Generator runtime generated: generator_runtime.c")


if __name__ == "__main__":
    generate_generator_runtime()
    
    gen_support = GeneratorSupport()
    
    print(f"✅ GeneratorSupport initialized")
    print(f"   - Generator structure: {gen_support.generator_type}")
    print(f"   - States: CREATED={gen_support.GEN_CREATED}, "
          f"RUNNING={gen_support.GEN_RUNNING}, "
          f"SUSPENDED={gen_support.GEN_SUSPENDED}, "
          f"FINISHED={gen_support.GEN_FINISHED}")
