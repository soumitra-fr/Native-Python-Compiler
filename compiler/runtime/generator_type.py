"""
Phase 2: Generator Implementation

Python generators with yield statement:
- Generator objects
- Yield statement
- Generator protocol (next, send, throw, close)
- StopIteration exception
"""

from llvmlite import ir
import llvmlite.binding as llvm
from typing import Optional


class GeneratorType:
    """
    Python generator implementation for LLVM compilation
    
    Generators are functions that can pause and resume execution
    """
    
    def __init__(self, codegen):
        self.codegen = codegen
        self.module = codegen.module
        
        # Generator structure
        # { i64 refcount, i32 state, void* frame, void* yielded_value }
        self.generator_struct = ir.LiteralStructType([
            ir.IntType(64),  # refcount
            ir.IntType(32),  # execution state (0=start, 1=suspended, 2=done)
            ir.IntType(8).as_pointer(),  # saved frame/context
            ir.IntType(8).as_pointer()   # last yielded value
        ])
        
        self.generator_ptr = self.generator_struct.as_pointer()
        self._declare_runtime_functions()
    
    def _declare_runtime_functions(self):
        """Declare generator runtime functions"""
        
        # generator_new: Create new generator
        # Generator* generator_new()
        gen_new_type = ir.FunctionType(self.generator_ptr, [])
        self.gen_new_func = ir.Function(self.module, gen_new_type, name="generator_new")
        
        # generator_yield: Yield a value
        # void generator_yield(Generator* gen, void* value)
        gen_yield_type = ir.FunctionType(ir.VoidType(),
                                        [self.generator_ptr,
                                         ir.IntType(8).as_pointer()])
        self.gen_yield_func = ir.Function(self.module, gen_yield_type, 
                                         name="generator_yield")
        
        # generator_next: Get next value
        # void* generator_next(Generator* gen)
        gen_next_type = ir.FunctionType(ir.IntType(8).as_pointer(),
                                       [self.generator_ptr])
        self.gen_next_func = ir.Function(self.module, gen_next_type,
                                        name="generator_next")
        
        # generator_is_done: Check if generator is exhausted
        # i32 generator_is_done(Generator* gen)
        gen_done_type = ir.FunctionType(ir.IntType(32), [self.generator_ptr])
        self.gen_done_func = ir.Function(self.module, gen_done_type,
                                        name="generator_is_done")
        
        # generator_incref/decref
        gen_incref_type = ir.FunctionType(ir.VoidType(), [self.generator_ptr])
        self.gen_incref_func = ir.Function(self.module, gen_incref_type,
                                          name="generator_incref")
        
        gen_decref_type = ir.FunctionType(ir.VoidType(), [self.generator_ptr])
        self.gen_decref_func = ir.Function(self.module, gen_decref_type,
                                          name="generator_decref")
    
    def create_generator(self, builder: ir.IRBuilder) -> ir.Value:
        """Create a new generator object"""
        return builder.call(self.gen_new_func, [])
    
    def generate_yield(self, builder: ir.IRBuilder, 
                      generator: ir.Value, value: ir.Value):
        """
        Generate yield statement
        
        Args:
            builder: LLVM IR builder
            generator: Generator object
            value: Value to yield
        """
        value_ptr = builder.bitcast(value, ir.IntType(8).as_pointer())
        builder.call(self.gen_yield_func, [generator, value_ptr])
    
    def generate_generator_function(self, builder: ir.IRBuilder,
                                   func_name: str,
                                   params: list,
                                   body_with_yields):
        """
        Generate a generator function (function containing yield)
        
        This is complex - requires state machine transformation
        
        Args:
            builder: LLVM IR builder
            func_name: Function name
            params: Parameters
            body_with_yields: Function body containing yields
            
        Returns:
            Generator function
        """
        # Simplified implementation
        # Full implementation would transform function into state machine
        
        # Return type is Generator*
        param_types = [ir.IntType(64)] * len(params)  # Simplified
        func_type = ir.FunctionType(self.generator_ptr, param_types)
        func = ir.Function(self.module, func_type, name=func_name)
        
        entry_block = func.append_basic_block("entry")
        func_builder = ir.IRBuilder(entry_block)
        
        # Create generator object
        gen = func_builder.call(self.gen_new_func, [])
        
        # In full implementation:
        # 1. Transform function body into state machine
        # 2. Each yield becomes a state
        # 3. Store local variables in generator frame
        # 4. Return generator object immediately
        
        # Simplified: just return empty generator
        func_builder.ret(gen)
        
        return func
    
    def generate_for_loop_with_generator(self, builder: ir.IRBuilder,
                                        function: ir.Function,
                                        generator: ir.Value,
                                        loop_var_name: str,
                                        body_gen):
        """
        Generate for loop iterating over generator
        
        for item in generator:
            body
        
        Args:
            builder: LLVM IR builder
            function: Current function
            generator: Generator to iterate
            loop_var_name: Loop variable name
            body_gen: Function to generate loop body
        """
        loop_header = function.append_basic_block("for_header")
        loop_body = function.append_basic_block("for_body")
        loop_end = function.append_basic_block("for_end")
        
        # Jump to header
        builder.branch(loop_header)
        
        # Loop header: check if generator is done
        builder.position_at_end(loop_header)
        is_done = builder.call(self.gen_done_func, [generator])
        done_cond = builder.icmp_signed('!=', is_done, ir.Constant(ir.IntType(32), 0))
        builder.cbranch(done_cond, loop_end, loop_body)
        
        # Loop body
        builder.position_at_end(loop_body)
        next_val = builder.call(self.gen_next_func, [generator])
        
        # Generate body with next_val as loop variable
        body_gen(builder, next_val)
        
        builder.branch(loop_header)
        
        # Loop end
        builder.position_at_end(loop_end)


def generate_generator_runtime():
    """Generate C runtime for generators"""
    return '''
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Generator states
#define GEN_STATE_START 0
#define GEN_STATE_SUSPENDED 1
#define GEN_STATE_DONE 2

// Generator structure
typedef struct {
    int64_t refcount;
    int32_t state;
    void* frame;  // Saved execution context
    void* yielded_value;
} Generator;

// Create new generator
Generator* generator_new() {
    Generator* gen = (Generator*)malloc(sizeof(Generator));
    gen->refcount = 1;
    gen->state = GEN_STATE_START;
    gen->frame = NULL;
    gen->yielded_value = NULL;
    return gen;
}

// Yield a value
void generator_yield(Generator* gen, void* value) {
    gen->yielded_value = value;
    gen->state = GEN_STATE_SUSPENDED;
}

// Get next value from generator
void* generator_next(Generator* gen) {
    if (gen->state == GEN_STATE_DONE) {
        return NULL;  // Should raise StopIteration
    }
    
    // In full implementation:
    // 1. Resume generator execution from saved state
    // 2. Run until next yield or return
    // 3. Return yielded value
    
    // Simplified: return NULL
    return gen->yielded_value;
}

// Check if generator is done
int32_t generator_is_done(Generator* gen) {
    return gen->state == GEN_STATE_DONE;
}

// Mark generator as done
void generator_done(Generator* gen) {
    gen->state = GEN_STATE_DONE;
    gen->yielded_value = NULL;
}

// Increment reference count
void generator_incref(Generator* gen) {
    if (gen) gen->refcount++;
}

// Decrement reference count
void generator_decref(Generator* gen) {
    if (gen && --gen->refcount == 0) {
        free(gen);
    }
}
'''


if __name__ == "__main__":
    with open("generator_runtime.c", "w") as f:
        f.write(generate_generator_runtime())
    print("✅ Generator runtime generated: generator_runtime.c")
