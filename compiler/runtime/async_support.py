"""
Phase 6: Async/Await Support
Provides async function and await expression support for asynchronous programming.
"""

from llvmlite import ir


class AsyncSupport:
    """
    Handles async/await in compiled code.
    
    Features:
    - async def functions (coroutine functions)
    - await expressions
    - Coroutine objects
    - __await__ protocol
    - Integration with event loops
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.double = ir.DoubleType()
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Coroutine structure
        self.coroutine_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # frame pointer
            self.int32,           # state (running, suspended, finished)
            self.void_ptr,        # awaitable result
            self.void_ptr,        # exception
            self.char_ptr,        # name
            self.int32,           # flags
        ])
        
        # Coroutine states
        self.STATE_CREATED = 0
        self.STATE_RUNNING = 1
        self.STATE_SUSPENDED = 2
        self.STATE_FINISHED = 3
    
    def create_async_function(self, builder, module, func_name, body_func):
        """
        Create an async function (coroutine function).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            func_name: Name of the async function
            body_func: Function pointer to coroutine body
        
        Returns:
            Coroutine function pointer
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.char_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="create_async_function")
        
        name_str = self._create_string_literal(builder, module, func_name)
        body_ptr = builder.bitcast(body_func, self.void_ptr)
        
        result = builder.call(func, [name_str, body_ptr])
        
        return result
    
    def generate_await(self, builder, module, awaitable):
        """
        Generate await expression.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            awaitable: Object to await (coroutine, future, etc.)
        
        Returns:
            Awaited result
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="await_object")
        
        awaitable_ptr = builder.bitcast(awaitable, self.void_ptr)
        result = builder.call(func, [awaitable_ptr])
        
        return result
    
    def create_coroutine(self, builder, module, func_ptr, *args):
        """
        Create a coroutine object from an async function call.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            func_ptr: Async function pointer
            *args: Arguments to pass to async function
        
        Returns:
            Coroutine object
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="create_coroutine")
        
        num_args = ir.Constant(self.int32, len(args))
        args_array = self._create_pointer_array(builder, module, args)
        
        func_void_ptr = builder.bitcast(func_ptr, self.void_ptr)
        result = builder.call(func, [func_void_ptr, num_args, args_array])
        
        return result
    
    def coroutine_send(self, builder, module, coro_ptr, value=None):
        """
        Send value to coroutine (coro.send(value)).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            coro_ptr: Coroutine pointer
            value: Value to send (None for first send)
        
        Returns:
            Next yielded/returned value
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="coroutine_send")
        
        coro_void_ptr = builder.bitcast(coro_ptr, self.void_ptr)
        value_ptr = builder.bitcast(value, self.void_ptr) if value else ir.Constant(self.void_ptr, None)
        
        result = builder.call(func, [coro_void_ptr, value_ptr])
        
        return result
    
    def coroutine_throw(self, builder, module, coro_ptr, exception):
        """
        Throw exception into coroutine (coro.throw(exc)).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            coro_ptr: Coroutine pointer
            exception: Exception to throw
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="coroutine_throw")
        
        coro_void_ptr = builder.bitcast(coro_ptr, self.void_ptr)
        exc_ptr = builder.bitcast(exception, self.void_ptr)
        
        builder.call(func, [coro_void_ptr, exc_ptr])
    
    def coroutine_close(self, builder, module, coro_ptr):
        """
        Close coroutine (coro.close()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            coro_ptr: Coroutine pointer
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="coroutine_close")
        
        coro_void_ptr = builder.bitcast(coro_ptr, self.void_ptr)
        builder.call(func, [coro_void_ptr])
    
    def async_for_loop(self, builder, module, async_iterable, body_func):
        """
        Generate async for loop (async for item in iterable).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            async_iterable: Async iterable object
            body_func: Loop body function
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="async_for_loop")
        
        iter_ptr = builder.bitcast(async_iterable, self.void_ptr)
        body_ptr = builder.bitcast(body_func, self.void_ptr)
        
        builder.call(func, [iter_ptr, body_ptr])
    
    def async_with_statement(self, builder, module, async_context_manager, body_func):
        """
        Generate async with statement (async with ctx as var).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            async_context_manager: Async context manager object
            body_func: With block body function
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="async_with_statement")
        
        ctx_ptr = builder.bitcast(async_context_manager, self.void_ptr)
        body_ptr = builder.bitcast(body_func, self.void_ptr)
        
        builder.call(func, [ctx_ptr, body_ptr])
    
    def get_awaitable(self, builder, module, obj):
        """
        Get awaitable from object (__await__ protocol).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            obj: Object to get awaitable from
        
        Returns:
            Awaitable iterator
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="get_awaitable")
        
        obj_ptr = builder.bitcast(obj, self.void_ptr)
        result = builder.call(func, [obj_ptr])
        
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


def generate_async_runtime():
    """Generate C runtime code for async/await support."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Coroutine states
#define STATE_CREATED   0
#define STATE_RUNNING   1
#define STATE_SUSPENDED 2
#define STATE_FINISHED  3

// Coroutine structure
typedef struct Coroutine {
    int64_t refcount;
    void* frame;
    int32_t state;
    void* result;
    void* exception;
    char* name;
    int32_t flags;
} Coroutine;

// Create async function (coroutine function)
void* create_async_function(char* name, void* body_func) {
    // Would create coroutine function object
    return NULL;
}

// Await an object
void* await_object(void* awaitable) {
    Coroutine* coro = (Coroutine*)awaitable;
    
    if (coro->state == STATE_FINISHED) {
        return coro->result;
    }
    
    // Suspend current coroutine and resume awaited one
    // This would integrate with event loop
    coro->state = STATE_RUNNING;
    
    // Simulate execution
    coro->state = STATE_FINISHED;
    
    return coro->result;
}

// Create coroutine from async function call
void* create_coroutine(void* func_ptr, int32_t num_args, void* args) {
    Coroutine* coro = (Coroutine*)malloc(sizeof(Coroutine));
    coro->refcount = 1;
    coro->frame = NULL;
    coro->state = STATE_CREATED;
    coro->result = NULL;
    coro->exception = NULL;
    coro->name = NULL;
    coro->flags = 0;
    
    return coro;
}

// Send value to coroutine
void* coroutine_send(void* coro_ptr, void* value) {
    Coroutine* coro = (Coroutine*)coro_ptr;
    
    if (coro->state == STATE_FINISHED) {
        // Raise StopIteration
        return NULL;
    }
    
    coro->state = STATE_RUNNING;
    // Execute coroutine until next await
    coro->state = STATE_SUSPENDED;
    
    return coro->result;
}

// Throw exception into coroutine
void coroutine_throw(void* coro_ptr, void* exception) {
    Coroutine* coro = (Coroutine*)coro_ptr;
    coro->exception = exception;
    coro->state = STATE_FINISHED;
}

// Close coroutine
void coroutine_close(void* coro_ptr) {
    Coroutine* coro = (Coroutine*)coro_ptr;
    coro->state = STATE_FINISHED;
    coro->refcount--;
    if (coro->refcount == 0) {
        free(coro);
    }
}

// Async for loop
void async_for_loop(void* async_iterable, void* body_func) {
    // Would iterate using __aiter__/__anext__
}

// Async with statement
void async_with_statement(void* async_context_manager, void* body_func) {
    // Would call __aenter__/__aexit__
}

// Get awaitable from object
void* get_awaitable(void* obj) {
    // Would call __await__() method
    return obj;
}
"""
    
    # Write to file
    with open('async_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Async runtime generated: async_runtime.c")


if __name__ == "__main__":
    # Generate runtime C code
    generate_async_runtime()
    
    # Test async support
    async_support = AsyncSupport()
    
    print(f"✅ AsyncSupport initialized")
    print(f"   - Coroutine structure: {async_support.coroutine_type}")
    print(f"   - States: CREATED={async_support.STATE_CREATED}, "
          f"RUNNING={async_support.STATE_RUNNING}, "
          f"SUSPENDED={async_support.STATE_SUSPENDED}, "
          f"FINISHED={async_support.STATE_FINISHED}")
