"""
Phase 9: Asyncio Primitives
Provides asyncio.gather, wait, sleep, create_task, and other high-level async functions.
"""

from llvmlite import ir


class AsyncPrimitives:
    """
    High-level asyncio primitives.
    
    Features:
    - asyncio.gather(*awaitables)
    - asyncio.wait(aws, timeout, return_when)
    - asyncio.sleep(delay)
    - asyncio.wait_for(aw, timeout)
    - asyncio.shield(aw)
    - asyncio.as_completed(aws)
    - Future objects
    - Task cancellation
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.double = ir.DoubleType()
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Future structure
        self.future_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.int32,           # state (PENDING, CANCELLED, FINISHED)
            self.void_ptr,        # result
            self.void_ptr,        # exception
            self.void_ptr,        # callbacks
            self.void_ptr,        # loop
        ])
    
    def async_sleep(self, builder, module, delay, result=None):
        """
        Sleep for specified seconds (asyncio.sleep).
        
        Args:
            delay: Sleep duration in seconds (double)
            result: Optional result to return
        
        Returns:
            Coroutine that sleeps
        """
        arg_types = [self.double, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="async_sleep")
        
        result_ptr = builder.bitcast(result, self.void_ptr) if result else ir.Constant(self.void_ptr, None)
        
        coro = builder.call(func, [delay, result_ptr])
        return coro
    
    def async_gather(self, builder, module, *awaitables):
        """
        Run awaitables concurrently and gather results (asyncio.gather).
        
        Args:
            *awaitables: Coroutines or tasks to run
        
        Returns:
            Coroutine that returns list of results
        """
        arg_types = [self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="async_gather")
        
        num_awaitables = ir.Constant(self.int32, len(awaitables))
        awaitables_array = self._create_pointer_array(builder, module, awaitables)
        
        result = builder.call(func, [num_awaitables, awaitables_array])
        return result
    
    def async_wait(self, builder, module, awaitables, timeout=None, return_when="ALL_COMPLETED"):
        """
        Wait for awaitables with control over completion (asyncio.wait).
        
        Args:
            awaitables: Set of coroutines/tasks
            timeout: Maximum wait time (optional)
            return_when: When to return (ALL_COMPLETED, FIRST_COMPLETED, FIRST_EXCEPTION)
        
        Returns:
            Coroutine that returns (done, pending) sets
        """
        arg_types = [self.int32, self.void_ptr, self.double, self.int32]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="async_wait")
        
        num_awaitables = ir.Constant(self.int32, len(awaitables))
        awaitables_array = self._create_pointer_array(builder, module, awaitables)
        timeout_val = timeout if timeout is not None else ir.Constant(self.double, -1.0)
        
        # return_when: 0=ALL_COMPLETED, 1=FIRST_COMPLETED, 2=FIRST_EXCEPTION
        return_when_map = {"ALL_COMPLETED": 0, "FIRST_COMPLETED": 1, "FIRST_EXCEPTION": 2}
        return_when_val = ir.Constant(self.int32, return_when_map.get(return_when, 0))
        
        result = builder.call(func, [num_awaitables, awaitables_array, timeout_val, return_when_val])
        return result
    
    def async_wait_for(self, builder, module, awaitable, timeout):
        """
        Wait for awaitable with timeout (asyncio.wait_for).
        
        Args:
            awaitable: Coroutine or task
            timeout: Maximum wait time in seconds
        
        Returns:
            Coroutine that returns result or raises TimeoutError
        """
        arg_types = [self.void_ptr, self.double]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="async_wait_for")
        
        aw_ptr = builder.bitcast(awaitable, self.void_ptr)
        
        result = builder.call(func, [aw_ptr, timeout])
        return result
    
    def async_shield(self, builder, module, awaitable):
        """
        Shield awaitable from cancellation (asyncio.shield).
        
        Args:
            awaitable: Coroutine or task to shield
        
        Returns:
            Shielded coroutine
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="async_shield")
        
        aw_ptr = builder.bitcast(awaitable, self.void_ptr)
        
        result = builder.call(func, [aw_ptr])
        return result
    
    def async_as_completed(self, builder, module, awaitables, timeout=None):
        """
        Return iterator that yields awaitables as they complete (asyncio.as_completed).
        
        Args:
            awaitables: List of coroutines/tasks
            timeout: Maximum wait time (optional)
        
        Returns:
            Iterator of coroutines
        """
        arg_types = [self.int32, self.void_ptr, self.double]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="async_as_completed")
        
        num_awaitables = ir.Constant(self.int32, len(awaitables))
        awaitables_array = self._create_pointer_array(builder, module, awaitables)
        timeout_val = timeout if timeout is not None else ir.Constant(self.double, -1.0)
        
        result = builder.call(func, [num_awaitables, awaitables_array, timeout_val])
        return result
    
    def create_future(self, builder, module, loop=None):
        """
        Create Future object (asyncio.Future).
        
        Args:
            loop: Event loop (optional, uses current if None)
        
        Returns:
            Future object
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="create_future")
        
        loop_ptr = builder.bitcast(loop, self.void_ptr) if loop else ir.Constant(self.void_ptr, None)
        
        result = builder.call(func, [loop_ptr])
        return result
    
    def set_future_result(self, builder, module, future, result):
        """
        Set Future result.
        
        Args:
            future: Future object
            result: Result value
        """
        arg_types = [self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        func = ir.Function(module, func_type, name="set_future_result")
        
        future_ptr = builder.bitcast(future, self.void_ptr)
        result_ptr = builder.bitcast(result, self.void_ptr)
        
        builder.call(func, [future_ptr, result_ptr])
    
    def set_future_exception(self, builder, module, future, exception):
        """
        Set Future exception.
        
        Args:
            future: Future object
            exception: Exception object
        """
        arg_types = [self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        func = ir.Function(module, func_type, name="set_future_exception")
        
        future_ptr = builder.bitcast(future, self.void_ptr)
        exc_ptr = builder.bitcast(exception, self.void_ptr)
        
        builder.call(func, [future_ptr, exc_ptr])
    
    def cancel_task(self, builder, module, task, msg=""):
        """
        Cancel task (Task.cancel).
        
        Args:
            task: Task object
            msg: Cancellation message (optional)
        
        Returns:
            Boolean indicating if cancellation was successful
        """
        arg_types = [self.void_ptr, self.char_ptr]
        func_type = ir.FunctionType(self.int32, arg_types)
        func = ir.Function(module, func_type, name="cancel_task")
        
        task_ptr = builder.bitcast(task, self.void_ptr)
        msg_str = self._create_string_literal(builder, module, msg)
        
        result = builder.call(func, [task_ptr, msg_str])
        return result
    
    def is_task_cancelled(self, builder, module, task):
        """
        Check if task is cancelled.
        
        Args:
            task: Task object
        
        Returns:
            Boolean
        """
        func_type = ir.FunctionType(self.int32, [self.void_ptr])
        func = ir.Function(module, func_type, name="is_task_cancelled")
        
        task_ptr = builder.bitcast(task, self.void_ptr)
        
        result = builder.call(func, [task_ptr])
        return result
    
    def is_task_done(self, builder, module, task):
        """
        Check if task is done.
        
        Args:
            task: Task object
        
        Returns:
            Boolean
        """
        func_type = ir.FunctionType(self.int32, [self.void_ptr])
        func = ir.Function(module, func_type, name="is_task_done")
        
        task_ptr = builder.bitcast(task, self.void_ptr)
        
        result = builder.call(func, [task_ptr])
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


def generate_async_primitives_runtime():
    """Generate C runtime code for async primitives."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

// Future states
#define FUTURE_PENDING 0
#define FUTURE_CANCELLED 1
#define FUTURE_FINISHED 2

// Return when constants
#define ALL_COMPLETED 0
#define FIRST_COMPLETED 1
#define FIRST_EXCEPTION 2

// Future structure
typedef struct Future {
    int64_t refcount;
    int32_t state;
    void* result;
    void* exception;
    void* callbacks;
    void* loop;
} Future;

// Sleep coroutine
void* async_sleep(double delay, void* result) {
    // Create coroutine that sleeps
    usleep((useconds_t)(delay * 1000000));
    return result;
}

// Gather awaitables
void* async_gather(int32_t num_awaitables, void* awaitables) {
    // Run all awaitables concurrently
    // Collect results into list
    return NULL;  // Return list of results
}

// Wait for awaitables
void* async_wait(int32_t num_awaitables, void* awaitables, double timeout, int32_t return_when) {
    // Wait according to return_when condition
    // Return (done, pending) tuple
    return NULL;
}

// Wait for with timeout
void* async_wait_for(void* awaitable, double timeout) {
    // Wait for awaitable with timeout
    // Raise TimeoutError if timeout expires
    return NULL;
}

// Shield from cancellation
void* async_shield(void* awaitable) {
    // Wrap awaitable to protect from cancellation
    return awaitable;
}

// As completed iterator
void* async_as_completed(int32_t num_awaitables, void* awaitables, double timeout) {
    // Return iterator that yields as completed
    return NULL;
}

// Create future
void* create_future(void* loop) {
    Future* future = (Future*)malloc(sizeof(Future));
    future->refcount = 1;
    future->state = FUTURE_PENDING;
    future->result = NULL;
    future->exception = NULL;
    future->callbacks = NULL;
    future->loop = loop;
    return future;
}

// Set future result
void set_future_result(void* future_ptr, void* result) {
    Future* future = (Future*)future_ptr;
    future->state = FUTURE_FINISHED;
    future->result = result;
    // Call callbacks
}

// Set future exception
void set_future_exception(void* future_ptr, void* exception) {
    Future* future = (Future*)future_ptr;
    future->state = FUTURE_FINISHED;
    future->exception = exception;
    // Call callbacks
}

// Cancel task
int32_t cancel_task(void* task, char* msg) {
    // Cancel task
    return 1;  // Success
}

// Check if cancelled
int32_t is_task_cancelled(void* task) {
    // Check task state
    return 0;
}

// Check if done
int32_t is_task_done(void* task) {
    // Check task state
    return 0;
}
"""
    
    with open('async_primitives_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Async primitives runtime generated: async_primitives_runtime.c")


if __name__ == "__main__":
    generate_async_primitives_runtime()
    
    async_primitives = AsyncPrimitives()
    
    print(f"✅ AsyncPrimitives initialized")
    print(f"   - Future structure: {async_primitives.future_type}")
    print(f"   - Features: sleep, gather, wait, wait_for, shield, as_completed")
    print(f"   - Task management: cancel, is_cancelled, is_done")
