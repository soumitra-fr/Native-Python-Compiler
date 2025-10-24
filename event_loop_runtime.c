
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

// Task states
#define TASK_PENDING 0
#define TASK_RUNNING 1
#define TASK_DONE 2
#define TASK_CANCELLED 3

// Event loop structure
typedef struct EventLoop {
    int64_t refcount;
    void* ready_queue;
    void* scheduled_queue;
    int32_t is_running;
    int32_t is_closed;
    double current_time;
    void* exception_handler;
} EventLoop;

// Task structure
typedef struct Task {
    int64_t refcount;
    void* coroutine;
    void* callback;
    int32_t state;
    void* result;
    void* exception;
    char* name;
} Task;

// Handle structure
typedef struct Handle {
    int64_t refcount;
    void* callback;
    void* args;
    double scheduled_time;
    int32_t cancelled;
} Handle;

// Get current time in seconds
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Create event loop
void* create_event_loop() {
    EventLoop* loop = (EventLoop*)malloc(sizeof(EventLoop));
    loop->refcount = 1;
    loop->ready_queue = NULL;  // Would be actual queue
    loop->scheduled_queue = NULL;
    loop->is_running = 0;
    loop->is_closed = 0;
    loop->current_time = get_time();
    loop->exception_handler = NULL;
    return loop;
}

// Run until complete
void* run_until_complete(void* loop_ptr, void* coroutine) {
    EventLoop* loop = (EventLoop*)loop_ptr;
    loop->is_running = 1;
    loop->current_time = get_time();
    
    // Execute coroutine until done
    // This would integrate with coroutine execution
    
    loop->is_running = 0;
    return NULL;  // Return coroutine result
}

// Run forever
void run_forever(void* loop_ptr) {
    EventLoop* loop = (EventLoop*)loop_ptr;
    loop->is_running = 1;
    
    while (loop->is_running && !loop->is_closed) {
        loop->current_time = get_time();
        // Process ready queue
        // Process scheduled queue
        // Sleep until next scheduled callback
    }
}

// Stop event loop
void stop_event_loop(void* loop_ptr) {
    EventLoop* loop = (EventLoop*)loop_ptr;
    loop->is_running = 0;
}

// Create task
void* create_task(void* loop_ptr, void* coroutine, char* name) {
    Task* task = (Task*)malloc(sizeof(Task));
    task->refcount = 1;
    task->coroutine = coroutine;
    task->callback = NULL;
    task->state = TASK_PENDING;
    task->result = NULL;
    task->exception = NULL;
    task->name = name;
    
    // Add to ready queue
    return task;
}

// Schedule callback soon
void* call_soon(void* loop_ptr, void* callback, int32_t num_args, void* args) {
    Handle* handle = (Handle*)malloc(sizeof(Handle));
    handle->refcount = 1;
    handle->callback = callback;
    handle->args = args;
    handle->scheduled_time = 0.0;  // Run immediately
    handle->cancelled = 0;
    
    // Add to ready queue
    return handle;
}

// Schedule callback later
void* call_later(void* loop_ptr, double delay, void* callback, int32_t num_args, void* args) {
    EventLoop* loop = (EventLoop*)loop_ptr;
    
    Handle* handle = (Handle*)malloc(sizeof(Handle));
    handle->refcount = 1;
    handle->callback = callback;
    handle->args = args;
    handle->scheduled_time = loop->current_time + delay;
    handle->cancelled = 0;
    
    // Add to scheduled queue
    return handle;
}

// Schedule callback at time
void* call_at(void* loop_ptr, double when, void* callback, int32_t num_args, void* args) {
    Handle* handle = (Handle*)malloc(sizeof(Handle));
    handle->refcount = 1;
    handle->callback = callback;
    handle->args = args;
    handle->scheduled_time = when;
    handle->cancelled = 0;
    
    // Add to scheduled queue
    return handle;
}

// Get running loop
static EventLoop* g_running_loop = NULL;

void* get_running_loop() {
    return g_running_loop;
}

// Check if running
int32_t is_loop_running(void* loop_ptr) {
    EventLoop* loop = (EventLoop*)loop_ptr;
    return loop->is_running;
}

// Close event loop
void close_event_loop(void* loop_ptr) {
    EventLoop* loop = (EventLoop*)loop_ptr;
    loop->is_closed = 1;
    // Clean up queues
}
