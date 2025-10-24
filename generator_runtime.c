
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
