
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    int64_t refcount;
    int64_t length;
    void** items;
} Tuple;

Tuple* tuple_new(void** items, int64_t length) {
    Tuple* tup = (Tuple*)malloc(sizeof(Tuple));
    tup->refcount = 1;
    tup->length = length;
    tup->items = (void**)malloc(sizeof(void*) * length);
    memcpy(tup->items, items, sizeof(void*) * length);
    return tup;
}

void tuple_incref(Tuple* tup) {
    if (tup) tup->refcount++;
}

void tuple_decref(Tuple* tup) {
    if (tup && --tup->refcount == 0) {
        free(tup->items);
        free(tup);
    }
}

int64_t tuple_len(Tuple* tup) {
    return tup->length;
}

void* tuple_get(Tuple* tup, int64_t index) {
    if (index < 0 || index >= tup->length) return NULL;
    return tup->items[index];
}
