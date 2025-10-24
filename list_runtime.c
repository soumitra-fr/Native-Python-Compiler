
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    int64_t refcount;
    int64_t length;
    int64_t capacity;
    void** items;
} List;

List* list_new() {
    List* lst = (List*)malloc(sizeof(List));
    lst->refcount = 1;
    lst->length = 0;
    lst->capacity = 8;
    lst->items = (void**)malloc(sizeof(void*) * lst->capacity);
    return lst;
}

void list_incref(List* lst) {
    if (lst) lst->refcount++;
}

void list_decref(List* lst) {
    if (lst && --lst->refcount == 0) {
        free(lst->items);
        free(lst);
    }
}

int64_t list_len(List* lst) {
    return lst->length;
}

void list_resize(List* lst) {
    if (lst->length >= lst->capacity) {
        lst->capacity *= 2;
        lst->items = (void**)realloc(lst->items, sizeof(void*) * lst->capacity);
    }
}

void list_append(List* lst, void* item) {
    list_resize(lst);
    lst->items[lst->length++] = item;
}

void* list_get(List* lst, int64_t index) {
    if (index < 0 || index >= lst->length) return NULL;
    return lst->items[index];
}

void list_set(List* lst, int64_t index, void* item) {
    if (index >= 0 && index < lst->length) {
        lst->items[index] = item;
    }
}

void* list_pop(List* lst) {
    if (lst->length == 0) return NULL;
    return lst->items[--lst->length];
}

List* list_slice(List* lst, int64_t start, int64_t end) {
    if (start < 0) start = 0;
    if (end > lst->length) end = lst->length;
    if (start >= end) return list_new();
    
    List* result = list_new();
    for (int64_t i = start; i < end; i++) {
        list_append(result, lst->items[i]);
    }
    return result;
}
