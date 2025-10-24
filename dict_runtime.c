
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    void* key;
    void* value;
    int64_t hash;
    int32_t occupied;
} DictEntry;

typedef struct {
    int64_t refcount;
    int64_t size;
    int64_t capacity;
    DictEntry* entries;
} Dict;

// Forward declarations
void dict_set(Dict* dct, void* key, void* value);

int64_t dict_hash_ptr(void* ptr) {
    return (int64_t)ptr;
}

Dict* dict_new() {
    Dict* dct = (Dict*)malloc(sizeof(Dict));
    dct->refcount = 1;
    dct->size = 0;
    dct->capacity = 16;
    dct->entries = (DictEntry*)calloc(dct->capacity, sizeof(DictEntry));
    return dct;
}

void dict_incref(Dict* dct) {
    if (dct) dct->refcount++;
}

void dict_decref(Dict* dct) {
    if (dct && --dct->refcount == 0) {
        free(dct->entries);
        free(dct);
    }
}

int64_t dict_len(Dict* dct) {
    return dct->size;
}

void dict_resize(Dict* dct) {
    if (dct->size >= dct->capacity * 0.75) {
        int64_t old_capacity = dct->capacity;
        DictEntry* old_entries = dct->entries;
        
        dct->capacity *= 2;
        dct->entries = (DictEntry*)calloc(dct->capacity, sizeof(DictEntry));
        dct->size = 0;
        
        for (int64_t i = 0; i < old_capacity; i++) {
            if (old_entries[i].occupied) {
                dict_set(dct, old_entries[i].key, old_entries[i].value);
            }
        }
        free(old_entries);
    }
}

void dict_set(Dict* dct, void* key, void* value) {
    dict_resize(dct);
    
    int64_t hash = dict_hash_ptr(key);
    int64_t index = hash % dct->capacity;
    
    while (dct->entries[index].occupied && dct->entries[index].key != key) {
        index = (index + 1) % dct->capacity;
    }
    
    if (!dct->entries[index].occupied) {
        dct->size++;
    }
    
    dct->entries[index].key = key;
    dct->entries[index].value = value;
    dct->entries[index].hash = hash;
    dct->entries[index].occupied = 1;
}

void* dict_get(Dict* dct, void* key) {
    int64_t hash = dict_hash_ptr(key);
    int64_t index = hash % dct->capacity;
    
    while (dct->entries[index].occupied) {
        if (dct->entries[index].key == key) {
            return dct->entries[index].value;
        }
        index = (index + 1) % dct->capacity;
    }
    return NULL;
}

int32_t dict_contains(Dict* dct, void* key) {
    return dict_get(dct, key) != NULL;
}
