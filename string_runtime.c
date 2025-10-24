
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

// String structure
typedef struct {
    int64_t refcount;
    int64_t length;
    int64_t hash;
    int32_t interned;
    char data[];
} String;

// Create new string
String* str_new(const char* data, int64_t length) {
    String* s = (String*)malloc(sizeof(String) + length + 1);
    s->refcount = 1;
    s->length = length;
    s->hash = -1;  // Not computed yet
    s->interned = 0;
    memcpy(s->data, data, length);
    s->data[length] = '\0';
    return s;
}

// Increment reference count
void str_incref(String* s) {
    if (s) s->refcount++;
}

// Decrement reference count and free if zero
void str_decref(String* s) {
    if (s && --s->refcount == 0) {
        free(s);
    }
}

// Get string length
int64_t str_len(String* s) {
    return s->length;
}

// Concatenate two strings
String* str_concat(String* s1, String* s2) {
    int64_t new_length = s1->length + s2->length;
    String* result = (String*)malloc(sizeof(String) + new_length + 1);
    result->refcount = 1;
    result->length = new_length;
    result->hash = -1;
    result->interned = 0;
    memcpy(result->data, s1->data, s1->length);
    memcpy(result->data + s1->length, s2->data, s2->length);
    result->data[new_length] = '\0';
    return result;
}

// Slice string
String* str_slice(String* s, int64_t start, int64_t end) {
    if (start < 0) start = 0;
    if (end > s->length) end = s->length;
    if (start >= end) return str_new("", 0);
    
    int64_t length = end - start;
    return str_new(s->data + start, length);
}

// Convert to uppercase
String* str_upper(String* s) {
    String* result = str_new(s->data, s->length);
    for (int64_t i = 0; i < result->length; i++) {
        result->data[i] = toupper(result->data[i]);
    }
    return result;
}

// Convert to lowercase
String* str_lower(String* s) {
    String* result = str_new(s->data, s->length);
    for (int64_t i = 0; i < result->length; i++) {
        result->data[i] = tolower(result->data[i]);
    }
    return result;
}

// Find substring
int64_t str_find(String* haystack, String* needle) {
    if (needle->length == 0) return 0;
    if (needle->length > haystack->length) return -1;
    
    for (int64_t i = 0; i <= haystack->length - needle->length; i++) {
        if (memcmp(haystack->data + i, needle->data, needle->length) == 0) {
            return i;
        }
    }
    return -1;
}

// Replace substring
String* str_replace(String* s, String* old, String* new) {
    int64_t pos = str_find(s, old);
    if (pos == -1) {
        str_incref(s);
        return s;
    }
    
    // For simplicity, replace only first occurrence
    int64_t new_length = s->length - old->length + new->length;
    String* result = (String*)malloc(sizeof(String) + new_length + 1);
    result->refcount = 1;
    result->length = new_length;
    result->hash = -1;
    result->interned = 0;
    
    memcpy(result->data, s->data, pos);
    memcpy(result->data + pos, new->data, new->length);
    memcpy(result->data + pos + new->length, 
           s->data + pos + old->length, 
           s->length - pos - old->length);
    result->data[new_length] = '\0';
    
    return result;
}

// Strip whitespace
String* str_strip(String* s) {
    int64_t start = 0;
    int64_t end = s->length;
    
    while (start < end && isspace(s->data[start])) start++;
    while (end > start && isspace(s->data[end - 1])) end--;
    
    return str_slice(s, start, end);
}

// Check if starts with prefix
int32_t str_startswith(String* s, String* prefix) {
    if (prefix->length > s->length) return 0;
    return memcmp(s->data, prefix->data, prefix->length) == 0;
}

// Check if ends with suffix
int32_t str_endswith(String* s, String* suffix) {
    if (suffix->length > s->length) return 0;
    return memcmp(s->data + s->length - suffix->length, 
                  suffix->data, suffix->length) == 0;
}

// Compute hash (FNV-1a)
int64_t str_hash(String* s) {
    if (s->hash != -1) return s->hash;
    
    uint64_t hash = 14695981039346656037ULL;
    for (int64_t i = 0; i < s->length; i++) {
        hash ^= (uint64_t)s->data[i];
        hash *= 1099511628211ULL;
    }
    
    s->hash = (int64_t)hash;
    return s->hash;
}
