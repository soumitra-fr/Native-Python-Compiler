"""
Phase 5: Pandas Interface
Provides Pandas DataFrame and Series support in compiled code.
"""

from llvmlite import ir

class PandasInterface:
    """
    Handles Pandas integration in compiled code.
    
    Features:
    - DataFrame and Series support
    - Column operations
    - Row indexing and selection
    - Data manipulation (merge, groupby, pivot)
    - CSV/Parquet I/O
    """
    
    def __init__(self, c_extension_interface, numpy_interface):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.double = ir.DoubleType()
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        self.c_ext = c_extension_interface
        self.numpy = numpy_interface
        
        # DataFrame structure
        self.dataframe_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # data (dict of columns)
            self.void_ptr,        # index (array)
            self.void_ptr,        # columns (array of strings)
            self.int32,           # num_rows
            self.int32,           # num_cols
        ])
        
        # Series structure
        self.series_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # data (array)
            self.void_ptr,        # index (array)
            self.char_ptr,        # name
            self.int32,           # length
            self.int32,           # dtype
        ])
    
    def create_dataframe(self, builder, module, data_dict):
        """
        Create a Pandas DataFrame.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            data_dict: Dictionary mapping column names to arrays
        
        Returns:
            Pointer to DataFrame structure
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.int32])
        func = ir.Function(module, func_type, name="pandas_create_dataframe")
        
        num_cols = ir.Constant(self.int32, len(data_dict))
        data_ptr = self._create_dict_ptr(builder, module, data_dict)
        
        result = builder.call(func, [data_ptr, num_cols])
        
        return result
    
    def dataframe_getitem(self, builder, module, df_ptr, column_name):
        """
        Get column from DataFrame (df['column']).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df_ptr: Pointer to DataFrame
            column_name: Column name string
        
        Returns:
            Series pointer
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="pandas_getitem")
        
        col_str = self._create_string_literal(builder, module, column_name)
        df_void_ptr = builder.bitcast(df_ptr, self.void_ptr)
        
        result = builder.call(func, [df_void_ptr, col_str])
        
        return result
    
    def dataframe_setitem(self, builder, module, df_ptr, column_name, series_ptr):
        """
        Set column in DataFrame (df['column'] = series).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df_ptr: Pointer to DataFrame
            column_name: Column name string
            series_ptr: Series to set
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.char_ptr, self.void_ptr]
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        func = ir.Function(module, func_type, name="pandas_setitem")
        
        col_str = self._create_string_literal(builder, module, column_name)
        df_void_ptr = builder.bitcast(df_ptr, self.void_ptr)
        series_void_ptr = builder.bitcast(series_ptr, self.void_ptr)
        
        builder.call(func, [df_void_ptr, col_str, series_void_ptr])
    
    def dataframe_iloc(self, builder, module, df_ptr, row_index):
        """
        Integer-location based indexing (df.iloc[i]).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df_ptr: Pointer to DataFrame
            row_index: Row index
        
        Returns:
            Series (row) pointer
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.int32])
        func = ir.Function(module, func_type, name="pandas_iloc")
        
        df_void_ptr = builder.bitcast(df_ptr, self.void_ptr)
        row_idx = ir.Constant(self.int32, row_index)
        
        result = builder.call(func, [df_void_ptr, row_idx])
        
        return result
    
    def dataframe_loc(self, builder, module, df_ptr, label):
        """
        Label-based indexing (df.loc[label]).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df_ptr: Pointer to DataFrame
            label: Index label
        
        Returns:
            Series (row) pointer
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="pandas_loc")
        
        label_str = self._create_string_literal(builder, module, label)
        df_void_ptr = builder.bitcast(df_ptr, self.void_ptr)
        
        result = builder.call(func, [df_void_ptr, label_str])
        
        return result
    
    def dataframe_merge(self, builder, module, df1_ptr, df2_ptr, on_column, how='inner'):
        """
        Merge two DataFrames (pd.merge).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df1_ptr: First DataFrame
            df2_ptr: Second DataFrame
            on_column: Column to merge on
            how: Merge type ('inner', 'outer', 'left', 'right')
        
        Returns:
            Merged DataFrame pointer
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.void_ptr, self.char_ptr, self.char_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="pandas_merge")
        
        df1_void_ptr = builder.bitcast(df1_ptr, self.void_ptr)
        df2_void_ptr = builder.bitcast(df2_ptr, self.void_ptr)
        on_str = self._create_string_literal(builder, module, on_column)
        how_str = self._create_string_literal(builder, module, how)
        
        result = builder.call(func, [df1_void_ptr, df2_void_ptr, on_str, how_str])
        
        return result
    
    def dataframe_groupby(self, builder, module, df_ptr, by_column):
        """
        Group DataFrame by column (df.groupby).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df_ptr: Pointer to DataFrame
            by_column: Column to group by
        
        Returns:
            GroupBy object pointer
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="pandas_groupby")
        
        by_str = self._create_string_literal(builder, module, by_column)
        df_void_ptr = builder.bitcast(df_ptr, self.void_ptr)
        
        result = builder.call(func, [df_void_ptr, by_str])
        
        return result
    
    def groupby_aggregate(self, builder, module, groupby_ptr, agg_func):
        """
        Aggregate grouped data (groupby.sum(), .mean(), etc.).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            groupby_ptr: GroupBy object
            agg_func: Aggregation function name ('sum', 'mean', 'count')
        
        Returns:
            Aggregated DataFrame
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="pandas_groupby_aggregate")
        
        agg_str = self._create_string_literal(builder, module, agg_func)
        groupby_void_ptr = builder.bitcast(groupby_ptr, self.void_ptr)
        
        result = builder.call(func, [groupby_void_ptr, agg_str])
        
        return result
    
    def read_csv(self, builder, module, filepath):
        """
        Read CSV file (pd.read_csv).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            filepath: Path to CSV file
        
        Returns:
            DataFrame pointer
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.char_ptr])
        func = ir.Function(module, func_type, name="pandas_read_csv")
        
        path_str = self._create_string_literal(builder, module, filepath)
        result = builder.call(func, [path_str])
        
        return result
    
    def to_csv(self, builder, module, df_ptr, filepath):
        """
        Write DataFrame to CSV (df.to_csv).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df_ptr: Pointer to DataFrame
            filepath: Path to output CSV file
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="pandas_to_csv")
        
        df_void_ptr = builder.bitcast(df_ptr, self.void_ptr)
        path_str = self._create_string_literal(builder, module, filepath)
        
        builder.call(func, [df_void_ptr, path_str])
    
    def dataframe_describe(self, builder, module, df_ptr):
        """
        Statistical summary (df.describe()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df_ptr: Pointer to DataFrame
        
        Returns:
            Summary DataFrame
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="pandas_describe")
        
        df_void_ptr = builder.bitcast(df_ptr, self.void_ptr)
        result = builder.call(func, [df_void_ptr])
        
        return result
    
    def dataframe_head(self, builder, module, df_ptr, n=5):
        """
        Get first n rows (df.head()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            df_ptr: Pointer to DataFrame
            n: Number of rows
        
        Returns:
            DataFrame with first n rows
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.int32])
        func = ir.Function(module, func_type, name="pandas_head")
        
        df_void_ptr = builder.bitcast(df_ptr, self.void_ptr)
        n_val = ir.Constant(self.int32, n)
        
        result = builder.call(func, [df_void_ptr, n_val])
        
        return result
    
    def series_sum(self, builder, module, series_ptr):
        """
        Sum Series values (series.sum()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            series_ptr: Pointer to Series
        
        Returns:
            Sum value
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.double, [self.void_ptr])
        func = ir.Function(module, func_type, name="pandas_series_sum")
        
        series_void_ptr = builder.bitcast(series_ptr, self.void_ptr)
        result = builder.call(func, [series_void_ptr])
        
        return result
    
    def series_mean(self, builder, module, series_ptr):
        """
        Mean of Series values (series.mean()).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            series_ptr: Pointer to Series
        
        Returns:
            Mean value
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.double, [self.void_ptr])
        func = ir.Function(module, func_type, name="pandas_series_mean")
        
        series_void_ptr = builder.bitcast(series_ptr, self.void_ptr)
        result = builder.call(func, [series_void_ptr])
        
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
    
    def _create_dict_ptr(self, builder, module, data_dict):
        """Create a pointer to dictionary data."""
        # Simplified - would actually serialize dict
        return ir.Constant(self.void_ptr, None)


def generate_pandas_runtime():
    """Generate C runtime code for Pandas interface."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Simplified DataFrame structure
typedef struct DataFrame {
    int64_t refcount;
    void* data;          // dict of columns
    void* index;         // array
    void* columns;       // array of strings
    int32_t num_rows;
    int32_t num_cols;
} DataFrame;

// Simplified Series structure
typedef struct Series {
    int64_t refcount;
    void* data;          // array
    void* index;         // array
    char* name;
    int32_t length;
    int32_t dtype;
} Series;

// Create DataFrame
void* pandas_create_dataframe(void* data_dict, int32_t num_cols) {
    DataFrame* df = (DataFrame*)malloc(sizeof(DataFrame));
    df->refcount = 1;
    df->data = data_dict;
    df->num_cols = num_cols;
    df->num_rows = 0;  // Would calculate from data
    return df;
}

// Get column
void* pandas_getitem(void* df_ptr, char* column_name) {
    // Would extract column from DataFrame
    return NULL;
}

// Set column
void pandas_setitem(void* df_ptr, char* column_name, void* series_ptr) {
    // Would set column in DataFrame
}

// Integer-location indexing
void* pandas_iloc(void* df_ptr, int32_t row_index) {
    // Would get row by index
    return NULL;
}

// Label-based indexing
void* pandas_loc(void* df_ptr, char* label) {
    // Would get row by label
    return NULL;
}

// Merge DataFrames
void* pandas_merge(void* df1_ptr, void* df2_ptr, char* on_column, char* how) {
    // Would perform merge operation
    return NULL;
}

// Group by
void* pandas_groupby(void* df_ptr, char* by_column) {
    // Would create GroupBy object
    return NULL;
}

// Aggregate grouped data
void* pandas_groupby_aggregate(void* groupby_ptr, char* agg_func) {
    // Would perform aggregation
    return NULL;
}

// Read CSV
void* pandas_read_csv(char* filepath) {
    // Would read CSV file into DataFrame
    return NULL;
}

// Write CSV
void pandas_to_csv(void* df_ptr, char* filepath) {
    // Would write DataFrame to CSV
}

// Describe
void* pandas_describe(void* df_ptr) {
    // Would generate statistical summary
    return NULL;
}

// Head
void* pandas_head(void* df_ptr, int32_t n) {
    // Would return first n rows
    return NULL;
}

// Series sum
double pandas_series_sum(void* series_ptr) {
    // Would sum series values
    return 0.0;
}

// Series mean
double pandas_series_mean(void* series_ptr) {
    // Would calculate mean
    return 0.0;
}
"""
    
    # Write to file
    with open('pandas_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Pandas runtime generated: pandas_runtime.c")


if __name__ == "__main__":
    from c_extension_interface import CExtensionInterface
    from numpy_interface import NumPyInterface
    
    # Generate runtime C code
    generate_pandas_runtime()
    
    # Test Pandas interface
    c_ext = CExtensionInterface()
    numpy_int = NumPyInterface(c_ext)
    pandas_interface = PandasInterface(c_ext, numpy_int)
    
    print(f"✅ PandasInterface initialized")
    print(f"   - DataFrame structure: {pandas_interface.dataframe_type}")
    print(f"   - Series structure: {pandas_interface.series_type}")
