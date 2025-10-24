"""
Phase 5 Tests: C Extension Interface
Comprehensive test suite for NumPy and Pandas support.
"""

import unittest
import os
import sys
from llvmlite import ir

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compiler.runtime.phase5_c_extensions import Phase5CExtensions


class TestPhase5CExtensions(unittest.TestCase):
    """Test Phase 5: C Extension Interface"""
    
    def setUp(self):
        """Set up test fixtures."""
        self.phase5 = Phase5CExtensions()
        self.module = ir.Module(name="test_c_ext")
        
        # Create a basic function for testing
        func_type = ir.FunctionType(ir.VoidType(), [])
        self.func = ir.Function(self.module, func_type, name="test_func")
        self.block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.block)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.builder.ret_void()
    
    # C Extension Tests
    
    def test_pyobject_structure(self):
        """Test PyObject structure definition"""
        pyobj_type = self.phase5.c_extension.pyobject_type
        self.assertIsNotNone(pyobj_type)
        # PyObject: {refcount, type_ptr}
        self.assertEqual(len(pyobj_type.elements), 2)
        print("✅ test_pyobject_structure")
    
    def test_cfunction_structure(self):
        """Test CFunction structure definition"""
        cfunc_type = self.phase5.c_extension.cfunction_type
        self.assertIsNotNone(cfunc_type)
        # CFunction: {refcount, function_ptr, name, num_args, flags}
        self.assertEqual(len(cfunc_type.elements), 5)
        print("✅ test_cfunction_structure")
    
    # NumPy Array Tests
    
    def test_numpy_array_structure(self):
        """Test NumPy array structure definition"""
        ndarray_type = self.phase5.numpy.ndarray_type
        self.assertIsNotNone(ndarray_type)
        # NDArray: {refcount, data, ndim, shape, strides, dtype, itemsize, size}
        self.assertEqual(len(ndarray_type.elements), 8)
        print("✅ test_numpy_array_structure")
    
    def test_numpy_create_array_1d(self):
        """Test 1D array creation: np.zeros(100)"""
        arr = self.phase5.numpy_create_array(self.builder, self.module, (100,), 'float64')
        self.assertIsNotNone(arr)
        print("✅ test_numpy_create_array_1d")
    
    def test_numpy_create_array_2d(self):
        """Test 2D array creation: np.zeros((10, 20))"""
        arr = self.phase5.numpy_create_array(self.builder, self.module, (10, 20), 'float64')
        self.assertIsNotNone(arr)
        print("✅ test_numpy_create_array_2d")
    
    def test_numpy_create_array_3d(self):
        """Test 3D array creation: np.zeros((5, 10, 15))"""
        arr = self.phase5.numpy_create_array(self.builder, self.module, (5, 10, 15), 'float64')
        self.assertIsNotNone(arr)
        print("✅ test_numpy_create_array_3d")
    
    def test_numpy_array_indexing(self):
        """Test array indexing: arr[i, j]"""
        arr = self.phase5.numpy_create_array(self.builder, self.module, (10, 10))
        value = self.phase5.numpy_getitem(self.builder, self.module, arr, [5, 3])
        self.assertIsNotNone(value)
        print("✅ test_numpy_array_indexing")
    
    def test_numpy_array_assignment(self):
        """Test array assignment: arr[i, j] = value"""
        arr = self.phase5.numpy_create_array(self.builder, self.module, (10, 10))
        value = ir.Constant(ir.DoubleType(), 42.0)
        self.phase5.numpy_setitem(self.builder, self.module, arr, [5, 3], value)
        print("✅ test_numpy_array_assignment")
    
    def test_numpy_array_sum(self):
        """Test array sum: np.sum(arr)"""
        arr = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        result = self.phase5.numpy_sum(self.builder, self.module, arr)
        self.assertIsNotNone(result)
        print("✅ test_numpy_array_sum")
    
    def test_numpy_array_reshape(self):
        """Test array reshape: arr.reshape((5, 20))"""
        arr = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        reshaped = self.phase5.numpy_reshape(self.builder, self.module, arr, (5, 20))
        self.assertIsNotNone(reshaped)
        print("✅ test_numpy_array_reshape")
    
    def test_numpy_matrix_dot(self):
        """Test matrix multiplication: np.dot(a, b)"""
        mat1 = self.phase5.numpy_create_array(self.builder, self.module, (10, 5))
        mat2 = self.phase5.numpy_create_array(self.builder, self.module, (5, 10))
        result = self.phase5.numpy_dot(self.builder, self.module, mat1, mat2)
        self.assertIsNotNone(result)
        print("✅ test_numpy_matrix_dot")
    
    def test_numpy_ufunc_add(self):
        """Test ufunc: np.add(arr1, arr2)"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        arr2 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        result = self.phase5.numpy_ufunc(self.builder, self.module, "add", [arr1, arr2])
        self.assertIsNotNone(result)
        print("✅ test_numpy_ufunc_add")
    
    def test_numpy_ufunc_multiply(self):
        """Test ufunc: np.multiply(arr1, arr2)"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        arr2 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        result = self.phase5.numpy_ufunc(self.builder, self.module, "multiply", [arr1, arr2])
        self.assertIsNotNone(result)
        print("✅ test_numpy_ufunc_multiply")
    
    # Pandas DataFrame Tests
    
    def test_pandas_dataframe_structure(self):
        """Test DataFrame structure definition"""
        df_type = self.phase5.pandas.dataframe_type
        self.assertIsNotNone(df_type)
        # DataFrame: {refcount, data, index, columns, num_rows, num_cols}
        self.assertEqual(len(df_type.elements), 6)
        print("✅ test_pandas_dataframe_structure")
    
    def test_pandas_series_structure(self):
        """Test Series structure definition"""
        series_type = self.phase5.pandas.series_type
        self.assertIsNotNone(series_type)
        # Series: {refcount, data, index, name, length, dtype}
        self.assertEqual(len(series_type.elements), 6)
        print("✅ test_pandas_series_structure")
    
    def test_pandas_create_dataframe(self):
        """Test DataFrame creation: pd.DataFrame({'A': [...], 'B': [...]})"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        arr2 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'A': arr1, 'B': arr2})
        self.assertIsNotNone(df)
        print("✅ test_pandas_create_dataframe")
    
    def test_pandas_getitem(self):
        """Test column access: df['column']"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'price': arr1})
        series = self.phase5.pandas_getitem(self.builder, self.module, df, 'price')
        self.assertIsNotNone(series)
        print("✅ test_pandas_getitem")
    
    def test_pandas_setitem(self):
        """Test column assignment: df['column'] = series"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'A': arr1})
        arr2 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        self.phase5.pandas_setitem(self.builder, self.module, df, 'B', arr2)
        print("✅ test_pandas_setitem")
    
    def test_pandas_iloc(self):
        """Test integer indexing: df.iloc[i]"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'A': arr1})
        row = self.phase5.pandas_iloc(self.builder, self.module, df, 50)
        self.assertIsNotNone(row)
        print("✅ test_pandas_iloc")
    
    def test_pandas_merge(self):
        """Test merge: pd.merge(df1, df2, on='key')"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        df1 = self.phase5.pandas_create_dataframe(self.builder, self.module, {'id': arr1})
        df2 = self.phase5.pandas_create_dataframe(self.builder, self.module, {'id': arr1})
        merged = self.phase5.pandas_merge(self.builder, self.module, df1, df2, 'id', 'inner')
        self.assertIsNotNone(merged)
        print("✅ test_pandas_merge")
    
    def test_pandas_groupby(self):
        """Test groupby: df.groupby('column')"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'category': arr1})
        grouped = self.phase5.pandas_groupby(self.builder, self.module, df, 'category')
        self.assertIsNotNone(grouped)
        print("✅ test_pandas_groupby")
    
    def test_pandas_aggregate(self):
        """Test aggregation: grouped.sum()"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'value': arr1})
        grouped = self.phase5.pandas_groupby(self.builder, self.module, df, 'value')
        result = self.phase5.pandas_aggregate(self.builder, self.module, grouped, 'sum')
        self.assertIsNotNone(result)
        print("✅ test_pandas_aggregate")
    
    def test_pandas_read_csv(self):
        """Test CSV reading: pd.read_csv('file.csv')"""
        df = self.phase5.pandas_read_csv(self.builder, self.module, 'data.csv')
        self.assertIsNotNone(df)
        print("✅ test_pandas_read_csv")
    
    def test_pandas_to_csv(self):
        """Test CSV writing: df.to_csv('file.csv')"""
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'A': arr1})
        self.phase5.pandas_to_csv(self.builder, self.module, df, 'output.csv')
        print("✅ test_pandas_to_csv")
    
    # Integration Tests
    
    def test_numpy_pandas_integration(self):
        """Test NumPy and Pandas working together"""
        # Create NumPy array
        arr1 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        arr2 = self.phase5.numpy_create_array(self.builder, self.module, (100,))
        
        # Create DataFrame from arrays
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'X': arr1, 'Y': arr2})
        
        # Get column back as Series
        series = self.phase5.pandas_getitem(self.builder, self.module, df, 'X')
        
        self.assertIsNotNone(df)
        self.assertIsNotNone(series)
        print("✅ test_numpy_pandas_integration")
    
    def test_llvm_ir_generation(self):
        """Test that LLVM IR is generated correctly"""
        # Generate some operations
        arr = self.phase5.numpy_create_array(self.builder, self.module, (10, 10))
        df = self.phase5.pandas_create_dataframe(self.builder, self.module, {'A': arr})
        
        # Check that IR was generated
        ir_str = str(self.module)
        self.assertIsNotNone(ir_str)
        self.assertGreater(len(ir_str), 0)
        self.assertIn("numpy_create_array", ir_str)
        self.assertIn("pandas_create_dataframe", ir_str)
        print("✅ test_llvm_ir_generation")


def run_tests():
    """Run all Phase 5 tests."""
    print("=" * 60)
    print("PHASE 5 TEST SUITE: C Extension Interface")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase5CExtensions)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n⚠️  Some tests need runtime C extension loading")
    
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_tests()
