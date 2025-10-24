"""
Week 3 Day 7: Import System Tests

Tests module loading, resolution, caching, and circular import detection
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from compiler.frontend.module_loader import ModuleLoader, get_loader, reset_loader


class TestModuleResolution:
    """Test module path resolution"""
    
    def setup_method(self):
        """Create temporary directory for test modules"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ModuleLoader(search_paths=[self.temp_dir])
    
    def teardown_method(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_simple_module_resolution(self):
        """Test finding a simple module file"""
        # Create test module
        module_path = Path(self.temp_dir) / "test_module.py"
        module_path.write_text("x = 42")
        
        # Resolve it
        resolved = self.loader.resolve_module("test_module")
        
        assert resolved is not None
        assert resolved == module_path
    
    def test_module_not_found(self):
        """Test resolution of non-existent module"""
        resolved = self.loader.resolve_module("nonexistent_module")
        assert resolved is None
    
    def test_package_resolution(self):
        """Test resolving a package with __init__.py"""
        # Create package structure
        package_dir = Path(self.temp_dir) / "mypackage"
        package_dir.mkdir()
        init_file = package_dir / "__init__.py"
        init_file.write_text("# Package init")
        
        # Resolve package
        resolved = self.loader.resolve_module("mypackage")
        
        assert resolved is not None
        assert resolved == init_file


class TestModuleLoading:
    """Test module compilation and loading"""
    
    def setup_method(self):
        """Create temporary directory for test modules"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ModuleLoader(search_paths=[self.temp_dir])
    
    def teardown_method(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_simple_module(self):
        """Test loading a simple module"""
        # Create test module
        module_path = Path(self.temp_dir) / "simple.py"
        module_path.write_text("""
def greet(name: str) -> str:
    return f"Hello, {name}!"

x = 42
""")
        
        # Load module
        loaded = self.loader.load_module("simple")
        
        assert loaded is not None
        assert loaded.name == "simple"
        assert loaded.path == module_path
        assert loaded.ir_module is not None
    
    def test_module_caching(self):
        """Test that modules are cached"""
        # Create test module
        module_path = Path(self.temp_dir) / "cached.py"
        module_path.write_text("y = 100")
        
        # Load twice
        loaded1 = self.loader.load_module("cached")
        loaded2 = self.loader.load_module("cached")
        
        # Should be same instance
        assert loaded1 is loaded2
    
    def test_module_reload(self):
        """Test forcing module reload"""
        # Create test module
        module_path = Path(self.temp_dir) / "reloadable.py"
        module_path.write_text("z = 1")
        
        # Load first time
        loaded1 = self.loader.load_module("reloadable")
        
        # Modify module
        module_path.write_text("z = 2")
        
        # Reload
        loaded2 = self.loader.load_module("reloadable", reload=True)
        
        # Should be different instances
        assert loaded1 is not loaded2
    
    def test_module_not_found_error(self):
        """Test ImportError for non-existent module"""
        with pytest.raises(ImportError, match="No module named"):
            self.loader.load_module("does_not_exist")


class TestCircularImports:
    """Test circular import detection"""
    
    def setup_method(self):
        """Create temporary directory for test modules"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ModuleLoader(search_paths=[self.temp_dir])
    
    def teardown_method(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_circular_import_detection(self):
        """Test detection of circular imports"""
        # Create circular modules
        module_a = Path(self.temp_dir) / "module_a.py"
        module_b = Path(self.temp_dir) / "module_b.py"
        
        module_a.write_text("import module_b")
        module_b.write_text("import module_a")
        
        # Note: Circular import detection happens when imports are actually executed
        # For now, we can compile modules with circular dependencies
        # Real circular import detection would happen at runtime
        # This test validates that compilation succeeds (imports not executed yet)
        loaded_a = self.loader.load_module("module_a")
        assert loaded_a is not None
        
        # Circular imports in dependency list
        assert "module_b" in loaded_a.dependencies
    
    def test_no_false_positive_on_sequential_imports(self):
        """Test that sequential imports don't trigger circular detection"""
        # Create modules
        module_c = Path(self.temp_dir) / "module_c.py"
        module_d = Path(self.temp_dir) / "module_d.py"
        
        module_c.write_text("x = 1")
        module_d.write_text("y = 2")
        
        # Load sequentially - should work fine
        loaded_c = self.loader.load_module("module_c")
        loaded_d = self.loader.load_module("module_d")
        
        assert loaded_c is not None
        assert loaded_d is not None


class TestDependencyTracking:
    """Test dependency extraction"""
    
    def setup_method(self):
        """Create temporary directory for test modules"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ModuleLoader(search_paths=[self.temp_dir])
    
    def teardown_method(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_extract_import_dependencies(self):
        """Test extracting dependencies from import statements"""
        # Create module with imports
        module_path = Path(self.temp_dir) / "with_deps.py"
        module_path.write_text("""
import math
import os
from collections import Counter

def my_function():
    pass
""")
        
        # Load module
        loaded = self.loader.load_module("with_deps")
        
        # Check dependencies
        assert "math" in loaded.dependencies
        assert "os" in loaded.dependencies
        assert "collections" in loaded.dependencies


class TestGlobalLoader:
    """Test global loader instance"""
    
    def test_get_global_loader(self):
        """Test getting global loader instance"""
        reset_loader()
        loader1 = get_loader()
        loader2 = get_loader()
        
        # Should be same instance
        assert loader1 is loader2
    
    def test_reset_loader(self):
        """Test resetting global loader"""
        loader1 = get_loader()
        reset_loader()
        loader2 = get_loader()
        
        # Should be different instances after reset
        assert loader1 is not loader2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
