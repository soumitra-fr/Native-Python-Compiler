"""
Tests for Module Cache System
Week 4: Persistent .pym caching

Tests:
- Cache creation and retrieval
- Staleness detection
- Invalidation
- Cleanup operations
- Cache statistics
"""

import os
import time
import tempfile
import pytest
from pathlib import Path
from compiler.frontend.module_cache import ModuleCache, CachedModule


class TestModuleCache:
    """Test module cache functionality"""
    
    def setup_method(self):
        """Create temporary directory for cache tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = ModuleCache(cache_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, name: str, content: str = "# test") -> Path:
        """Helper to create a test Python file"""
        file_path = Path(self.temp_dir) / f"{name}.py"
        file_path.write_text(content)
        return file_path
    
    def test_cache_creation(self):
        """Test cache initialization"""
        assert self.cache.cache_dir == self.temp_dir
        assert len(self.cache.cache) == 0
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache"""
        # Create a test file
        test_file = self.create_test_file("test_module")
        
        # Store in cache
        ir_data = {'name': 'test_module', 'functions': []}
        llvm_ir = "define i32 @main() { ret i32 0 }"
        dependencies = []
        
        self.cache.put(str(test_file), ir_data, llvm_ir, dependencies)
        
        # Retrieve from cache
        cached = self.cache.get(str(test_file))
        
        assert cached is not None
        assert cached.source_file == str(test_file.resolve())
        assert cached.ir_module_json == ir_data
        assert cached.llvm_ir == llvm_ir
        assert cached.dependencies == dependencies
    
    def test_cache_persistence(self):
        """Test that cache persists to disk"""
        test_file = self.create_test_file("persist_test")
        
        # Store in cache
        self.cache.put(
            str(test_file),
            {'name': 'persist_test'},
            "llvm ir here",
            []
        )
        
        # Create new cache instance (simulates restart)
        new_cache = ModuleCache(cache_dir=self.temp_dir)
        
        # Should load from disk
        cached = new_cache.get(str(test_file))
        assert cached is not None
        assert cached.ir_module_json == {'name': 'persist_test'}
    
    def test_staleness_detection_source_modified(self):
        """Test that cache detects when source file is modified"""
        test_file = self.create_test_file("stale_test", "# version 1")
        
        # Cache the file
        self.cache.put(str(test_file), {'v': 1}, "llvm1", [])
        
        # Verify cached
        cached = self.cache.get(str(test_file))
        assert cached is not None
        
        # Modify the source file
        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("# version 2")
        
        # Should detect staleness
        cached = self.cache.get(str(test_file))
        assert cached is None  # Cache invalidated
    
    def test_staleness_detection_dependency_modified(self):
        """Test that cache detects when dependency is modified"""
        # Create dependency first
        dep_file = self.create_test_file("dep", "# dep v1")
        time.sleep(0.01)  # Ensure different mtime
        
        # Then create main file
        main_file = self.create_test_file("main", "import dep")
        
        # Cache main with dep as dependency
        self.cache.put(
            str(main_file),
            {'name': 'main'},
            "llvm",
            [str(dep_file)]
        )
        
        # Verify cached
        cached = self.cache.get(str(main_file))
        assert cached is not None
        
        # Modify dependency
        time.sleep(0.01)
        dep_file.write_text("# dep v2")
        
        # Should detect staleness
        cached = self.cache.get(str(main_file))
        assert cached is None
    
    def test_invalidation(self):
        """Test manual cache invalidation"""
        test_file = self.create_test_file("invalidate_test")
        
        # Cache the file
        self.cache.put(str(test_file), {'name': 'test'}, "llvm", [])
        
        # Verify cached
        assert self.cache.get(str(test_file)) is not None
        
        # Invalidate
        self.cache.invalidate(str(test_file))
        
        # Should be gone
        assert self.cache.get(str(test_file)) is None
    
    def test_clear_all(self):
        """Test clearing all cache"""
        # Cache multiple files
        for i in range(3):
            test_file = self.create_test_file(f"clear_test_{i}")
            self.cache.put(str(test_file), {'id': i}, f"llvm{i}", [])
        
        # Verify all cached
        assert len(self.cache.cache) == 3
        
        # Clear all
        count = self.cache.clear_all()
        assert count == 3
        assert len(self.cache.cache) == 0
    
    def test_cleanup_stale(self):
        """Test cleaning up stale cache files"""
        # Create and cache a file
        test_file = self.create_test_file("cleanup_test")
        self.cache.put(str(test_file), {'name': 'test'}, "llvm", [])
        
        # Make it stale by modifying source
        time.sleep(0.01)
        test_file.write_text("# modified")
        
        # Clear memory cache so cleanup finds disk cache
        self.cache.cache.clear()
        
        # Cleanup stale
        count = self.cache.cleanup_stale()
        assert count == 1
    
    def test_cache_stats(self):
        """Test cache statistics"""
        # Initial stats
        stats = self.cache.get_stats()
        assert stats['memory_cached'] == 0
        assert stats['disk_cached'] == 0
        
        # Cache some files
        for i in range(2):
            test_file = self.create_test_file(f"stats_test_{i}")
            self.cache.put(str(test_file), {'id': i}, f"llvm{i}", [])
        
        stats = self.cache.get_stats()
        assert stats['memory_cached'] == 2
        assert stats['disk_cached'] == 2
        assert stats['total_size_bytes'] > 0
    
    def test_missing_source_file(self):
        """Test handling when source file is deleted"""
        test_file = self.create_test_file("delete_test")
        
        # Cache the file
        self.cache.put(str(test_file), {'name': 'test'}, "llvm", [])
        
        # Delete source file
        test_file.unlink()
        
        # Should detect as stale
        cached = self.cache.get(str(test_file))
        assert cached is None
    
    def test_corrupted_cache_file(self):
        """Test handling of corrupted cache file"""
        test_file = self.create_test_file("corrupt_test")
        
        # Create corrupted cache file
        cache_path = self.cache._get_cache_path(str(test_file))
        cache_path.write_text("not valid json {{{")
        
        # Should handle gracefully
        cached = self.cache.get(str(test_file))
        assert cached is None
        
        # Corrupted file should be removed
        assert not cache_path.exists()
    
    def test_cache_path_generation(self):
        """Test cache path generation"""
        test_file = self.create_test_file("path_test")
        
        cache_path = self.cache._get_cache_path(str(test_file))
        
        # Should be in cache directory
        assert cache_path.parent == Path(self.temp_dir)
        # Should have .pym extension
        assert cache_path.suffix == ".pym"
        # Should have module name
        assert "path_test" in cache_path.name
    
    def test_multiple_dependencies(self):
        """Test caching with multiple dependencies"""
        # Create dependencies first
        dep1 = self.create_test_file("dep1")
        dep2 = self.create_test_file("dep2")
        dep3 = self.create_test_file("dep3")
        time.sleep(0.01)  # Ensure different mtime
        
        # Then create main file
        main_file = self.create_test_file("main")
        
        dependencies = [str(dep1), str(dep2), str(dep3)]
        
        # Cache with multiple dependencies
        self.cache.put(str(main_file), {'name': 'main'}, "llvm", dependencies)
        
        # Should load successfully
        cached = self.cache.get(str(main_file))
        assert cached is not None
        assert len(cached.dependencies) == 3
        
        # Modify one dependency
        time.sleep(0.01)
        dep2.write_text("# modified")
        
        # Should detect staleness
        cached = self.cache.get(str(main_file))
        assert cached is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
