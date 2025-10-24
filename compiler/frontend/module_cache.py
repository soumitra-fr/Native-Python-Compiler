"""
Module Cache System for Native Python Compiler

Provides file-based caching of compiled modules (.pym files) with:
- Staleness detection based on source file modification times
- Cache invalidation and cleanup
- Dependency tracking for transitive recompilation
- Persistent storage for compiled IR and LLVM code

.pym file format (JSON):
{
    "version": "1.0",
    "source_file": "/path/to/module.py",
    "source_mtime": 1234567890.123,
    "dependencies": ["dep1.py", "dep2.py"],
    "ir_module": {...},  # Serialized IR
    "llvm_ir": "...",    # LLVM IR string
    "compiled_at": "2025-10-23T12:00:00"
}
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CachedModule:
    """Represents a cached compiled module"""
    source_file: str
    source_mtime: float
    dependencies: List[str]
    ir_module_json: dict  # Serialized IR module
    llvm_ir: str
    compiled_at: str
    version: str = "1.0"


class ModuleCache:
    """
    File-based module cache for compiled Python modules
    
    Features:
    - Persistent .pym files stored alongside source
    - Automatic staleness detection
    - Dependency tracking
    - Cache invalidation
    - Cleanup utilities
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize module cache
        
        Args:
            cache_dir: Directory for cache files. If None, uses __pycache__
        """
        self.cache_dir = cache_dir
        self.cache: Dict[str, CachedModule] = {}
        
    def _get_cache_path(self, source_file: str) -> Path:
        """
        Get cache file path for a source file
        
        Args:
            source_file: Path to Python source file
            
        Returns:
            Path to .pym cache file
        """
        source_path = Path(source_file).resolve()
        
        if self.cache_dir:
            # Use specified cache directory
            cache_dir = Path(self.cache_dir)
        else:
            # Use __pycache__ directory next to source
            cache_dir = source_path.parent / "__pycache__"
        
        # Create cache directory if needed
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache filename: module_name.pym
        module_name = source_path.stem
        cache_file = cache_dir / f"{module_name}.pym"
        
        return cache_file
    
    def _is_stale(self, cached: CachedModule) -> bool:
        """
        Check if cached module is stale
        
        Args:
            cached: Cached module to check
            
        Returns:
            True if cache is stale and needs recompilation
        """
        # Check if source file still exists
        if not os.path.exists(cached.source_file):
            return True
        
        # Check if source has been modified
        current_mtime = os.path.getmtime(cached.source_file)
        if current_mtime > cached.source_mtime:
            return True
        
        # Check if any dependency has been modified
        for dep in cached.dependencies:
            if not os.path.exists(dep):
                return True
            dep_mtime = os.path.getmtime(dep)
            if dep_mtime > cached.source_mtime:
                return True
        
        return False
    
    def get(self, source_file: str) -> Optional[CachedModule]:
        """
        Get cached module if valid
        
        Args:
            source_file: Path to source file
            
        Returns:
            CachedModule if valid cache exists, None otherwise
        """
        source_file = str(Path(source_file).resolve())
        
        # Check in-memory cache first
        if source_file in self.cache:
            cached = self.cache[source_file]
            if not self._is_stale(cached):
                return cached
            else:
                # Remove stale cache
                del self.cache[source_file]
        
        # Try loading from disk
        cache_path = self._get_cache_path(source_file)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            cached = CachedModule(**data)
            
            # Check if stale
            if self._is_stale(cached):
                # Remove stale cache file
                cache_path.unlink()
                return None
            
            # Store in memory cache
            self.cache[source_file] = cached
            return cached
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupted cache file - remove it
            cache_path.unlink()
            return None
    
    def put(self, source_file: str, ir_module_json: dict, llvm_ir: str, 
            dependencies: List[str]) -> None:
        """
        Store compiled module in cache
        
        Args:
            source_file: Path to source file
            ir_module_json: Serialized IR module (as dict)
            llvm_ir: Generated LLVM IR string
            dependencies: List of dependency file paths
        """
        source_file = str(Path(source_file).resolve())
        source_mtime = os.path.getmtime(source_file)
        
        # Resolve dependency paths to absolute paths
        resolved_deps = [str(Path(dep).resolve()) for dep in dependencies]
        
        cached = CachedModule(
            source_file=source_file,
            source_mtime=source_mtime,
            dependencies=resolved_deps,
            ir_module_json=ir_module_json,
            llvm_ir=llvm_ir,
            compiled_at=datetime.now().isoformat()
        )
        
        # Store in memory
        self.cache[source_file] = cached
        
        # Write to disk
        cache_path = self._get_cache_path(source_file)
        try:
            with open(cache_path, 'w') as f:
                json.dump(asdict(cached), f, indent=2)
        except (OSError, TypeError) as e:
            # Failed to write cache - non-fatal
            pass
    
    def invalidate(self, source_file: str) -> None:
        """
        Invalidate cache for a source file
        
        Args:
            source_file: Path to source file
        """
        source_file = str(Path(source_file).resolve())
        
        # Remove from memory
        if source_file in self.cache:
            del self.cache[source_file]
        
        # Remove from disk
        cache_path = self._get_cache_path(source_file)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear_all(self) -> int:
        """
        Clear all cached modules
        
        Returns:
            Number of cache files removed
        """
        count = 0
        
        # Clear memory cache
        self.cache.clear()
        
        # Clear disk cache
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.pym"):
                    cache_file.unlink()
                    count += 1
        
        return count
    
    def cleanup_stale(self) -> int:
        """
        Remove all stale cache files
        
        Returns:
            Number of stale cache files removed
        """
        count = 0
        
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.pym"):
                    try:
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                        cached = CachedModule(**data)
                        if self._is_stale(cached):
                            cache_file.unlink()
                            count += 1
                    except:
                        # Corrupted file - remove it
                        cache_file.unlink()
                        count += 1
        
        return count
    
    def get_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            'memory_cached': len(self.cache),
            'disk_cached': 0,
            'total_size_bytes': 0,
            'cache_dir': str(self.cache_dir) if self.cache_dir else 'multiple'
        }
        
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.pym"))
                stats['disk_cached'] = len(cache_files)
                stats['total_size_bytes'] = sum(f.stat().st_size for f in cache_files)
        
        return stats


# Global cache instance
_global_cache = ModuleCache()


def get_global_cache() -> ModuleCache:
    """Get the global module cache instance"""
    return _global_cache


def set_cache_dir(cache_dir: str) -> None:
    """Set the global cache directory"""
    global _global_cache
    _global_cache = ModuleCache(cache_dir)
