"""
ML Compilation Decider - Phase 0, Week 3
Simple ML model to predict if a function should be compiled
"""

import ast
import inspect
import pickle
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class FunctionFeatures:
    """Features extracted from a Python function for ML prediction"""
    
    # Code complexity features
    num_lines: int = 0
    num_loops: int = 0
    num_conditionals: int = 0
    num_function_calls: int = 0
    num_arithmetic_ops: int = 0
    num_comparisons: int = 0
    max_loop_depth: int = 0
    
    # Type hint features
    has_type_hints: bool = False
    has_return_type: bool = False
    num_typed_params: int = 0
    
    # Code pattern features
    has_recursion: bool = False
    has_list_comprehension: bool = False
    has_generator: bool = False
    
    # Unsupported feature flags
    has_print: bool = False
    has_file_io: bool = False
    has_dynamic_code: bool = False  # eval, exec
    has_imports: bool = False
    has_class_def: bool = False
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML model"""
        return np.array([
            self.num_lines,
            self.num_loops,
            self.num_conditionals,
            self.num_function_calls,
            self.num_arithmetic_ops,
            self.num_comparisons,
            self.max_loop_depth,
            int(self.has_type_hints),
            int(self.has_return_type),
            self.num_typed_params,
            int(self.has_recursion),
            int(self.has_list_comprehension),
            int(self.has_generator),
            int(self.has_print),
            int(self.has_file_io),
            int(self.has_dynamic_code),
            int(self.has_imports),
            int(self.has_class_def),
        ], dtype=float)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get list of feature names"""
        return [
            'num_lines', 'num_loops', 'num_conditionals', 'num_function_calls',
            'num_arithmetic_ops', 'num_comparisons', 'max_loop_depth',
            'has_type_hints', 'has_return_type', 'num_typed_params',
            'has_recursion', 'has_list_comprehension', 'has_generator',
            'has_print', 'has_file_io', 'has_dynamic_code',
            'has_imports', 'has_class_def'
        ]


class FeatureExtractor(ast.NodeVisitor):
    """Extract features from Python AST"""
    
    def __init__(self):
        self.features = FunctionFeatures()
        self.loop_depth = 0
        self.max_loop_depth = 0
        self.function_name = None
        
    def visit_FunctionDef(self, node):
        """Extract function-level features"""
        if self.function_name is None:
            self.function_name = node.name
            
            # Check for type hints
            if node.returns is not None:
                self.features.has_return_type = True
                self.features.has_type_hints = True
            
            # Count typed parameters
            for arg in node.args.args:
                if arg.annotation is not None:
                    self.features.num_typed_params += 1
                    self.features.has_type_hints = True
        
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Count for loops and track depth"""
        self.features.num_loops += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1
    
    def visit_While(self, node):
        """Count while loops"""
        self.features.num_loops += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1
    
    def visit_If(self, node):
        """Count conditionals"""
        self.features.num_conditionals += 1
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Count function calls and detect specific patterns"""
        self.features.num_function_calls += 1
        
        # Check for specific function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check for recursion
            if func_name == self.function_name:
                self.features.has_recursion = True
            
            # Check for unsupported features
            if func_name in ['print', 'input']:
                self.features.has_print = True
            elif func_name in ['open', 'read', 'write']:
                self.features.has_file_io = True
            elif func_name in ['eval', 'exec', 'compile']:
                self.features.has_dynamic_code = True
        
        self.generic_visit(node)
    
    def visit_BinOp(self, node):
        """Count arithmetic operations"""
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, 
                                ast.FloorDiv, ast.Mod, ast.Pow)):
            self.features.num_arithmetic_ops += 1
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        """Count comparison operations"""
        self.features.num_comparisons += len(node.ops)
        self.generic_visit(node)
    
    def visit_ListComp(self, node):
        """Detect list comprehensions"""
        self.features.has_list_comprehension = True
        self.generic_visit(node)
    
    def visit_GeneratorExp(self, node):
        """Detect generators"""
        self.features.has_generator = True
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Detect imports"""
        self.features.has_imports = True
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Detect from imports"""
        self.features.has_imports = True
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Detect class definitions"""
        self.features.has_class_def = True
        self.generic_visit(node)


def extract_features(func: Callable) -> Optional[FunctionFeatures]:
    """
    Extract features from a Python function
    
    Args:
        func: Python function to analyze
        
    Returns:
        FunctionFeatures object or None if extraction fails
    """
    try:
        # Get source code
        source = inspect.getsource(func)
        
        # Remove leading indentation (dedent)
        import textwrap
        source = textwrap.dedent(source)
        
        # Parse to AST
        tree = ast.parse(source)
        
        # Extract features
        extractor = FeatureExtractor()
        extractor.visit(tree)
        
        # Add line count
        extractor.features.num_lines = len(source.strip().split('\n'))
        extractor.features.max_loop_depth = extractor.max_loop_depth
        
        return extractor.features
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None


class CompilationDecider:
    """
    ML model to predict if a function should be compiled
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 'logistic' or 'random_forest'
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        self.model_type = model_type
        
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 1 for compilable, 0 for not
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary of metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, features: FunctionFeatures) -> Tuple[bool, float]:
        """
        Predict if a function should be compiled
        
        Args:
            features: FunctionFeatures object
            
        Returns:
            (should_compile, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = features.to_array().reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        should_compile = bool(prediction)
        confidence = probabilities[1] if should_compile else probabilities[0]
        
        return should_compile, confidence
    
    def predict_function(self, func: Callable) -> Tuple[bool, float]:
        """
        Predict directly from a function
        
        Args:
            func: Python function
            
        Returns:
            (should_compile, confidence)
        """
        features = extract_features(func)
        if features is None:
            return False, 0.0
        
        return self.predict(features)
    
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance (for random forest)"""
        if self.model_type != 'random_forest':
            return {}
        
        importances = self.model.feature_importances_
        feature_names = FunctionFeatures.feature_names()
        
        return dict(zip(feature_names, importances))
    
    def save(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True


def generate_synthetic_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data
    This will be replaced with real data from profiling
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        features = FunctionFeatures()
        
        # Generate random features
        features.num_lines = np.random.randint(1, 100)
        features.num_loops = np.random.randint(0, 5)
        features.num_conditionals = np.random.randint(0, 10)
        features.num_function_calls = np.random.randint(0, 20)
        features.num_arithmetic_ops = np.random.randint(0, 50)
        features.num_comparisons = np.random.randint(0, 15)
        features.max_loop_depth = np.random.randint(0, 3)
        
        features.has_type_hints = np.random.choice([True, False])
        features.has_return_type = np.random.choice([True, False]) if features.has_type_hints else False
        features.num_typed_params = np.random.randint(0, 5) if features.has_type_hints else 0
        
        features.has_recursion = np.random.choice([True, False], p=[0.1, 0.9])
        features.has_list_comprehension = np.random.choice([True, False], p=[0.2, 0.8])
        features.has_generator = np.random.choice([True, False], p=[0.1, 0.9])
        
        # Unsupported features (more likely to be not compilable)
        features.has_print = np.random.choice([True, False], p=[0.3, 0.7])
        features.has_file_io = np.random.choice([True, False], p=[0.1, 0.9])
        features.has_dynamic_code = np.random.choice([True, False], p=[0.05, 0.95])
        features.has_imports = np.random.choice([True, False], p=[0.2, 0.8])
        features.has_class_def = np.random.choice([True, False], p=[0.1, 0.9])
        
        # Label: compilable if no unsupported features and has loops/arithmetic
        is_compilable = (
            not features.has_print and
            not features.has_file_io and
            not features.has_dynamic_code and
            not features.has_imports and
            not features.has_class_def and
            (features.num_loops > 0 or features.num_arithmetic_ops > 10)
        )
        
        X.append(features.to_array())
        y.append(1 if is_compilable else 0)
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Example usage
    print("Training ML compilation decider...")
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000)
    
    print(f"Generated {len(X)} training samples")
    print(f"Positive examples (compilable): {sum(y)}")
    print(f"Negative examples (not compilable): {len(y) - sum(y)}")
    
    # Train model
    decider = CompilationDecider(model_type='random_forest')
    metrics = decider.train(X, y)
    
    print("\nModel performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Test on example functions
    print("\nTesting on example functions:")
    
    def numeric_loop(n):
        total = 0
        for i in range(n):
            total += i * i
        return total
    
    def with_print(x):
        print(f"Value: {x}")
        return x * 2
    
    def simple_math(a, b):
        return a * b + a / b
    
    for func in [numeric_loop, with_print, simple_math]:
        should_compile, confidence = decider.predict_function(func)
        print(f"\n{func.__name__}:")
        print(f"  Should compile: {should_compile}")
        print(f"  Confidence: {confidence:.2%}")
    
    # Feature importance
    print("\nFeature importance:")
    importances = decider.feature_importance()
    for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {importance:.3f}")
