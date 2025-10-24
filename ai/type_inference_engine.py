"""
Phase 2.2: AI Type Inference Engine

Uses machine learning to infer types from code patterns.
Based on collected runtime data from Phase 2.1.

This is a simplified version for Phase 2 - full transformer model
would require GPU training on Kaggle/Colab.

Phase: 2.2 (AI Type Inference)
"""

import ast
import json
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

# ML imports (simplified for now)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class TypePrediction:
    """Result of type inference"""
    variable_name: str
    predicted_type: str
    confidence: float
    alternatives: List[Tuple[str, float]]  # [(type, confidence), ...]


class TypeInferenceEngine:
    """
    ML-based type inference engine
    
    Features extracted from code:
    - Variable name patterns
    - Context (operations, function calls)
    - Value patterns (literals)
    - Usage patterns (how variable is used)
    
    Training data comes from runtime profiles.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.type_mapping = {
            'int': 0,
            'float': 1,
            'bool': 2,
            'str': 3,
            'None': 4,
            'unknown': 5
        }
        self.reverse_type_mapping = {v: k for k, v in self.type_mapping.items()}
        self.is_trained = False
    
    def extract_features(self, code_snippet: str, variable_name: str) -> str:
        """
        Extract features from code for ML model
        
        Features:
        - Variable name
        - Operations used
        - Function calls
        - Literals
        """
        features = []
        
        # Variable name pattern
        features.append(f"var:{variable_name}")
        
        # Name-based heuristics
        if any(x in variable_name.lower() for x in ['count', 'num', 'idx', 'index', 'size', 'len']):
            features.append("name_pattern:integer")
        if any(x in variable_name.lower() for x in ['rate', 'ratio', 'percent', 'avg', 'mean']):
            features.append("name_pattern:float")
        if any(x in variable_name.lower() for x in ['is_', 'has_', 'can_', 'flag']):
            features.append("name_pattern:bool")
        if any(x in variable_name.lower() for x in ['str', 'text', 'name', 'msg', 'message']):
            features.append("name_pattern:string")
        
        # Parse code to extract operations
        try:
            tree = ast.parse(code_snippet)
            for node in ast.walk(tree):
                # Binary operations
                if isinstance(node, ast.BinOp):
                    if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                        features.append("op:arithmetic")
                    if isinstance(node.op, ast.Div):
                        features.append("op:division")
                    if isinstance(node.op, ast.FloorDiv):
                        features.append("op:floordiv")
                
                # Comparisons
                if isinstance(node, ast.Compare):
                    features.append("op:comparison")
                
                # Function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        features.append(f"call:{node.func.id}")
                
                # Literals
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, int):
                        features.append("literal:int")
                    elif isinstance(node.value, float):
                        features.append("literal:float")
                    elif isinstance(node.value, str):
                        features.append("literal:str")
                    elif isinstance(node.value, bool):
                        features.append("literal:bool")
        except:
            pass
        
        return " ".join(features)
    
    def train(self, training_data: List[Tuple[str, str, str]]) -> float:
        """
        Train the model on labeled data
        
        Args:
            training_data: List of (code_snippet, variable_name, true_type)
        
        Returns:
            Accuracy on validation set
        """
        if not SKLEARN_AVAILABLE:
            print("Error: scikit-learn required for training")
            return 0.0
        
        # Extract features
        X_text = []
        y = []
        
        for code, var_name, true_type in training_data:
            features = self.extract_features(code, var_name)
            X_text.append(features)
            y.append(self.type_mapping.get(true_type, self.type_mapping['unknown']))
        
        # Convert to features
        self.vectorizer = TfidfVectorizer(max_features=100)
        X = self.vectorizer.fit_transform(X_text)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        accuracy = self.classifier.score(X_val, y_val)
        self.is_trained = True
        
        return accuracy
    
    def predict(self, code_snippet: str, variable_name: str) -> TypePrediction:
        """
        Predict type for a variable
        
        Args:
            code_snippet: Code containing the variable
            variable_name: Name of variable to infer type for
        
        Returns:
            TypePrediction with predicted type and confidence
        """
        if not self.is_trained:
            # Fallback to heuristics
            return self._heuristic_prediction(code_snippet, variable_name)
        
        # Extract features
        features = self.extract_features(code_snippet, variable_name)
        X = self.vectorizer.transform([features])
        
        # Predict
        probas = self.classifier.predict_proba(X)[0]
        predicted_class = np.argmax(probas)
        confidence = probas[predicted_class]
        
        # Get alternatives
        alternatives = []
        for class_idx, prob in enumerate(probas):
            if class_idx != predicted_class and prob > 0.1:
                alternatives.append((self.reverse_type_mapping[class_idx], float(prob)))
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return TypePrediction(
            variable_name=variable_name,
            predicted_type=self.reverse_type_mapping[predicted_class],
            confidence=float(confidence),
            alternatives=alternatives[:3]  # Top 3 alternatives
        )
    
    def _heuristic_prediction(self, code_snippet: str, variable_name: str) -> TypePrediction:
        """Simple heuristic-based prediction when model not trained"""
        # Name-based heuristics
        name = variable_name.lower()
        
        if any(x in name for x in ['count', 'num', 'idx', 'index', 'size', 'len', 'total']):
            return TypePrediction(name, 'int', 0.7, [('float', 0.2)])
        elif any(x in name for x in ['rate', 'ratio', 'percent', 'avg', 'mean', 'price', 'cost']):
            return TypePrediction(name, 'float', 0.7, [('int', 0.2)])
        elif any(x in name for x in ['is_', 'has_', 'can_', 'flag', 'enabled', 'disabled']):
            return TypePrediction(name, 'bool', 0.8, [])
        elif any(x in name for x in ['str', 'text', 'name', 'msg', 'message', 'title']):
            return TypePrediction(name, 'str', 0.7, [])
        else:
            return TypePrediction(name, 'unknown', 0.3, [('int', 0.3), ('str', 0.2)])
    
    def save(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'type_mapping': self.type_mapping
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.type_mapping = model_data['type_mapping']
        self.reverse_type_mapping = {v: k for k, v in self.type_mapping.items()}
        self.is_trained = True


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("AI TYPE INFERENCE ENGINE - Demo")
    print("="*80)
    
    # Create engine
    engine = TypeInferenceEngine()
    
    # Generate synthetic training data
    training_data = [
        ("count = 0\ncount = count + 1", "count", "int"),
        ("total = sum(numbers)", "total", "int"),
        ("price = 19.99", "price", "float"),
        ("ratio = a / b", "ratio", "float"),
        ("is_valid = x > 0", "is_valid", "bool"),
        ("has_error = check()", "has_error", "bool"),
        ("name = 'John'", "name", "str"),
        ("message = get_message()", "message", "str"),
    ] * 50  # Repeat for more training samples
    
    if SKLEARN_AVAILABLE:
        # Train
        print("\nTraining model...")
        accuracy = engine.train(training_data)
        print(f"✅ Training complete! Validation accuracy: {accuracy:.1%}")
    else:
        print("\n⚠️  Using heuristic-based inference (sklearn not available)")
    
    # Test predictions
    print("\nTest Predictions:")
    print("-" * 80)
    
    test_cases = [
        ("user_count = len(users)", "user_count"),
        ("average_score = sum(scores) / len(scores)", "average_score"),
        ("is_admin = user.role == 'admin'", "is_admin"),
        ("username = input('Name: ')", "username"),
    ]
    
    for code, var in test_cases:
        prediction = engine.predict(code, var)
        print(f"\nCode: {code}")
        print(f"Variable: {var}")
        print(f"Predicted Type: {prediction.predicted_type} (confidence: {prediction.confidence:.1%})")
        if prediction.alternatives:
            alts = ", ".join([f"{t} ({c:.1%})" for t, c in prediction.alternatives])
            print(f"Alternatives: {alts}")
    
    print("\n" + "="*80)
    print("✅ Type inference engine working!")
    print("="*80)
