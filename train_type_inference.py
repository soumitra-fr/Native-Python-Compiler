#!/usr/bin/env python3
"""
Train the Type Inference Engine on collected runtime profiles

This script:
1. Loads all runtime profiles from training_data/
2. Extracts training examples (variable names, types, contexts)
3. Trains Random Forest classifier
4. Saves trained model
5. Tests predictions
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.type_inference_engine import TypeInferenceEngine


def load_training_data(data_dir='training_data') -> List[Tuple[str, str, str]]:
    """
    Load all profile JSON files and extract training examples
    
    Returns:
        List of (code_snippet, variable_name, true_type) tuples
    """
    training_data = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"⚠️  Warning: {data_dir}/ directory not found!")
        print("   Create it and add runtime profile JSON files.")
        print("   See AI_TRAINING_GUIDE.md for instructions.")
        return []
    
    profile_files = list(data_path.glob('*.json'))
    if not profile_files:
        print(f"⚠️  Warning: No JSON files found in {data_dir}/")
        print("   Run code with RuntimeTracer to generate profiles.")
        return []
    
    print(f"📊 Found {len(profile_files)} profile files")
    
    for profile_file in profile_files:
        try:
            with open(profile_file) as f:
                profile = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Skipping invalid JSON: {profile_file}")
            continue
        
        # Extract training examples from function calls
        for func_name, calls in profile.get('function_calls', {}).items():
            for call in calls:
                # Create code context
                code_context = f"def {func_name}(): pass"
                
                # Extract argument types
                for i, arg_type in enumerate(call.get('arg_types', [])):
                    var_name = f"arg{i}"
                    training_data.append((code_context, var_name, arg_type))
                
                # Extract return type
                return_type = call.get('return_type')
                if return_type:
                    training_data.append((code_context, 'return_value', return_type))
    
    return training_data


def main():
    print("=" * 70)
    print("🎓 TYPE INFERENCE ENGINE TRAINING")
    print("=" * 70)
    print()
    
    # Load training data
    print("📊 Loading training data...")
    training_data = load_training_data()
    
    if not training_data:
        print("❌ No training data found!")
        print()
        print("To collect training data:")
        print("  1. Run: python examples/phase0_demo.py")
        print("  2. Or see: docs/AI_TRAINING_GUIDE.md")
        return 1
    
    print(f"   ✅ Loaded {len(training_data)} training examples")
    types_found = set(t[2] for t in training_data)
    print(f"   📈 Types: {types_found}")
    print()
    
    # Create and train engine
    print("🔧 Training ML model...")
    engine = TypeInferenceEngine()
    
    try:
        accuracy = engine.train(training_data)
        print(f"   ✅ Training complete!")
        print(f"   📊 Accuracy: {accuracy:.2%}")
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Save trained model
    model_dir = Path('ai/models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'type_inference.pkl'
    
    print(f"💾 Saving model...")
    try:
        engine.save(str(model_path))
        print(f"   ✅ Saved to {model_path}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Test predictions
    print("🧪 Testing predictions:")
    print()
    
    test_cases = [
        ("count", "count = 0\nfor i in range(10):\n    count += 1"),
        ("name", "name = 'Alice'\nprint(name.upper())"),
        ("total", "total = sum([1, 2, 3, 4, 5])"),
        ("is_valid", "is_valid = True\nif is_valid:\n    pass"),
        ("price", "price = 19.99\ntotal = price * 1.08"),
        ("items", "items = []\nitems.append(1)"),
    ]
    
    for var_name, code in test_cases:
        prediction = engine.predict(code, var_name)
        confidence_bar = "█" * int(prediction.confidence * 20)
        
        print(f"   {var_name:12} → {prediction.predicted_type:8} "
              f"[{confidence_bar:<20}] {prediction.confidence:.1%}")
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  • Train strategy agent: python train_strategy_agent.py")
    print("  • Use in compilation: AI pipeline will auto-load trained model")
    print("  • See improvements: Better type predictions = better optimization")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
