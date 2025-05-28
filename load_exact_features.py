import pickle
import pandas as pd
import numpy as np

def load_exact_training_features():
    """Load the exact training features from the pickle file."""
    
    with open('exact_training_features.pkl', 'rb') as f:
        feature_mapping = pickle.load(f)
    
    print("=== EXACT TRAINING FEATURES ===")
    print(f"Feature count: {feature_mapping['feature_count']}")
    print()
    print("Feature columns in exact order:")
    for i, col in enumerate(feature_mapping['feature_columns']):
        print(f"{i:2d}: {col}")
    print()
    
    if 'preprocessing_steps' in feature_mapping:
        print("Preprocessing steps used in training:")
        for step, value in feature_mapping['preprocessing_steps'].items():
            print(f"  {step}: {value}")
        print()
    
    print("Sample features (first row):")
    if 'sample_features' in feature_mapping:
        for col, val in feature_mapping['sample_features'].items():
            print(f"  {col}: {val}")
    
    return feature_mapping

def create_inference_preprocessor(feature_mapping):
    """Create a preprocessing function that matches the exact training preprocessing."""
    
    def preprocess_for_inference(csv_file='training_data.csv'):
        """Preprocess data using the exact same steps as training."""
        
        # Load the raw data
        df = pd.read_csv(csv_file)
        print(f"Loaded raw data: {df.shape}")
        
        # Get the exact feature columns from training
        feature_columns = feature_mapping['feature_columns']
        expected_count = feature_mapping['feature_count']
        
        print(f"Expected features: {expected_count}")
        print(f"Feature columns: {feature_columns}")
        
        # Extract the exact features in the exact order
        try:
            X = df[feature_columns].copy()
            print(f"Successfully extracted features: {X.shape}")
            
            # Verify we have the right number of features
            if X.shape[1] != expected_count:
                raise ValueError(f"Feature count mismatch: got {X.shape[1]}, expected {expected_count}")
            
            # Save the preprocessed data
            X.to_csv('exact_preprocessed_data.csv', index=False)
            print("Saved exact preprocessed data to 'exact_preprocessed_data.csv'")
            
            # Also save ground truth
            if 'GT_Label' in df.columns:
                df['GT_Label'].to_csv('exact_ground_truth.csv', index=False)
                print("Saved ground truth to 'exact_ground_truth.csv'")
            
            return X
            
        except KeyError as e:
            print(f"ERROR: Missing column in data: {e}")
            print("Available columns in CSV:")
            for i, col in enumerate(df.columns):
                print(f"  {i}: {col}")
            raise
    
    return preprocess_for_inference

if __name__ == "__main__":
    # Load the exact training features
    feature_mapping = load_exact_training_features()
    
    # Create and run the exact preprocessor
    preprocess_func = create_inference_preprocessor(feature_mapping)
    X = preprocess_func()
    
    print(f"\nFinal preprocessed data shape: {X.shape}")
    print("First few rows:")
    print(X.head()) 