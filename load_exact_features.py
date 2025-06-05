import pickle
import pandas as pd
import numpy as np
import difflib

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
    print()
    
    return feature_mapping

def create_inference_preprocessor(feature_mapping):
    """
    Return a function that:
      1. Loads an inference CSV
      2. Strips whitespace from column names
      3. Fuzzily matches each training feature name to a CSV column (if possible)
      4. Renames/fills so that we end up with exactly feature_mapping['feature_columns']
      5. Excludes GT_Label from X but saves it separately
    """
    
    def preprocess_for_inference(csv_file='training_data.csv'):
        # 1) Load raw data
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()  # remove leading/trailing spaces
        print(f"Loaded raw data: {df.shape}")
        
        # 2) Grab the exact feature names and count
        feature_columns = list(feature_mapping['feature_columns'])  # e.g. 26 names
        expected_count = feature_mapping['feature_count']           # e.g. 26
        
        # 3) Exclude GT_Label if it ever snuck into feature_columns
        if 'GT_Label' in feature_columns:
            feature_columns.remove('GT_Label')
        
        print(f"Expected feature count (excluding GT_Label): {expected_count}")
        print(f"Training feature list (to match against):\n  {feature_columns}\n")
        
        actual_cols = list(df.columns)
        rename_map = {}
        missing_features = []
        
        # 4) For each training feature name, try exact match first; else find a close match
        for tgt in feature_columns:
            if tgt in actual_cols:
                continue  # exact match—no rename needed
            
            # find best fuzzy match among actual_cols
            close = difflib.get_close_matches(tgt, actual_cols, n=1, cutoff=0.6)
            if close:
                rename_map[close[0]] = tgt
                print(f"🔁 Mapping '{close[0]}' → '{tgt}'")
            else:
                missing_features.append(tgt)
                print(f"⚠️ No match for '{tgt}'; will fill with 0.0")
        
        # 5) Apply renames in one shot
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        
        # 6) For any training feature still missing, add a column of zeros
        for col in missing_features:
            df[col] = 0.0
        
        # 7) Now extract X = df[feature_columns] in the exact order
        try:
            X = df[feature_columns].copy()
            print(f"\n✅ Final feature matrix shape: {X.shape}")
            
            # 8) Double-check column count
            if X.shape[1] != expected_count:
                print(f"⚠️  WARNING: Expected {expected_count} features, but got {X.shape[1]}")
            
            # 9) Save X out
            X.to_csv('exact_preprocessed_data.csv', index=False)
            print("Saved exact preprocessed features to 'exact_preprocessed_data.csv'")
            
            # 10) Save GT_Label if present
            if 'GT_Label' in df.columns:
                df['GT_Label'].to_csv('exact_ground_truth.csv', index=False)
                print("Saved ground truth (GT_Label) to 'exact_ground_truth.csv'")
            
            return X
        
        except KeyError as e:
            print(f"\n❌ ERROR: Could not extract all features: {e}")
            print("Available columns in CSV after renaming:")
            for i, col in enumerate(df.columns):
                print(f"  {i}: {col}")
            raise
    
    return preprocess_for_inference

if __name__ == "__main__":
    # 1) Load training feature mapping
    feature_mapping = load_exact_training_features()
    
    # 2) Instantiate the preprocessor
    preprocess_func = create_inference_preprocessor(feature_mapping)
    
    # 3) Run it on 'training_data.csv' (or any CSV you pass in)
    X = preprocess_func()
    
    print(f"\nFinal preprocessed data shape (X): {X.shape}")
    print("First few rows of X:")
    print(X.head())
