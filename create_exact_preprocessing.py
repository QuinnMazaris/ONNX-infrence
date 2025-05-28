import pickle
import pandas as pd
import numpy as np

def load_exact_training_features():
    """Load the exact training features from the pickle file."""
    
    with open('exact_training_features.pkl', 'rb') as f:
        feature_mapping = pickle.load(f)
    
    return feature_mapping

def create_exact_features(df, feature_mapping):
    """Create the exact same features as used in training."""
    
    print("=== CREATING EXACT TRAINING FEATURES ===")
    
    # Get the expected feature columns
    expected_features = feature_mapping['feature_columns']
    print(f"Expected {len(expected_features)} features:")
    for i, col in enumerate(expected_features):
        print(f"  {i:2d}: {col}")
    print()
    
    # Start with an empty dataframe for our features
    X = pd.DataFrame(index=df.index)
    
    # 1. Add numerical features that exist directly in the data
    numerical_features = [
        'AnomalyScore', 'RegionArea', 'MaxVal', 'RegionCount', 
        'BurnThru', 'Concavity', 'Good', 'Porosity', 'Skip',
        'RegionRow', 'RegionCol', 'RegionAreaFrac', 'CL_ConfMax', 'CL_ConfMargin'
    ]
    
    for feature in numerical_features:
        if feature in df.columns and feature in expected_features:
            X[feature] = df[feature]
            print(f"✓ Added numerical feature: {feature}")
    
    # 2. Create one-hot encoded Weld features
    # Find all Weld_* columns in expected features
    weld_features = [col for col in expected_features if col.startswith('Weld_')]
    
    if weld_features:
        print(f"\nCreating one-hot encoded Weld features:")
        print(f"Expected Weld features: {weld_features}")
        
        # Get unique weld values from the data
        unique_welds = df['Weld'].unique()
        print(f"Unique welds in data: {unique_welds}")
        
        # Create one-hot encoding for each expected weld feature
        for weld_feature in weld_features:
            # Extract the weld name (e.g., 'WL21' from 'Weld_WL21')
            weld_name = weld_feature.replace('Weld_', '')
            
            # Create binary column: 1 if this row has this weld, 0 otherwise
            X[weld_feature] = (df['Weld'] == weld_name).astype(int)
            
            count = X[weld_feature].sum()
            print(f"  ✓ {weld_feature}: {count} samples")
    
    # 3. Verify we have all expected features
    missing_features = set(expected_features) - set(X.columns)
    if missing_features:
        print(f"\n❌ Missing features: {missing_features}")
        print("Available columns in original data:")
        for col in df.columns:
            print(f"  - {col}")
        raise ValueError(f"Cannot create missing features: {missing_features}")
    
    # 4. Reorder columns to match training exactly
    X = X[expected_features]
    
    print(f"\n✅ Successfully created {X.shape[1]} features matching training")
    print(f"Final shape: {X.shape}")
    
    return X

def preprocess_for_exact_inference():
    """Create preprocessed data that exactly matches the training preprocessing."""
    
    # Load the exact training feature mapping
    feature_mapping = load_exact_training_features()
    
    # Load the raw data
    df = pd.read_csv('training_data.csv')
    print(f"Loaded raw data: {df.shape}")
    
    # Create the exact features
    X = create_exact_features(df, feature_mapping)
    
    # Save the exact preprocessed data
    X.to_csv('exact_preprocessed_data.csv', index=False)
    print(f"\n✅ Saved exact preprocessed data to 'exact_preprocessed_data.csv'")
    
    # Save ground truth
    if 'GT_Label' in df.columns:
        df['GT_Label'].to_csv('exact_ground_truth.csv', index=False)
        print("✅ Saved ground truth to 'exact_ground_truth.csv'")
    
    # Verify first row matches training sample
    if 'sample_features' in feature_mapping:
        print("\n=== VERIFICATION ===")
        sample_row = X.iloc[0]
        training_sample = feature_mapping['sample_features']
        
        print("Comparing first row with training sample:")
        matches = 0
        total = 0
        
        for col in X.columns:
            if col in training_sample:
                our_val = sample_row[col]
                training_val = training_sample[col]
                
                # For boolean/binary features, compare as int
                if isinstance(training_val, bool):
                    our_val = bool(our_val)
                    match = our_val == training_val
                else:
                    # For numerical features, allow small differences
                    match = abs(float(our_val) - float(training_val)) < 1e-6
                
                status = "✓" if match else "✗"
                print(f"  {status} {col}: {our_val} vs {training_val}")
                
                if match:
                    matches += 1
                total += 1
        
        print(f"\nVerification: {matches}/{total} features match ({matches/total*100:.1f}%)")
    
    return X

if __name__ == "__main__":
    X = preprocess_for_exact_inference()
    print(f"\nFinal preprocessed data shape: {X.shape}")
    print("\nFirst few rows:")
    print(X.head()) 