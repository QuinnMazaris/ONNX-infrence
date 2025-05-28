"""
Script to extract feature preprocessing information from training script.
This ensures our ONNX inference uses the exact same features as training.
"""

def extract_training_features():
    """
    This function should mirror your training script's Initialize function
    to extract the exact feature preprocessing steps.
    """
    
    print("=== FEATURE EXTRACTION FROM TRAINING ===")
    print("Please add this code to your training script after the Initialize function:")
    print()
    
    code_to_add = '''
# Add this after: df, X, y = Initialize(config, SAVE_MODEL)

print("=== TRAINING FEATURE INFO ===")
print(f"Final X shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print()
print("Feature columns in exact order:")
for i, col in enumerate(X.columns):
    print(f"{i:2d}: {col}")
print()
print("First row of X (sample features):")
print(X.iloc[0].values)
print()
print("Feature statistics:")
print(X.describe())

# Save the exact feature info for inference
import pickle
feature_mapping = {
    'feature_columns': list(X.columns),
    'feature_count': X.shape[1],
    'sample_features': X.iloc[0].to_dict(),
    'feature_dtypes': X.dtypes.to_dict(),
    'preprocessing_steps': {
        'variance_filter': config.FilterData and hasattr(config, 'VARIANCE_THRESH'),
        'correlation_filter': config.FilterData and hasattr(config, 'CORRELATION_THRESH'),
        'variance_threshold': getattr(config, 'VARIANCE_THRESH', None),
        'correlation_threshold': getattr(config, 'CORRELATION_THRESH', None)
    }
}

with open('exact_training_features.pkl', 'wb') as f:
    pickle.dump(feature_mapping, f)
    
print("Saved exact training features to 'exact_training_features.pkl'")
'''
    
    print(code_to_add)
    print()
    print("After running your training script with this code, share the output and the")
    print("'exact_training_features.pkl' file. This will give me the exact feature")
    print("preprocessing to match in our inference script.")

if __name__ == "__main__":
    extract_training_features() 