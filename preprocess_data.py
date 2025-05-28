import pandas as pd
import pickle
import numpy as np

def load_feature_info():
    """Load the feature preprocessing information from the pickle file."""
    with open('feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    return feature_info

def preprocess_data():
    """Preprocess the training data to match the model's expected input format."""
    
    # Load the original data
    df = pd.read_csv('training_data.csv')
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Start with numerical features (all except GT_Label_Num which is the target)
    numerical_features = [
        'AnomalyScore', 'RegionArea', 'MaxVal', 'RegionCount', 
        'BurnThru', 'Concavity', 'Good', 'Porosity', 'Skip',
        'AD_ClassID', 'CL_ClassID', 'RegionRow', 'RegionCol', 
        'RegionAreaFrac', 'CL_ConfMax', 'CL_ConfMargin',
        'SegThresh', 'ClassThresh'
    ]
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    
    # Start with numerical features
    processed_df = df[numerical_features].copy()
    
    # One-hot encode categorical features (NOT including GT_Label since it's the target)
    categorical_features = ['AD_Decision', 'CL_Decision']
    
    for feature in categorical_features:
        # Create one-hot encoding for Bad/Good categories
        for category in ['Bad', 'Good']:
            col_name = f"{feature}_{category}"
            processed_df[col_name] = (df[feature] == category).astype(int)
            print(f"Created {col_name}: {processed_df[col_name].sum()} samples")
    
    print(f"Final processed data shape: {processed_df.shape}")
    print(f"Final columns ({len(processed_df.columns)}): {list(processed_df.columns)}")
    
    # Save the preprocessed data
    processed_df.to_csv('preprocessed_data.csv', index=False)
    print("Preprocessed data saved to 'preprocessed_data.csv'")
    
    # Also save the ground truth labels separately
    gt_labels = df['GT_Label'].copy()
    gt_labels.to_csv('ground_truth_labels.csv', index=False)
    print("Ground truth labels saved to 'ground_truth_labels.csv'")
    
    return processed_df

def save_preprocessed_data(feature_matrix, output_file='preprocessed_data.csv'):
    """Save the preprocessed feature matrix to a CSV file."""
    feature_matrix.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

def get_feature_vector_for_row(feature_matrix, row_index):
    """Get a single feature vector for the specified row."""
    if row_index >= len(feature_matrix):
        raise IndexError(f"Row index {row_index} is out of bounds. Data has {len(feature_matrix)} rows.")
    
    feature_vector = feature_matrix.iloc[row_index].values.astype(np.float32)
    return feature_vector

def analyze_preprocessing():
    """Analyze the preprocessing and show some examples."""
    feature_matrix = preprocess_data()
    
    print("\n" + "="*50)
    print("PREPROCESSING ANALYSIS")
    print("="*50)
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Expected: (2412, 23)")
    
    print("\nFirst few rows of processed features:")
    print(feature_matrix.head())
    
    print("\nFeature statistics:")
    print(feature_matrix.describe())

if __name__ == "__main__":
    result = preprocess_data()
    print(f"\nWe have {result.shape[1]} features, but need 23. Missing: {23 - result.shape[1]} feature(s)")
    
    # Let's check what we might be missing
    print("\nLet's add the Weld column as a numerical feature...")
    
    # Load original data again and add Weld as numerical
    df = pd.read_csv('training_data.csv')
    
    # Convert Weld to numerical (extract number from WL format)
    df['Weld_Num'] = df['Weld'].str.extract('(\d+)').astype(int)
    
    # Add this to our processed data
    result['Weld_Num'] = df['Weld_Num']
    
    print(f"After adding Weld_Num: {result.shape}")
    print(f"Final columns ({len(result.columns)}): {list(result.columns)}")
    
    # Save the corrected data
    result.to_csv('preprocessed_data.csv', index=False)
    print("Updated preprocessed data saved!") 