import pickle
import json
import numpy as np
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

def get_project_root():
    """Get the root directory of the project."""
    # Start from the current directory and look for the .csproj file
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if list(current_dir.glob('*.csproj')):
            return current_dir
        current_dir = current_dir.parent
    return Path.cwd()  # Return current directory if not found

def convert_pickle_to_json(pickle_path, json_path):
    """Convert the pickle file to a JSON format that C# can easily read."""
    with open(pickle_path, 'rb') as f:
        feature_mapping = pickle.load(f)
    
    # Convert numpy types to Python native types for JSON serialization
    json_mapping = {
        'feature_count': int(feature_mapping['feature_count']),
        'feature_columns': feature_mapping['feature_columns'],
        'preprocessing_steps': {}
    }
    
    # Convert preprocessing steps if they exist
    if 'preprocessing_steps' in feature_mapping:
        for step, value in feature_mapping['preprocessing_steps'].items():
            if isinstance(value, np.ndarray):
                json_mapping['preprocessing_steps'][step] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                json_mapping['preprocessing_steps'][step] = float(value)
            else:
                json_mapping['preprocessing_steps'][step] = value
    
    # Add sample features if they exist
    if 'sample_features' in feature_mapping:
        json_mapping['sample_features'] = {}
        for col, val in feature_mapping['sample_features'].items():
            if isinstance(val, (np.int64, np.float64)):
                json_mapping['sample_features'][col] = float(val)
            else:
                json_mapping['sample_features'][col] = val
    
    # Write to JSON file
    with open(json_path, 'w') as f:
        json.dump(json_mapping, f, indent=2)
    
    print(f"Successfully converted {pickle_path} to {json_path}")
    return json_mapping

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

def run_csharp_prediction(row_number=None):
    """Run the C# prediction program."""
    project_root = get_project_root()
    csproj_path = project_root / 'OnnxModelApp' / 'OnnxModelApp.csproj'
    
    if not csproj_path.exists():
        print(f"Error: Could not find C# project file at {csproj_path}")
        print("Current directory:", os.getcwd())
        print("Looking for .csproj files in:", project_root)
        return False
    
    print(f"Found project file at: {csproj_path}")
    
    # Build the C# project
    print("\nBuilding C# project...")
    try:
        build_result = subprocess.run(
            ['dotnet', 'build', str(csproj_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print("Build output:")
        print(build_result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error building C# project:")
        print("Exit code:", e.returncode)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during build: {e}")
        return False
    
    # Run the C# program
    print("\nRunning C# prediction...")
    try:
        cmd = ['dotnet', 'run', '--project', str(csproj_path)]
        if row_number is not None:
            cmd.append(str(row_number))
        
        run_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("\nC# Program Output:")
        print(run_result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Error running C# program:")
        print("Exit code:", e.returncode)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during run: {e}")
        return False

def main():
    # Get absolute paths
    project_root = get_project_root()
    pickle_path = project_root / 'exact_training_features.pkl'
    json_path = project_root / 'exact_training_features.json'
    config_path = project_root / 'OnnxModelApp' / 'appsettings.json'
    
    print(f"Project root: {project_root}")
    print(f"Working directory: {os.getcwd()}")
    
    if not pickle_path.exists():
        print(f"Error: {pickle_path} not found")
        return 1
    
    try:
        convert_pickle_to_json(pickle_path, json_path)
    except Exception as e:
        print(f"Error converting pickle to JSON: {e}")
        return 1
    
    # Update appsettings.json to use the JSON file
    try:
        if not config_path.exists():
            print(f"Error: {config_path} not found")
            return 1
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config['ModelSettings']['FeatureMappingPath'] = str(json_path.relative_to(project_root))
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        print(f"Updated {config_path} to use {json_path.name}")
    except Exception as e:
        print(f"Error updating appsettings.json: {e}")
        return 1
    
    # Run C# prediction
    row_number = None
    if len(sys.argv) > 1:
        try:
            row_number = int(sys.argv[1])
        except ValueError:
            print(f"Invalid row number: {sys.argv[1]}")
            return 1
    
    if not run_csharp_prediction(row_number):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 