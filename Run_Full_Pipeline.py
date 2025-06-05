import pickle
import json
import numpy as np
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import re

def load_pipeline_config(config_path='pipeline_config.json'):
    """Load the pipeline configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded pipeline configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_path}: {e}")
        sys.exit(1)

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
    
    print(f"✅ Successfully converted pickle to JSON:")
    print(f"   📥 From: {pickle_path}")
    print(f"   📤 To: {json_path}")
    return json_mapping

def load_exact_training_features(pickle_path):
    """Load the exact training features from the pickle file."""
    
    with open(pickle_path, 'rb') as f:
        feature_mapping = pickle.load(f)
    
    print("📋 EXACT TRAINING FEATURES LOADED:")
    print(f"   🔢 Feature count: {feature_mapping['feature_count']}")
    print(f"   📝 Feature columns (in exact training order):")
    for i, col in enumerate(feature_mapping['feature_columns']):
        print(f"      {i:2d}: {col}")
    
    if 'preprocessing_steps' in feature_mapping:
        print(f"   ⚙️  Preprocessing steps used in training:")
        for step, value in feature_mapping['preprocessing_steps'].items():
            print(f"      {step}: {value}")
    
    if 'sample_features' in feature_mapping:
        print(f"   💡 Sample features (first row):")
        for col, val in feature_mapping['sample_features'].items():
            print(f"      {col}: {val}")
    
    return feature_mapping

def create_inference_preprocessor(feature_mapping, raw_csv_path, preprocessed_csv_path, ground_truth_csv_path):
    """Create a preprocessing function that matches the exact training preprocessing."""
    
    def preprocess_for_inference():
        """Preprocess data using the exact same steps as training."""
        
        # Load the raw data
        df = pd.read_csv(raw_csv_path)
        print(f"Loaded raw data from {raw_csv_path}: {df.shape}")
        
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
            X.to_csv(preprocessed_csv_path, index=False)
            print(f"Saved exact preprocessed data to '{preprocessed_csv_path}'")
            
            # Also save ground truth
            if 'GT_Label' in df.columns:
                df['GT_Label'].to_csv(ground_truth_csv_path, index=False)
                print(f"Saved ground truth to '{ground_truth_csv_path}'")
            
            return X
            
        except KeyError as e:
            print(f"ERROR: Missing column in data: {e}")
            print("Available columns in CSV:")
            for i, col in enumerate(df.columns):
                print(f"  {i}: {col}")
            raise
    
    return preprocess_for_inference

def update_appsettings(config, project_root):
    """Update the C# appsettings.json with paths from pipeline config."""
    appsettings_path = Path(config['paths']['appsettings'])
    
    try:
        if appsettings_path.exists():
            with open(appsettings_path, 'r') as f:
                appsettings = json.load(f)
        else:
            # Create a default structure if the file doesn't exist
            appsettings = {
                "ModelSettings": {},
                "SearchPaths": {
                    "ModelPaths": [],
                    "DataPaths": [],
                    "FeatureMappingPaths": []
                }
            }
            print(f"Created new default appsettings.json at {appsettings_path}")

        # Update paths using config values - these should be just filenames for FindFile to work correctly
        appsettings['ModelSettings']['ModelPath'] = Path(config['paths']['onnx_model']).name
        appsettings['ModelSettings']['RawDataPath'] = Path(config['paths']['ensemble_csv']).name
        appsettings['ModelSettings']['PreprocessedDataPath'] = Path(config['paths']['preprocessed_csv']).name
        appsettings['ModelSettings']['FeatureMappingPath'] = Path(config['paths']['json_file']).name
        
        appsettings['ModelSettings']['PredictionThreshold'] = config['model_settings']['prediction_threshold']

        # SearchPaths should contain paths relative to the project root (where the Python script is run from)
        onnx_model_filename = Path(config['paths']['onnx_model']).name
        ensemble_csv_filename = Path(config['paths']['ensemble_csv']).name
        json_filename = Path(config['paths']['json_file']).name
        preprocessed_csv_filename = Path(config['paths']['preprocessed_csv']).name
        output_dir_name = config['paths']['output_dir']

        appsettings['SearchPaths']['ModelPaths'] = [
            onnx_model_filename, # e.g. "RandomForest_production.onnx"
        ]
        appsettings['SearchPaths']['DataPaths'] = [
            ensemble_csv_filename, # e.g. "ensemble.csv"
        ]
        
        appsettings['SearchPaths']['FeatureMappingPaths'] = [
            f"{output_dir_name}/{json_filename}", # e.g. "output_files/exact_training_features.json"
            f"{output_dir_name}/{preprocessed_csv_filename}" # e.g. "output_files/exact_preprocessed_data.csv"
        ]
        
        with open(appsettings_path, 'w') as f:
            json.dump(appsettings, f, indent=4)
            
        print(f"Updated {appsettings_path} with pipeline config values")
        return True
    except Exception as e:
        print(f"Error updating appsettings.json: {e}")
        return False

def parse_predictions_from_output(output_text):
    """Parse prediction results from C# program output."""
    predictions = []
    
    # Look for JSON prediction lines
    lines = output_text.split('\n')
    capturing = False
    
    for line in lines:
        line = line.strip()
        if line == "PREDICTIONS_START":
            capturing = True
            continue
        elif line == "PREDICTIONS_END":
            capturing = False
            continue
        elif capturing and line.startswith('{"row":'):
            try:
                # Parse the JSON prediction
                prediction_data = json.loads(line)
                predictions.append(prediction_data)
            except json.JSONDecodeError:
                continue
    
    return predictions

def compute_confusion_matrix(config, predictions):
    """Compute and display confusion matrix comparing predictions to ground truth."""
    
    # Load ground truth
    ground_truth_path = config['paths']['ground_truth_csv']
    if not Path(ground_truth_path).exists():
        print(f"❌ Error: Ground truth file not found at {ground_truth_path}")
        return False
    
    print(f"📊 Loading ground truth from: {ground_truth_path}")
    # Read ground truth labels
    gt_df = pd.read_csv(ground_truth_path)
    ground_truth_labels = gt_df['GT_Label'].tolist()
    
    # Convert string labels to numeric (0=Good, 1=Bad)
    gt_numeric = [1 if label == 'Bad' else 0 for label in ground_truth_labels]
    
    # Extract predictions in the same order
    if not predictions:
        print("❌ No predictions found to compare!")
        return False
    
    # Sort predictions by row number to ensure correct order
    predictions_sorted = sorted(predictions, key=lambda x: x['row'])
    pred_numeric = [pred['label'] for pred in predictions_sorted]
    
    # Ensure we have the same number of predictions and ground truth labels
    min_length = min(len(gt_numeric), len(pred_numeric))
    gt_numeric = gt_numeric[:min_length]
    pred_numeric = pred_numeric[:min_length]
    
    print(f"🔢 Comparing {len(gt_numeric)} predictions to ground truth...")
    
    # Compute confusion matrix
    cm = confusion_matrix(gt_numeric, pred_numeric)
    
    # Display results
    print(f"\n📈 CONFUSION MATRIX:")
    print("                 Predicted")
    print("                Good  Bad")
    print(f"Actual Good    {cm[0,0]:5d} {cm[0,1]:4d}")
    print(f"       Bad     {cm[1,0]:5d} {cm[1,1]:4d}")
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📏 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   🔍 Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   ⚖️  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    target_names = ['Good', 'Bad']
    print(classification_report(gt_numeric, pred_numeric, target_names=target_names))
    
    return True

def run_csharp_prediction(config, row_number=None):
    """Run the C# prediction program."""
    csproj_path = config['paths']['csharp_project_file']
    
    if not Path(csproj_path).exists():
        print(f"Error: Could not find C# project file at {csproj_path}")
        return False, None
    
    print(f"Found project file at: {csproj_path}")
    
    # Build the C# project
    print("\nBuilding C# project...")
    try:
        build_result = subprocess.run(
            ['dotnet', 'build', csproj_path],
            capture_output=True,
            text=True,
            check=True
        )
        print("Build successful!")
        if build_result.stdout.strip():
            print("Build output:")
            print(build_result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error building C# project:")
        print("Exit code:", e.returncode)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False, None
    except Exception as e:
        print(f"Unexpected error during build: {e}")
        return False, None
    
    # Run the C# program
    print("\nRunning C# prediction...")
    try:
        cmd = ['dotnet', 'run', '--project', csproj_path]
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
        
        # Parse predictions if running all rows
        predictions = None
        if row_number is None:
            predictions = parse_predictions_from_output(run_result.stdout)
            print(f"\nParsed {len(predictions)} predictions from output")
        
        return True, predictions
    except subprocess.CalledProcessError as e:
        print("Error running C# program:")
        print("Exit code:", e.returncode)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False, None
    except Exception as e:
        print(f"Unexpected error during run: {e}")
        return False, None

def run_preprocessing(config):
    """Run the preprocessing step using configuration paths."""
    print("\n" + "=" * 50)
    print("            PREPROCESSING STEP")
    print("=" * 50)
    
    pickle_path = config['paths']['pickle_file']
    json_path = config['paths']['json_file']
    raw_csv_path = config['paths']['raw_csv']
    preprocessed_csv_path = config['paths']['preprocessed_csv']
    ground_truth_csv_path = config['paths']['ground_truth_csv']
    
    print(f"📂 Input files:")
    print(f"   📊 Raw CSV: {raw_csv_path}")
    print(f"   🗂️  Feature pickle: {pickle_path}")
    print(f"\n📤 Output files:")
    print(f"   📄 Feature JSON: {json_path}")
    print(f"   💾 Preprocessed CSV: {preprocessed_csv_path}")
    print(f"   🎯 Ground truth CSV: {ground_truth_csv_path}")
    
    # Check if we should skip preprocessing if files already exist
    if config['preprocessing'].get('skip_if_exists', False):
        if Path(preprocessed_csv_path).exists() and Path(json_path).exists():
            print(f"\n⏭️  SKIPPING PREPROCESSING - Files already exist:")
            print(f"   ✓ {preprocessed_csv_path}")
            print(f"   ✓ {json_path}")
            return True
    
    # Check if pickle file exists
    if not Path(pickle_path).exists():
        print(f"\n❌ Error: Feature pickle file not found: {pickle_path}")
        return False
    
    try:
        print(f"\n🔄 Converting pickle to JSON...")
        # Convert pickle to JSON
        convert_pickle_to_json(pickle_path, json_path)
        
        print(f"\n📋 Loading exact training features...")
        # Load feature mapping
        feature_mapping = load_exact_training_features(pickle_path)
        
        print(f"\n⚙️  Creating and running preprocessor...")
        # Create and run preprocessor
        preprocessor = create_inference_preprocessor(
            feature_mapping, 
            raw_csv_path, 
            preprocessed_csv_path, 
            ground_truth_csv_path
        )
        preprocessor()
        
        print(f"\n✅ Preprocessing completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        return False

def main():
    print("=" * 60)
    print("          ONNX INFERENCE PIPELINE STARTING")
    print("=" * 60)
    
    # Load pipeline configuration
    config = load_pipeline_config()
    
    project_root = Path(config['paths']['project_root'])
    output_dir_name = config['paths']['output_dir']
    output_dir = project_root / output_dir_name
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory ensured: {output_dir}")

    # Get absolute paths
    pickle_path = project_root / config['paths']['pickle_file']
    json_path = output_dir / config['paths']['json_file']
    raw_csv_path = project_root / config['paths']['raw_csv']
    preprocessed_csv_path = output_dir / config['paths']['preprocessed_csv']
    ground_truth_csv_path = output_dir / config['paths']['ground_truth_csv']
    appsettings_path = project_root / config['paths']['appsettings']
    onnx_model_path = project_root / config['paths']['onnx_model']
    ensemble_csv_path = project_root / config['paths']['ensemble_csv']

    # Update config with full paths to be passed to functions
    config['paths']['json_file'] = str(json_path)
    config['paths']['raw_csv'] = str(raw_csv_path)
    config['paths']['preprocessed_csv'] = str(preprocessed_csv_path)
    config['paths']['ground_truth_csv'] = str(ground_truth_csv_path)
    config['paths']['appsettings'] = str(appsettings_path)
    config['paths']['onnx_model'] = str(onnx_model_path)
    config['paths']['ensemble_csv'] = str(ensemble_csv_path)

    print(f"\n📁 Working directory: {os.getcwd()}")
    print(f"⚙️  Pipeline configuration loaded successfully")
    
    print(f"\n🔧 PIPELINE CONFIGURATION:")
    print(f"   📊 Input CSV: {raw_csv_path}")
    print(f"   🧠 ONNX Model: {onnx_model_path}")  
    print(f"   🗂️  Feature Pickle: {pickle_path}")
    print(f"   📄 Feature JSON: {json_path}")
    print(f"   💾 Preprocessed CSV: {preprocessed_csv_path}")
    print(f"   🎯 Ground Truth CSV: {ground_truth_csv_path}")
    print(f"   ⚙️  C# appsettings: {appsettings_path}")
    
    # Check if key files exist
    print(f"\n🔍 FILE EXISTENCE CHECK:")
    print(f"   {'✓' if raw_csv_path.exists() else '✗'} Input CSV: {raw_csv_path}")
    print(f"   {'✓' if onnx_model_path.exists() else '✗'} ONNX Model: {onnx_model_path}")
    print(f"   {'✓' if pickle_path.exists() else '✗'} Feature Pickle: {pickle_path}")
    
    # Run preprocessing if enabled
    if config['preprocessing']['run_before_inference']:
        if not run_preprocessing(config):
            return 1
    
    # Update appsettings.json with config values
    if not update_appsettings(config, project_root):
        return 1
    
    # Get row number from command line if provided
    row_number = None
    if len(sys.argv) > 1:
        try:
            row_number = int(sys.argv[1])
            print(f"Will infer row {row_number}")
        except ValueError:
            print(f"Invalid row number: {sys.argv[1]}")
            return 1
    else:
        print("No row number specified - C# will loop through all rows")
    
    # Run C# prediction
    print("\n" + "=" * 50)
    print("           C# INFERENCE STEP")
    print("=" * 50)
    print(f"🖥️  Running C# ONNX inference...")
    if row_number is not None:
        print(f"🎯 Target row: {row_number}")
    else:
        print(f"🔄 Processing all rows in dataset")
    
    success, predictions = run_csharp_prediction(config, row_number)
    if not success:
        print("\n❌ C# inference failed!")
        return 1
    print(f"✅ C# inference completed successfully!")
    
    # Compute confusion matrix if we ran all rows
    if row_number is None and predictions:
        print(f"\n" + "=" * 50)
        print("        PERFORMANCE EVALUATION")
        print("=" * 50)
        if not compute_confusion_matrix(config, predictions):
            print("⚠️  Warning: Could not compute confusion matrix")
    
    print(f"\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 