# ONNX Model Inference Project

A C# and Python pipeline for running inference on a Random Forest ONNX model for weld quality prediction, with exact feature matching and robust automation.

## Project Structure

```
ONNX infrence/
├── OnnxModelApp/                    # Main C# application
│   ├── Program.cs                   # Main application logic (entry point)
│   ├── DataPreprocessor.cs          # Handles feature extraction and preprocessing
│   ├── ModelConfig.cs               # Loads config and resolves file paths
│   ├── appsettings.json             # Configuration for file paths and thresholds
│   ├── Model/                       # Model helper classes
│   │   ├── OnnxModel.cs             # ONNX model wrapper (low-level)
│   │   ├── TensorHelper.cs          # Tensor utilities
│   │   └── RandomForest_production.onnx  # ONNX model file (copy)
│   ├── exact_preprocessed_data.csv  # Preprocessed data (generated)
│   └── OnnxModelApp.csproj          # C# project file
├── load_exact_features.py           # Python script for manual data preprocessing
├── Run_Full_Pipeline.py             # (Optional) End-to-end pipeline script
├── exact_training_features.pkl      # Feature mapping from training (pickle)
├── exact_training_features.json     # Feature mapping (JSON, for C#)
├── exact_preprocessed_data.csv      # Preprocessed data (generated)
├── training_data.csv                # Raw data
├── RandomForest_production.onnx     # ONNX model file
└── README.md                        # This file
```

## Prerequisites

- .NET 8.0 or later
- Python 3.7+ (for optional/manual preprocessing)
- Python packages: `pandas`, `numpy`, `pickle`

## Quick Start

### 1. Run the C# Application

Preprocessing is automatic if needed. Simply run:

```bash
cd OnnxModelApp
# For batch predictions (default test rows):
dotnet run
# For a specific row (e.g., row 480):
dotnet run 480
```

If `exact_preprocessed_data.csv` does not exist, it will be generated using the feature mapping and raw data.

### 2. (Optional) Manual Preprocessing

If you want to manually preprocess or debug features:

```bash
python load_exact_features.py
```
This will generate:
- `exact_preprocessed_data.csv` (features in exact order)
- `exact_ground_truth.csv` (labels, if present)

## Usage Examples

**Batch prediction:**
```bash
cd OnnxModelApp
dotnet run
```
Output:
```
Using model: .../RandomForest_production.onnx
Using preprocessed CSV: .../exact_preprocessed_data.csv

=== Model Information ===
Input metadata:
  float_input: System.Single, Shape: [-1, 23]
Output metadata:
  output_label: System.Int64, Shape: [-1]

=== EXACT FEATURE PREDICTIONS ===
Row 0: {"row": 0, "prediction": "Good", "confidence": 0.98, "label": 0}
Row 1: {"row": 1, "prediction": "Good", "confidence": 0.99, "label": 0}
Row 480: {"row": 480, "prediction": "Bad", "confidence": 0.87, "label": 1}
...
```

**Single row prediction:**
```bash
dotnet run 100
```
Output:
```
{"row": 100, "prediction": "Good", "confidence": 0.97, "label": 0}
```

## How It Works

### C# Pipeline
- **Program.cs**: Entry point. Loads config, checks/generates preprocessed data, runs predictions (batch or single row).
- **WeldPredictor**: Main class. Handles ONNX session, data loading, and prediction logic.
- **DataPreprocessor**: Loads feature mapping, extracts features in exact order, handles one-hot encoding, and writes preprocessed CSV.
- **ModelConfig**: Loads `appsettings.json`, resolves file paths robustly (searches multiple locations).
- **Model/OnnxModel.cs & TensorHelper.cs**: Helpers for ONNX model and tensor creation.

### Python Preprocessing
- **load_exact_features.py**: Loads feature mapping from pickle, extracts features from raw CSV in exact order, saves preprocessed data and ground truth. Used for debugging or manual runs.
- **exact_training_features.json**: JSON version of feature mapping, used by C# for feature order and preprocessing consistency.

### Configuration
- **appsettings.json**: Controls paths for model, data, preprocessed data, and feature mapping. Also sets prediction threshold. The C# app will search for files in several likely locations.

## Output Format
- Predictions are printed in JSON format:
  - `row`: Row number
  - `prediction`: "Good" or "Bad"
  - `confidence`: Probability/confidence for the predicted class (if available)
  - `label`: Raw output (0 = Good, 1 = Bad)

## Troubleshooting

- **"Could not find exact_preprocessed_data.csv"**
  - The C# app will attempt to generate it automatically. If it fails, run `python load_exact_features.py` manually.
- **"Could not find RandomForest_production.onnx"**
  - Ensure the ONNX model file is in the project root or referenced locations.
- **Feature count mismatch**
  - The feature mapping ensures exact order. If you get this error, re-run preprocessing.
- **Other file not found errors**
  - Check `appsettings.json` and file locations. The app searches multiple directories for each file.

## Technical Details
- Uses `Microsoft.ML.OnnxRuntime` for ONNX inference
- Ensures exact feature order and preprocessing as in training
- Robust file path detection for cross-platform use
- JSON output for easy integration
- Handles one-hot encoding for categorical features (e.g., Weld type)

---

For questions or issues, please check the code comments or open an issue. 