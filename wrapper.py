import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

def load_config(path: Path):
    with open(path, 'r') as f:
        if path.suffix in {'.yaml', '.yml'}:
            return yaml.safe_load(f)
        return json.load(f)

def build_project():
    subprocess.run(['dotnet', 'build', 'SimpleEvaluator/SimpleEvaluator.csproj'], check=True)

def run_prediction(config_path: Path, feature_path: Path):
    result = subprocess.run(
        ['dotnet', 'run', '--no-build', '--project', 'SimpleEvaluator/SimpleEvaluator.csproj', str(config_path), str(feature_path)],
        capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip())

def evaluate(cfg):
    csv_path = Path(cfg['csv_path'])
    model_path = Path(cfg['model_path'])
    mapping_path = Path(cfg['feature_mapping_path'])
    gt_col = cfg['ground_truth_column']

    # Write runtime config for C#
    runtime_cfg = {
        'modelPath': str(model_path),
        'featureMappingPath': str(mapping_path)
    }
    config_json = Path('runtime_config.json')
    config_json.write_text(json.dumps(runtime_cfg))

    df = pd.read_csv(csv_path)
    y_true = df[gt_col].tolist()
    feature_df = df.drop(columns=[gt_col])

    preds = []
    temp_feat = Path('temp_features.json')
    for _, row in feature_df.iterrows():
        temp_feat.write_text(row.to_json())
        pred = run_prediction(config_json, temp_feat)
        preds.append(pred)

    cm = confusion_matrix(y_true, preds)
    out_df = df.copy()
    out_df['Prediction'] = preds
    out_df.to_csv('predictions.csv', index=False)

    print('Confusion matrix:')
    print(cm)
    print('Saved predictions to predictions.csv')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python wrapper.py <config.yaml>')
        sys.exit(1)
    cfg = load_config(Path(sys.argv[1]))
    build_project()
    evaluate(cfg)
