import pandas as pd
import subprocess
import json

def test_model_accuracy():
    """Test the model accuracy on several samples."""
    
    # Load original data to get ground truth
    df = pd.read_csv('training_data.csv')
    
    # Test a mix of Good and Bad samples
    good_samples = df[df['GT_Label'] == 'Good'].index.tolist()[:5]
    bad_samples = df[df['GT_Label'] == 'Bad'].index.tolist()[:5]
    test_rows = good_samples + bad_samples
    
    print(f"Testing {len(test_rows)} samples...")
    print(f"Good samples: {good_samples}")
    print(f"Bad samples: {bad_samples}")
    print()
    
    results = []
    
    for row in test_rows:
        # Get ground truth
        gt_label = df.iloc[row]['GT_Label']
        
        # Get model prediction
        try:
            result = subprocess.run(['dotnet', 'run', '--project', 'OnnxModelApp', str(row)], 
                                  capture_output=True, text=True, timeout=30)
            
            # Parse the JSON result
            lines = result.stdout.split('\n')
            json_start = False
            json_lines = []
            for line in lines:
                if '=== PREDICTION RESULT ===' in line:
                    json_start = True
                    continue
                if json_start and line.strip():
                    json_lines.append(line)
            
            if json_lines:
                prediction_data = json.loads('\n'.join(json_lines))
                model_prediction = prediction_data['Prediction']
                raw_output = prediction_data['RawOutput']
                
                correct = (gt_label == model_prediction)
                results.append({
                    'Row': row,
                    'Ground_Truth': gt_label,
                    'Model_Prediction': model_prediction,
                    'Raw_Output': raw_output,
                    'Correct': correct
                })
                
                status = "✓" if correct else "✗"
                print(f'{status} Row {row}: GT={gt_label}, Pred={model_prediction} (raw={raw_output})')
            else:
                print(f'✗ Row {row}: Failed to parse prediction output')
                
        except Exception as e:
            print(f'✗ Row {row}: Error - {e}')
    
    if results:
        accuracy = sum(r["Correct"] for r in results) / len(results) * 100
        print(f'\nOverall Accuracy: {sum(r["Correct"] for r in results)}/{len(results)} = {accuracy:.1f}%')
        
        # Show confusion matrix
        tp = sum(1 for r in results if r["Ground_Truth"] == "Good" and r["Model_Prediction"] == "Good")
        tn = sum(1 for r in results if r["Ground_Truth"] == "Bad" and r["Model_Prediction"] == "Bad")
        fp = sum(1 for r in results if r["Ground_Truth"] == "Bad" and r["Model_Prediction"] == "Good")
        fn = sum(1 for r in results if r["Ground_Truth"] == "Good" and r["Model_Prediction"] == "Bad")
        
        print(f'\nConfusion Matrix:')
        print(f'True Positives (Good->Good): {tp}')
        print(f'True Negatives (Bad->Bad): {tn}')
        print(f'False Positives (Bad->Good): {fp}')
        print(f'False Negatives (Good->Bad): {fn}')
    
    return results

if __name__ == "__main__":
    test_model_accuracy() 