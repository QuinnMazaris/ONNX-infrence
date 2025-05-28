import pandas as pd
import subprocess
import json

def test_exact_model_accuracy():
    """Test the model accuracy using the exact preprocessed features."""
    
    # Load ground truth
    gt_df = pd.read_csv('exact_ground_truth.csv')
    
    print("=== EXACT FEATURE ACCURACY TEST ===")
    print(f"Total samples: {len(gt_df)}")
    
    # Get distribution of ground truth
    gt_counts = gt_df['GT_Label'].value_counts()
    print(f"Ground truth distribution:")
    for label, count in gt_counts.items():
        print(f"  {label}: {count} samples ({count/len(gt_df)*100:.1f}%)")
    print()
    
    # Test a balanced sample of Good and Bad
    good_indices = gt_df[gt_df['GT_Label'] == 'Good'].index.tolist()
    bad_indices = gt_df[gt_df['GT_Label'] == 'Bad'].index.tolist()
    
    # Take first 10 of each type for testing
    test_good = good_indices[:10]
    test_bad = bad_indices[:10]
    test_rows = test_good + test_bad
    
    print(f"Testing {len(test_rows)} samples:")
    print(f"Good samples (rows): {test_good}")
    print(f"Bad samples (rows): {test_bad}")
    print()
    
    results = []
    
    for row in test_rows:
        # Get ground truth
        gt_label = gt_df.iloc[row]['GT_Label']
        
        # Get model prediction
        try:
            result = subprocess.run(['dotnet', 'run', '--project', 'OnnxModelApp', str(row)], 
                                  capture_output=True, text=True, timeout=30)
            
            # Parse the JSON result from the last line
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]  # Last line should be the JSON result
            
            if json_line.startswith('{') and json_line.endswith('}'):
                prediction_data = json.loads(json_line)
                model_prediction = prediction_data['prediction']
                raw_output = prediction_data['raw_output']
                
                correct = (gt_label == model_prediction)
                results.append({
                    'Row': row,
                    'Ground_Truth': gt_label,
                    'Model_Prediction': model_prediction,
                    'Raw_Output': raw_output,
                    'Correct': correct
                })
                
                status = "✓" if correct else "✗"
                print(f'{status} Row {row:4d}: GT={gt_label:4s}, Pred={model_prediction:4s} (raw={raw_output})')
            else:
                print(f'✗ Row {row}: Failed to parse prediction output: {json_line}')
                
        except Exception as e:
            print(f'✗ Row {row}: Error - {e}')
    
    if results:
        # Calculate overall accuracy
        correct_predictions = sum(r["Correct"] for r in results)
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions * 100
        
        print(f'\n=== RESULTS ===')
        print(f'Overall Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1f}%')
        
        # Calculate accuracy by class
        good_results = [r for r in results if r["Ground_Truth"] == "Good"]
        bad_results = [r for r in results if r["Ground_Truth"] == "Bad"]
        
        if good_results:
            good_accuracy = sum(r["Correct"] for r in good_results) / len(good_results) * 100
            print(f'Good Class Accuracy: {sum(r["Correct"] for r in good_results)}/{len(good_results)} = {good_accuracy:.1f}%')
        
        if bad_results:
            bad_accuracy = sum(r["Correct"] for r in bad_results) / len(bad_results) * 100
            print(f'Bad Class Accuracy: {sum(r["Correct"] for r in bad_results)}/{len(bad_results)} = {bad_accuracy:.1f}%')
        
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
        
        # Calculate precision, recall, F1
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f'Precision: {precision:.3f}')
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f'Recall: {recall:.3f}')
        
        if tp + fp > 0 and tp + fn > 0:
            f1 = 2 * precision * recall / (precision + recall)
            print(f'F1 Score: {f1:.3f}')
    
    return results

def test_larger_sample():
    """Test on a larger sample to get better accuracy estimate."""
    
    print("\n" + "="*50)
    print("LARGER SAMPLE TEST")
    print("="*50)
    
    # Load ground truth
    gt_df = pd.read_csv('exact_ground_truth.csv')
    
    # Test every 100th sample for speed
    test_indices = list(range(0, len(gt_df), 100))
    print(f"Testing {len(test_indices)} samples (every 100th): {test_indices}")
    
    results = []
    
    for i, row in enumerate(test_indices):
        gt_label = gt_df.iloc[row]['GT_Label']
        
        try:
            result = subprocess.run(['dotnet', 'run', '--project', 'OnnxModelApp', str(row)], 
                                  capture_output=True, text=True, timeout=30)
            
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]
            
            if json_line.startswith('{') and json_line.endswith('}'):
                prediction_data = json.loads(json_line)
                model_prediction = prediction_data['prediction']
                raw_output = prediction_data['raw_output']
                
                correct = (gt_label == model_prediction)
                results.append({
                    'Row': row,
                    'Ground_Truth': gt_label,
                    'Model_Prediction': model_prediction,
                    'Raw_Output': raw_output,
                    'Correct': correct
                })
                
                status = "✓" if correct else "✗"
                print(f'{status} Sample {i+1:2d}/Row {row:4d}: GT={gt_label:4s}, Pred={model_prediction:4s}')
            
        except Exception as e:
            print(f'✗ Sample {i+1}/Row {row}: Error - {e}')
    
    if results:
        accuracy = sum(r["Correct"] for r in results) / len(results) * 100
        print(f'\nLarger Sample Accuracy: {sum(r["Correct"] for r in results)}/{len(results)} = {accuracy:.1f}%')
    
    return results

if __name__ == "__main__":
    # Test detailed accuracy on small sample
    detailed_results = test_exact_model_accuracy()
    
    # Test larger sample for better estimate
    larger_results = test_larger_sample()
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    if detailed_results:
        detailed_acc = sum(r["Correct"] for r in detailed_results) / len(detailed_results) * 100
        print(f"Detailed test (20 samples): {detailed_acc:.1f}% accuracy")
    
    if larger_results:
        larger_acc = sum(r["Correct"] for r in larger_results) / len(larger_results) * 100
        print(f"Larger test ({len(larger_results)} samples): {larger_acc:.1f}% accuracy")
    
    print("\n✅ EXACT FEATURE MATCHING COMPLETE!") 