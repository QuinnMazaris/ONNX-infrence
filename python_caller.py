import subprocess
import json
import os
import sys

class WeldPredictorPython:
    def __init__(self, csharp_project_path="OnnxModelApp"):
        self.project_path = csharp_project_path
        
    def predict_row(self, row_number):
        """
        Predict a specific row from the CSV using the C# ONNX model
        
        Args:
            row_number (int): Row number to predict (1-based indexing)
            
        Returns:
            dict: Prediction result with image path, prediction, confidence, etc.
        """
        try:
            # Run the C# application with the row number as argument
            result = subprocess.run(
                ["dotnet", "run", "--project", self.project_path, str(row_number)],
                capture_output=True,
                text=True,
                cwd="."
            )
            
            if result.returncode != 0:
                return {"error": f"C# application failed: {result.stderr}"}
            
            # Parse the output
            output_lines = result.stdout.strip().split('\n')
            
            # Extract information from the output
            prediction_result = {"row": row_number}
            for line in output_lines:
                line = line.strip()
                if "Image:" in line:
                    prediction_result["image_path"] = line.split("Image: ")[1]
                elif "Prediction:" in line:
                    prediction_result["prediction"] = line.split("Prediction: ")[1]
                elif "Confidence:" in line:
                    try:
                        prediction_result["confidence"] = int(line.split("Confidence: ")[1])
                    except:
                        prediction_result["confidence"] = line.split("Confidence: ")[1]
                elif "Features processed:" in line:
                    prediction_result["feature_count"] = int(line.split("Features processed: ")[1])
            
            return prediction_result
            
        except Exception as e:
            return {"error": str(e)}
    
    def predict_multiple_rows(self, row_numbers):
        """
        Predict multiple rows
        
        Args:
            row_numbers (list): List of row numbers to predict
            
        Returns:
            list: List of prediction results
        """
        results = []
        for row_num in row_numbers:
            result = self.predict_row(row_num)
            results.append(result)
        return results

def main():
    # Example usage
    predictor = WeldPredictorPython()
    
    print("Testing C# ONNX Model from Python")
    print("=" * 50)
    
    # Test single prediction
    print("\n1. Single Prediction (Row 1):")
    result1 = predictor.predict_row(1)
    print(json.dumps(result1, indent=2))
    
    # Test multiple predictions
    print("\n2. Multiple Predictions (Rows 1-5):")
    results = predictor.predict_multiple_rows([1, 2, 3, 4, 5])
    for i, result in enumerate(results, 1):
        print(f"Row {i}: {result.get('prediction', 'Error')} - {os.path.basename(result.get('image_path', 'N/A'))}")
    
    # Test with command line argument
    if len(sys.argv) > 1:
        row_num = int(sys.argv[1])
        print(f"\n3. Command Line Prediction (Row {row_num}):")
        result = predictor.predict_row(row_num)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 