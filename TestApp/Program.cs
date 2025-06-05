using OnnxModelApp; // This using directive is important!
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq; // Added for LINQ operations like .Zip()
using System.Text.Json; // Added for JSON deserialization

// Class to hold the structure of the feature mapping JSON
public class FeatureMapping
{
    public List<string> feature_columns { get; set; } = new List<string>();
}

public class MyPredictionApp
{
    // Define your raw input vector here. Ensure the order matches the feature_columns in exact_training_features.json
    static float[] rawInputVector = new float[] { 0.380161f, 962f, 0.380161f, 3f, 0.947493f, 0.048672f, 0.003762f, 0.000050f, 0.000019f, 0.000002f, 0.000001f, 0.000000f, 0.000000f, 0.000000f, 0.947493f, 0.898821f, 96.054138f, 91.519630f, 273.355316f, 175.810791f, 1.000000f, 0.867711f, 133.665211f, 184.704727f, 84.291161f, 177.301178f };

    public static void Main(string[] args)
    {
        try
        {
            // Initialize the predictor with your model and threshold
            using (var predictor = new WeldPredictor())
            {
                // Get the prediction result
                PredictionResult result = predictor.Predict(rawInputVector);

                Console.WriteLine($"Prediction Result: {result.Result}"); // e.g., "Pass" or "Fail"
                Console.WriteLine($"Confidence: {result.Confidence:F4}"); // e.g., 0.8765
            }
        }
        catch (FileNotFoundException ex)
        {
            Console.WriteLine($"Error: One or more required files were not found. Please check your paths. {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An unexpected error occurred during prediction: {ex.Message}");
        }
    }
}
