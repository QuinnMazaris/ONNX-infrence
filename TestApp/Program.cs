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
    // Good
    //static float[] rawInputVector = new float[] { 0.380161f, 962f, 0.380161f, 3f, 0.947493f, 0.048672f, 0.003762f, 0.000050f, 0.000019f, 0.000002f, 0.000001f, 0.000000f, 0.000000f, 0.000000f, 0.947493f, 0.898821f, 96.054138f, 91.519630f, 273.355316f, 175.810791f, 1.000000f, 0.867711f, 133.665211f, 184.704727f, 84.291161f, 177.301178f };
    static float[] rawInputVector = new float[] { 0.544159f,1928f,0.544159f,5f,0.999972f,0.000015f,0.000010f,0.000001f,0.000001f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.999972f,0.999957f,75.514206f,92.535889f,318.379089f,167.688690f,1.000000f,0.974440f,130.112289f,196.946648f,75.152802f,242.864883f };
    // Bad
    //static float[] rawInputVector = new float[] { 0.146405f,0f,0.000000f,1f,0.996469f,0.002116f,0.001362f,0.000033f,0.000015f,0.000003f,0.000002f,0.000001f,0.000000f,0.000000f,0.996469f,0.994353f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f };
    //static float[] rawInputVector = new float[] {0.486316f,1348f,0.486316f,2f,1.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,1.000000f,1.000000f,27.562653f,86.260025f,352.886353f,186.670517f,0.000000f,0.959542f,136.465271f,190.224503f,100.410492f,325.323700f};




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
