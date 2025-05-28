using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelApp
{
    public class WeldPredictor : IDisposable
    {
        private InferenceSession _session;
        private string _modelPath;
        private string _csvPath;

        public WeldPredictor()
        {
            // Try multiple possible locations for the model file
            string[] possibleModelPaths = {
                "RandomForest_production.onnx",
                "../RandomForest_production.onnx",
                "../../RandomForest_production.onnx",
                @"C:\Users\QuinnMazaris\Desktop\ONNX infrence\RandomForest_production.onnx"
            };

            foreach (string path in possibleModelPaths)
            {
                if (File.Exists(path))
                {
                    _modelPath = path;
                    break;
                }
            }

            if (_modelPath == null)
            {
                throw new FileNotFoundException("Could not find RandomForest_production.onnx");
            }

            // Try multiple possible locations for the EXACT preprocessed CSV file
            string[] possibleCsvPaths = {
                "exact_preprocessed_data.csv",
                "../exact_preprocessed_data.csv", 
                "../../exact_preprocessed_data.csv",
                @"C:\Users\QuinnMazaris\Desktop\ONNX infrence\exact_preprocessed_data.csv"
            };

            foreach (string path in possibleCsvPaths)
            {
                if (File.Exists(path))
                {
                    _csvPath = path;
                    break;
                }
            }

            if (_csvPath == null)
            {
                throw new FileNotFoundException("Could not find exact_preprocessed_data.csv. Please run create_exact_preprocessing.py first.");
            }

            Console.WriteLine($"Using model: {_modelPath}");
            Console.WriteLine($"Using EXACT preprocessed CSV: {_csvPath}");

            // Create ONNX Runtime session
            _session = new InferenceSession(_modelPath);

            // Print model information
            PrintModelInfo();
        }

        private void PrintModelInfo()
        {
            Console.WriteLine("\n=== Model Information ===");
            
            Console.WriteLine("Input metadata:");
            foreach (var input in _session.InputMetadata)
            {
                Console.WriteLine($"  {input.Key}: {input.Value.ElementType}, Shape: [{string.Join(", ", input.Value.Dimensions)}]");
            }

            Console.WriteLine("Output metadata:");
            foreach (var output in _session.OutputMetadata)
            {
                try
                {
                    Console.WriteLine($"  {output.Key}: {output.Value.ElementType}, Shape: [{string.Join(", ", output.Value.Dimensions)}]");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  {output.Key}: (Could not read metadata - {ex.Message})");
                }
            }
        }

        public string PredictRow(int rowNumber)
        {
            try
            {
                // Read the EXACT preprocessed CSV file
                var lines = File.ReadAllLines(_csvPath);
                
                if (rowNumber < 0 || rowNumber >= lines.Length - 1) // -1 because first line is header
                {
                    return $"Error: Row {rowNumber} is out of range. Available rows: 0 to {lines.Length - 2}";
                }

                // Skip header (line 0) and get the requested row
                var dataLine = lines[rowNumber + 1];
                var values = dataLine.Split(',');

                // Convert to float array (all 23 features, exactly matching training)
                var features = values.Select(v => float.Parse(v.Trim())).ToArray();

                if (features.Length != 23)
                {
                    return $"Error: Expected 23 features, got {features.Length}";
                }

                // Create input tensor with exact shape expected by model
                var inputTensor = new DenseTensor<float>(features, new[] { 1, 23 });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
                };

                // Run inference
                using var results = _session.Run(inputs);
                var output = results.First().AsEnumerable<long>().First();

                // Convert output to readable format
                string prediction = output == 0 ? "Good" : "Bad";
                
                return $"{{\"row\": {rowNumber}, \"prediction\": \"{prediction}\", \"raw_output\": {output}}}";
            }
            catch (Exception ex)
            {
                return $"Error predicting row {rowNumber}: {ex.Message}";
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                using var predictor = new WeldPredictor();

                if (args.Length > 0 && int.TryParse(args[0], out int rowNumber))
                {
                    // Predict specific row
                    var result = predictor.PredictRow(rowNumber);
                    Console.WriteLine(result);
                }
                else
                {
                    // Test a comprehensive set of predictions
                    Console.WriteLine("\n=== EXACT FEATURE PREDICTIONS ===");
                    
                    // Test some specific rows including known Bad samples
                    int[] testRows = { 0, 1, 2, 480, 1000, 2000, 2411 }; // Include last row
                    
                    foreach (int row in testRows)
                    {
                        var result = predictor.PredictRow(row);
                        Console.WriteLine($"Row {row}: {result}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}
