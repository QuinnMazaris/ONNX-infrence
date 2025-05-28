using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelApp
{
    public class WeldPredictor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly ModelConfig _config;
        private readonly DataPreprocessor _preprocessor;
        private readonly string _csvPath;

        public WeldPredictor()
        {
            // Load configuration
            _config = ModelConfig.Load();

            // Find model file
            _config.ModelSettings.ModelPath = _config.FindFile(_config.ModelSettings.ModelPath, _config.SearchPaths.ModelPaths);
            if (_config.ModelSettings.ModelPath == null)
            {
                throw new FileNotFoundException("Could not find RandomForest_production.onnx");
            }

            // Find and preprocess data if needed
            _config.ModelSettings.RawDataPath = _config.FindFile(_config.ModelSettings.RawDataPath, _config.SearchPaths.DataPaths);
            if (_config.ModelSettings.RawDataPath == null)
            {
                throw new FileNotFoundException("Could not find training_data.csv");
            }

            // Find feature mapping file
            _config.ModelSettings.FeatureMappingPath = _config.FindFile(_config.ModelSettings.FeatureMappingPath, _config.SearchPaths.FeatureMappingPaths);
            if (_config.ModelSettings.FeatureMappingPath == null)
            {
                throw new FileNotFoundException("Could not find exact_training_features.json");
            }

            _preprocessor = new DataPreprocessor(_config);
            
            // Check if preprocessed data exists, if not, create it
            if (!File.Exists(_config.ModelSettings.PreprocessedDataPath))
            {
                Console.WriteLine("Preprocessed data not found. Running preprocessing...");
                _preprocessor.PreprocessData();
            }

            _csvPath = _config.ModelSettings.PreprocessedDataPath;
            Console.WriteLine($"Using model: {_config.ModelSettings.ModelPath}");
            Console.WriteLine($"Using preprocessed CSV: {_csvPath}");

            // Create ONNX Runtime session
            _session = new InferenceSession(_config.ModelSettings.ModelPath);

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
                // Read the preprocessed CSV file
                var lines = File.ReadAllLines(_csvPath);
                
                if (rowNumber < 0 || rowNumber >= lines.Length - 1) // -1 because first line is header
                {
                    return $"Error: Row {rowNumber} is out of range. Available rows: 0 to {lines.Length - 2}";
                }

                // Skip header (line 0) and get the requested row
                var dataLine = lines[rowNumber + 1];
                var values = dataLine.Split(',');

                // Convert to float array using preprocessor
                var features = _preprocessor.ExtractFeatures(values);

                // Create input tensor with exact shape expected by model
                var inputTensor = new DenseTensor<float>(features, new[] { 1, features.Length });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
                };

                // Run inference
                using var results = _session.Run(inputs);
                
                // Debug: Print all outputs
                Console.WriteLine($"Number of outputs: {results.Count()}");
                foreach (var result in results)
                {
                    Console.WriteLine($"Output name: {result.Name}, Type: {result.ValueType}");
                }

                // Get the label output (should be the prediction)
                var labelResult = results.FirstOrDefault(r => r.Name == "output_label");
                if (labelResult == null)
                {
                    return $"Error: Could not find output_label in model results";
                }
                
                Console.WriteLine($"Label result name: {labelResult.Name}");
                
                long prediction;
                if (labelResult.ValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                {
                    var tensor = labelResult.AsTensor<long>();
                    prediction = tensor.First();
                }
                else
                {
                    return $"Error: output_label is not a tensor, type: {labelResult.ValueType}";
                }

                // Get probability output for confidence
                var probResult = results.FirstOrDefault(r => r.Name == "output_probability");
                float confidence = 0.5f; // Default confidence
                
                if (probResult != null && probResult.ValueType == OnnxValueType.ONNX_TYPE_SEQUENCE)
                {
                    try
                    {
                        // Try to extract probability - this might be a sequence of probabilities
                        var probSequence = probResult.AsEnumerable<IEnumerable<float>>();
                        if (probSequence != null)
                        {
                            var probArray = probSequence.First().ToArray();
                            if (probArray.Length > 1)
                            {
                                confidence = probArray[1]; // Probability of class 1 (Bad)
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Could not extract probability: {ex.Message}");
                    }
                }

                // Convert output to readable format
                string predictionText = prediction == 1 ? "Bad" : "Good";
                
                return $"{{\"row\": {rowNumber}, \"prediction\": \"{predictionText}\", \"confidence\": {confidence:F4}, \"label\": {prediction}}}";
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
