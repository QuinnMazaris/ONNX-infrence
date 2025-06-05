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
        
        public string CsvPath => _csvPath;

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
                    return $"Error: label is not a tensor, type: {labelResult.ValueType}";
                }

                // Get probability output for confidence
                var probResult = results.FirstOrDefault(r => r.Name == "output_probability");
                float confidence = 0.5f; // Default confidence
                
                if (probResult != null)
                {
                    // Try to extract directly from the probability result if it's a map
                    if (probResult.ValueType == OnnxValueType.ONNX_TYPE_MAP)
                    {
                        try
                        {
                            var mapValue = probResult.AsEnumerable<IDictionary<long, float>>();
                            if (mapValue != null)
                            {
                                var probDict = mapValue.First();
                                if (probDict.ContainsKey(1))
                                {
                                    confidence = probDict[1];
                                }
                                else if (probDict.ContainsKey(0))
                                {
                                    confidence = 1.0f - probDict[0];
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Could not extract directly from map: {ex.Message}");
                        }
                    }
                    else if (probResult.ValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                    {
                        try
                        {
                            var tensor = probResult.AsTensor<float>();
                            Console.WriteLine($"Tensor length: {tensor.Length}");
                            var probArray = tensor.ToArray<float>();
                            Console.WriteLine($"Probabilities: [{string.Join(", ", probArray)}]");
                            if (probArray.Length > 1)
                            {
                                confidence = probArray[1]; // Probability of class 1 (Bad)
                            }
                            else if (probArray.Length == 1)
                            {
                                confidence = probArray[0]; // Single probability value
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Could not extract probability from tensor: {ex.Message}");
                        }
                    }
                    else if (probResult.ValueType == OnnxValueType.ONNX_TYPE_SEQUENCE)
                    {
                        try
                        {
                                                    // Try to extract from sequence
                        bool extracted = false;
                        try
                        {
                            var genericSequence = probResult.AsEnumerable<object>();
                            if (genericSequence != null)
                            {
                                var firstItem = genericSequence.FirstOrDefault();
                                if (firstItem != null)
                                {
                                    // If it's a DisposableNamedOnnxValue, extract its value
                                    if (firstItem is Microsoft.ML.OnnxRuntime.DisposableNamedOnnxValue namedValue)
                                    {
                                        if (namedValue.ValueType == OnnxValueType.ONNX_TYPE_MAP)
                                        {
                                                
                                                // Extract probability from map using reflection
                                                try
                                                {
                                                    var valueProperty = namedValue.GetType().GetProperty("Value");
                                                    if (valueProperty != null)
                                                    {
                                                        var actualValue = valueProperty.GetValue(namedValue);
                                                        
                                                        // If it's enumerable, try to iterate through it
                                                        if (actualValue is System.Collections.IEnumerable enumerable)
                                                        {
                                                            foreach (var item in enumerable)
                                                            {
                                                                // If it's a KeyValuePair<long, float>
                                                                if (item is System.Collections.Generic.KeyValuePair<long, float> kvpLongFloat)
                                                                {
                                                                    if (kvpLongFloat.Key == 1)
                                                                    {
                                                                        confidence = kvpLongFloat.Value;
                                                                        extracted = true;
                                                                    }
                                                                    else if (kvpLongFloat.Key == 0)
                                                                    {
                                                                        confidence = 1.0f - kvpLongFloat.Value;
                                                                        extracted = true;
                                                                    }
                                                                }
                                                                
                                                                if (extracted) break; // Exit loop if we found what we need
                                                            }
                                                        }
                                                    }
                                                }
                                                catch (Exception ex)
                                                {
                                                    Console.WriteLine($"Could not extract probability from map: {ex.Message}");
                                                }
                                                
                                                // If reflection didn't work, try the standard approaches
                                                if (!extracted)
                                                {
                                                    try
                                                    {
                                                        Console.WriteLine("Trying standard map extraction...");
                                                        var mapValue = namedValue.AsEnumerable<IDictionary<long, float>>();
                                                        if (mapValue != null)
                                                        {
                                                            var probDict = mapValue.First();
                                                            Console.WriteLine($"Probability map: {string.Join(", ", probDict.Select(kv => $"{kv.Key}:{kv.Value}"))}");
                                                            if (probDict.ContainsKey(1))
                                                            {
                                                                confidence = probDict[1];
                                                                Console.WriteLine($"Found class 1 probability: {confidence}");
                                                                extracted = true;
                                                            }
                                                            else if (probDict.ContainsKey(0))
                                                            {
                                                                confidence = 1.0f - probDict[0];
                                                                Console.WriteLine($"Found class 0 probability, calculated confidence: {confidence}");
                                                                extracted = true;
                                                            }
                                                        }
                                                    }
                                                    catch (Exception ex)
                                                    {
                                                        Console.WriteLine($"Standard map extraction failed: {ex.Message}");
                                                    }
                                                }
                                            }
                                            else if (namedValue.ValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                                            {
                                                Console.WriteLine("Value is a tensor");
                                                try
                                                {
                                                    var tensor = namedValue.AsTensor<float>();
                                                    var probArray = tensor.ToArray<float>();
                                                    Console.WriteLine($"Probabilities from tensor: [{string.Join(", ", probArray)}]");
                                                    if (probArray.Length > 1)
                                                    {
                                                        confidence = probArray[1]; // Probability of class 1 (Bad)
                                                        Console.WriteLine($"Using class 1 probability: {confidence}");
                                                        extracted = true;
                                                    }
                                                    else if (probArray.Length == 1)
                                                    {
                                                        confidence = probArray[0]; // Single probability value
                                                        Console.WriteLine($"Using single probability: {confidence}");
                                                        extracted = true;
                                                    }
                                                }
                                                catch (Exception ex)
                                                {
                                                    Console.WriteLine($"Could not extract from tensor: {ex.Message}");
                                                }
                                            }
                                        }
                                        // If it's a dictionary, try to extract from it
                                        else if (firstItem is IDictionary<object, object> dict)
                                        {
                                            Console.WriteLine("First item is a dictionary");
                                            foreach (var kvp in dict)
                                            {
                                                Console.WriteLine($"  Key: {kvp.Key} (type: {kvp.Key?.GetType()}), Value: {kvp.Value} (type: {kvp.Value?.GetType()})");
                                                if (kvp.Key?.ToString() == "1" && kvp.Value is float f1)
                                                {
                                                    confidence = f1;
                                                    Console.WriteLine($"Found class 1 probability: {confidence}");
                                                    extracted = true;
                                                }
                                                else if (kvp.Key?.ToString() == "0" && kvp.Value is float f0)
                                                {
                                                    confidence = 1.0f - f0;
                                                    Console.WriteLine($"Found class 0 probability, calculated confidence: {confidence}");
                                                    extracted = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Could not extract from sequence: {ex.Message}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Could not extract probability from sequence: {ex.Message}");
                        }
                    }
                }
                else
                {
                    Console.WriteLine("No probability output found - using default confidence");
                }

                // Apply prediction threshold after confidence is calculated
                if (confidence >= _config.ModelSettings.PredictionThreshold)
                {
                    prediction = 1; // Classify as Bad
                }
                else
                {
                    prediction = 0; // Classify as Good
                }
                
                // Debug: Print final prediction and confidence
                Console.WriteLine($"Row {rowNumber}: Final Prediction: {prediction}, Confidence: {confidence:F4}, Threshold: {_config.ModelSettings.PredictionThreshold}");

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
                    Console.WriteLine($"\n=== PREDICTING SINGLE ROW {rowNumber} ===");
                    var result = predictor.PredictRow(rowNumber);
                    Console.WriteLine(result);
                }
                else
                {
                    // Loop through all rows and predict each one
                    Console.WriteLine("\n=== PREDICTING ALL ROWS ===");
                    
                    // Read the preprocessed CSV to get total row count
                    var csvPath = predictor.CsvPath;
                    var lines = File.ReadAllLines(csvPath);
                    int totalRows = lines.Length - 1; // Subtract 1 for header
                    
                    Console.WriteLine($"Total rows to predict: {totalRows}");
                    Console.WriteLine("Starting predictions...\n");
                    
                    // Output header for CSV format
                    Console.WriteLine("PREDICTIONS_START");
                    
                    // Predict each row
                    for (int row = 0; row < totalRows; row++)
                    {
                        var result = predictor.PredictRow(row);
                        Console.WriteLine(result);
                        
                        // Add a small delay every 100 rows to prevent overwhelming output
                        if ((row + 1) % 100 == 0)
                        {
                            Console.WriteLine($"--- Completed {row + 1}/{totalRows} predictions ---");
                        }
                    }
                    
                    Console.WriteLine("PREDICTIONS_END");
                    Console.WriteLine($"\n=== COMPLETED ALL {totalRows} PREDICTIONS ===");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}
