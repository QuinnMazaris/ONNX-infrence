using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelApp
{
    public class PredictionResult
    {
        public required string Result { get; set; }
        public float Confidence { get; set; }
    }

    public class WeldPredictor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly float _predictionThreshold = 0.3f; // Setting to a common default threshold
        private readonly string _onnxModelFileName = "RandomForest_production.onnx"; // Hardcoded model file name

        public WeldPredictor()
        {
            // Construct the full path to the ONNX model file based on the application's base directory
            string fullOnnxModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, _onnxModelFileName);

            if (!File.Exists(fullOnnxModelPath))
            {
                throw new FileNotFoundException($"ONNX model file not found: {fullOnnxModelPath}");
            }

            Console.WriteLine($"Using model: {fullOnnxModelPath}");

            // Create ONNX Runtime session
            _session = new InferenceSession(fullOnnxModelPath);

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

        public PredictionResult Predict(float[] features)
        {
            try
            {
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
                    throw new InvalidOperationException("Could not find output_label in model results");
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
                    throw new InvalidOperationException($"Label is not a tensor, type: {labelResult.ValueType}");
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
                            bool extracted = false;
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
                                                            if (item is KeyValuePair<long, float> kvpLongFloat)
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
                                                Console.WriteLine($"Could not extract probability from map using reflection: {ex.Message}");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Could not extract probability from sequence: {ex.Message}");
                        }
                    }
                }

                // Corrected logic: if confidence (probability of Bad) is >= threshold, then it's Bad.
                string resultLabel = (confidence >= _predictionThreshold) ? "Bad" : "Good";

                return new PredictionResult { Result = resultLabel, Confidence = confidence };
            }
            catch (Exception ex)
            {
                return new PredictionResult { Result = $"Error: {ex.Message}", Confidence = 0.0f };
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
