using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SimpleEvaluator
{
    public class FeatureMapping
    {
        public required int FeatureCount { get; set; }
        public required List<string> FeatureColumns { get; set; }
    }

    public class OnnxPredictor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly FeatureMapping _mapping;

        public OnnxPredictor(string modelPath, string mappingPath)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException("Model file not found", modelPath);
            if (!File.Exists(mappingPath))
                throw new FileNotFoundException("Feature mapping file not found", mappingPath);

            _session = new InferenceSession(modelPath);
            string json = File.ReadAllText(mappingPath);
            _mapping = JsonSerializer.Deserialize<FeatureMapping>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            }) ?? throw new InvalidOperationException("Failed to parse feature mapping");
        }

        public long Predict(Dictionary<string, object> input)
        {
            var features = new float[_mapping.FeatureColumns.Count];

            for (int i = 0; i < _mapping.FeatureColumns.Count; i++)
            {
                string column = _mapping.FeatureColumns[i];

                if (column.StartsWith("Weld_", StringComparison.OrdinalIgnoreCase))
                {
                    string weldVal = input.TryGetValue("Weld", out var val) ? Convert.ToString(val) ?? string.Empty : string.Empty;
                    string expected = column.Substring(5);
                    features[i] = string.Equals(weldVal, expected, StringComparison.OrdinalIgnoreCase) ? 1f : 0f;
                }
                else if (input.TryGetValue(column, out var val))
                {
                    features[i] = float.TryParse(Convert.ToString(val), out var f) ? f : 0f;
                }
                else
                {
                    features[i] = 0f;
                }
            }

            var tensor = new DenseTensor<float>(features, new[] { 1, features.Length });
            string inputName = _session.InputMetadata.First().Key;
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) });
            return results.First().AsEnumerable<long>().First();
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}
