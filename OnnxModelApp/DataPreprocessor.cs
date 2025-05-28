using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;

namespace OnnxModelApp
{
    public class FeatureMapping
    {
        [JsonPropertyName("feature_count")]
        public required int FeatureCount { get; set; }
        
        [JsonPropertyName("feature_columns")]
        public required List<string> FeatureColumns { get; set; }
        
        [JsonPropertyName("preprocessing_steps")]
        public Dictionary<string, object>? PreprocessingSteps { get; set; }
        
        [JsonPropertyName("sample_features")]
        public Dictionary<string, object>? SampleFeatures { get; set; }
    }

    public class DataPreprocessor
    {
        private readonly ModelConfig _config;
        private readonly FeatureMapping _featureMapping;
        private readonly int _expectedFeatureCount;

        public DataPreprocessor(ModelConfig config)
        {
            _config = config;
            _featureMapping = LoadFeatureMapping();
            _expectedFeatureCount = _featureMapping.FeatureCount;
        }

        private FeatureMapping LoadFeatureMapping()
        {
            if (!File.Exists(_config.ModelSettings.FeatureMappingPath))
            {
                throw new FileNotFoundException($"Feature mapping file not found: {_config.ModelSettings.FeatureMappingPath}");
            }

            string jsonString = File.ReadAllText(_config.ModelSettings.FeatureMappingPath);
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
            
            var mapping = JsonSerializer.Deserialize<FeatureMapping>(jsonString, options);
            
            if (mapping == null)
            {
                throw new InvalidOperationException("Failed to deserialize feature mapping");
            }

            Console.WriteLine("\n=== Feature Mapping Information ===");
            Console.WriteLine($"Feature count: {mapping.FeatureCount}");
            Console.WriteLine("\nFeature columns in exact order:");
            for (int i = 0; i < mapping.FeatureColumns.Count; i++)
            {
                Console.WriteLine($"  {i,2}: {mapping.FeatureColumns[i]}");
            }

            if (mapping.PreprocessingSteps != null && mapping.PreprocessingSteps.Count > 0)
            {
                Console.WriteLine("\nPreprocessing steps:");
                foreach (var step in mapping.PreprocessingSteps)
                {
                    Console.WriteLine($"  {step.Key}: {step.Value}");
                }
            }

            return mapping;
        }

        public void PreprocessData()
        {
            if (!File.Exists(_config.ModelSettings.RawDataPath))
            {
                throw new FileNotFoundException($"Raw data file not found: {_config.ModelSettings.RawDataPath}");
            }

            using var reader = new StreamReader(_config.ModelSettings.RawDataPath);
            using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture));
            
            // Read all records
            var records = csv.GetRecords<dynamic>().ToList();
            
            // Extract features in the exact order, handling one-hot encoding for Weld column
            var preprocessedData = new List<string[]>();
            preprocessedData.Add(_featureMapping.FeatureColumns.ToArray()); // Add header

            foreach (var record in records)
            {
                var row = new List<string>();
                var recordDict = (IDictionary<string, object>)record;
                
                foreach (var column in _featureMapping.FeatureColumns)
                {
                    if (column.StartsWith("Weld_"))
                    {
                        // Handle one-hot encoding for Weld columns
                        var weldType = column.Substring(5); // Remove "Weld_" prefix
                        var actualWeldValue = recordDict["Weld"]?.ToString() ?? "";
                        row.Add(actualWeldValue == weldType ? "1" : "0");
                    }
                    else
                    {
                        // Handle regular columns
                        var value = recordDict.ContainsKey(column) ? recordDict[column] : null;
                        row.Add(value?.ToString() ?? "0");
                    }
                }
                preprocessedData.Add(row.ToArray());
            }

            // Write preprocessed data
            using var writer = new StreamWriter(_config.ModelSettings.PreprocessedDataPath);
            using var csvWriter = new CsvWriter(writer, new CsvConfiguration(CultureInfo.InvariantCulture));
            
            foreach (var row in preprocessedData)
            {
                csvWriter.WriteField(row);
                csvWriter.NextRecord();
            }

            Console.WriteLine($"\nPreprocessed data saved to: {_config.ModelSettings.PreprocessedDataPath}");
            Console.WriteLine($"Total rows processed: {preprocessedData.Count - 1}"); // -1 for header
        }

        public float[] ExtractFeatures(string[] rawData)
        {
            if (rawData.Length != _expectedFeatureCount)
            {
                throw new ArgumentException($"Expected {_expectedFeatureCount} features, got {rawData.Length}");
            }

            return rawData.Select(v => float.Parse(v.Trim())).ToArray();
        }

        public List<string> GetFeatureColumns() => _featureMapping.FeatureColumns;
    }
} 