using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OnnxModelApp
{
    public class ModelSettings
    {
        public required string ModelPath { get; set; }
        public required string RawDataPath { get; set; }
        public required string PreprocessedDataPath { get; set; }
        public required string FeatureMappingPath { get; set; }
        public float PredictionThreshold { get; set; }
    }

    public class SearchPaths
    {
        public required List<string> ModelPaths { get; set; }
        public required List<string> DataPaths { get; set; }
        public List<string> FeatureMappingPaths { get; set; } = new List<string>();
    }

    public class ModelConfig
    {
        public required ModelSettings ModelSettings { get; set; }
        public required SearchPaths SearchPaths { get; set; }

        private static string[] GetPossibleConfigPaths()
        {
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var currentDir = Directory.GetCurrentDirectory();
            var projectDir = Path.GetDirectoryName(typeof(ModelConfig).Assembly.Location);

            return new[]
            {
                Path.Combine(baseDir, "appsettings.json"),
                Path.Combine(currentDir, "appsettings.json"),
                Path.Combine(projectDir ?? ".", "appsettings.json"),
                Path.Combine(currentDir, "OnnxModelApp", "appsettings.json"),
                "appsettings.json"
            };
        }

        public static ModelConfig Load()
        {
            var possiblePaths = GetPossibleConfigPaths();
            string? configPath = null;

            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    configPath = path;
                    Console.WriteLine($"Found config file at: {path}");
                    break;
                }
            }

            if (configPath == null)
            {
                Console.WriteLine("Searched for config file in:");
                foreach (var path in possiblePaths)
                {
                    Console.WriteLine($"  - {path}");
                }
                throw new FileNotFoundException("Could not find appsettings.json in any of the expected locations");
            }

            string jsonString = File.ReadAllText(configPath);
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
            
            var config = JsonSerializer.Deserialize<ModelConfig>(jsonString, options);
            
            if (config == null)
            {
                throw new InvalidOperationException("Failed to deserialize configuration");
            }

            return config;
        }

        public string? FindFile(string basePath, List<string> searchPaths)
        {
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var currentDir = Directory.GetCurrentDirectory();
            var projectDir = Path.GetDirectoryName(typeof(ModelConfig).Assembly.Location);

            var searchLocations = new[]
            {
                baseDir,
                currentDir,
                projectDir ?? ".",
                Path.GetDirectoryName(basePath) ?? "."
            };

            foreach (var location in searchLocations)
            {
                foreach (var path in searchPaths)
                {
                    string fullPath = Path.Combine(location, path);
                    if (File.Exists(fullPath))
                    {
                        Console.WriteLine($"Found file at: {fullPath}");
                        return fullPath;
                    }
                }
            }

            Console.WriteLine($"Could not find file. Searched in:");
            foreach (var location in searchLocations)
            {
                foreach (var path in searchPaths)
                {
                    Console.WriteLine($"  - {Path.Combine(location, path)}");
                }
            }
            return null;
        }
    }
} 