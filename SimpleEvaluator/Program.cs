using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace SimpleEvaluator
{
    record Config(string ModelPath, string FeatureMappingPath);

    class Program
    {
        static int Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: dotnet run --project SimpleEvaluator <config.json> <features.json>");
                return 1;
            }

            if (!File.Exists(args[0]) || !File.Exists(args[1]))
            {
                Console.WriteLine("Config or feature file not found");
                return 1;
            }

            var config = JsonSerializer.Deserialize<Config>(File.ReadAllText(args[0]));
            var featureValues = JsonSerializer.Deserialize<Dictionary<string, object>>(File.ReadAllText(args[1]));

            if (config == null || featureValues == null)
            {
                Console.WriteLine("Failed to parse config or features");
                return 1;
            }

            using var predictor = new OnnxPredictor(config.ModelPath, config.FeatureMappingPath);
            long prediction = predictor.Predict(featureValues);
            Console.WriteLine(prediction);
            return 0;
        }
    }
}
