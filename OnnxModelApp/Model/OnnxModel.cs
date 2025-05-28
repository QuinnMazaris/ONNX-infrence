using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelApp.Model
{
    public class OnnxModel
    {
        private readonly InferenceSession _session;

        public OnnxModel(string modelPath)
        {
            try
            {
                _session = new InferenceSession(modelPath);
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to load ONNX model: {ex.Message}", ex);
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
} 