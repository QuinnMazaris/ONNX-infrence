using System;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelApp.Model
{
    public static class TensorHelper
    {
        public static DenseTensor<float> CreateTensor(float[] data, int[] dimensions)
        {
            try
            {
                return new DenseTensor<float>(data, dimensions);
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to create tensor: {ex.Message}", ex);
            }
        }
    }
} 