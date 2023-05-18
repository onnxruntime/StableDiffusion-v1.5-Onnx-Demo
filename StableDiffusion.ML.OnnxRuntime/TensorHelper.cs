using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;
using System;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class TensorHelper
    {
        public static DenseTensor<T> CreateTensor<T>(T[] data, int[] dimensions)
        {
            return new DenseTensor<T>(data, dimensions); ;
        }
        // function to cover float to float16 tensor
        public static DenseTensor<Float16> ConvertFloatToFloat16(DenseTensor<float> tensor)
        {
            ReadOnlySpan<float> fp32Values = tensor.Buffer.Span;

            DenseTensor<Float16> fp16Tensor = new DenseTensor<Float16>(tensor.Dimensions);
            Span<Float16> fp16Values = fp16Tensor.Buffer.Span;

            for (int i = 0; i < fp32Values.Length; i++)
            {
                fp16Values[i] = BitConverter.HalfToUInt16Bits((Half)fp32Values[i]);
            }

            return fp16Tensor;
        }
        public static DenseTensor<float> DivideTensorByFloat(float[] data, float value, int[] dimensions)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)(data[i] / value);
            }

            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> MultipleTensorByFloat(float[] data, float value, int[] dimensions)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)(data[i] * value);
            }

            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> MultipleTensorByFloat(Tensor<float> data, float value)
        {
            return MultipleTensorByFloat(data.ToArray(), value, data.Dimensions.ToArray());
        }

        public static DenseTensor<float> AddTensors(float[] sample, float[] sumTensor, int[] dimensions)
        {
            for(var i=0; i < sample.Length; i++)
            {
                sample[i] = (float)(sample[i] + sumTensor[i]);
            }
            return CreateTensor(sample, dimensions); ;
        }

        public static DenseTensor<float> AddTensors(Tensor<float> sample, Tensor<float> sumTensor)
        {
            return AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());
        }

        public static DenseTensor<float> SumTensors(Tensor<float>[] tensorArray, int[] dimensions)
        {
            var sumTensor = new DenseTensor<float>(dimensions);
            var sumArray = new float[sumTensor.Length];

            for (int m = 0; m < tensorArray.Count(); m++)
            {
                var tensorToSum = tensorArray[m].ToArray();
                for (var i = 0; i < tensorToSum.Length; i++)
                {
                    sumArray[i] +=(float)tensorToSum[i];
                }
            }

            return CreateTensor(sumArray, dimensions);
        }

        public static DenseTensor<float> Duplicate(float[] data, int[] dimensions)
        {
            data = data.Concat(data).ToArray();
            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(float[] sample, float[] subTensor, int[] dimensions)
        {
            for (var i = 0; i < sample.Length; i++)
            {
                sample[i] = (float)(sample[i] - subTensor[i]);
            }
            return CreateTensor(sample, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(Tensor<float> sample, Tensor<float> subTensor)
        {
            return SubtractTensors(sample.ToArray(), subTensor.ToArray(), sample.Dimensions.ToArray());
        }

        public static Tensor<float> GetRandomTensor(ReadOnlySpan<int> dimensions)
        {
            var random = new Random();
            var latents = new DenseTensor<float>(dimensions);
            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
                latentsArray[i] =(float)standardNormalRand;
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;

        }

        public static DenseTensor<float> ConvertFloat16ToFloat(DenseTensor<Float16>? result)
        {
            var floatTensor = new DenseTensor<float>(result.Dimensions.ToArray());
            var floatSpan = floatTensor.Buffer.Span;
            ReadOnlySpan<Float16> float16Array = result.Buffer.Span;

            for (int i = 0; i < floatSpan.Length; i++)
            {
                floatSpan[i] = (float)BitConverter.UInt16BitsToHalf(float16Array[i]);
            }

            return floatTensor;
        }
    }
}
