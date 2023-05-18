using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class VaeDecoder
    {
        private InferenceSession _session;

        public VaeDecoder(StableDiffusionConfig config)
        {
            var sessionOptions = config.GetSessionOptionsForEp();
            _session = new InferenceSession(config.VaeDecoderOnnxPath, sessionOptions);
        }

        public DenseTensor<float> Decode(List<NamedOnnxValue> input)
        {
           // Run session and send the input data in to get inference output. 
            var output = _session.Run(input);
            var result = (output.ToList().First().Value as DenseTensor<Float16>);
            var resultFloat = TensorHelper.ConvertFloat16ToFloat(result);

            return resultFloat;
        }

        // create method to convert float array to an image with imagesharp
        public static Image<Rgba32> ConvertToImage(DenseTensor<float> output, StableDiffusionConfig config, int width = 512, int height = 512)
        {
            var result = new Image<Rgba32>(width, height);

            int spatialSize = width * height;
            ReadOnlySpan<float> rChannel = output.Buffer.Span.Slice(0);
            ReadOnlySpan<float> gChannel = output.Buffer.Span.Slice(spatialSize);
            ReadOnlySpan<float> bChannel = output.Buffer.Span.Slice(2* spatialSize);
            int pixelIndex = 0;

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgba32(
                        (byte)(Math.Round(Math.Clamp((rChannel[pixelIndex] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((gChannel[pixelIndex] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((bChannel[pixelIndex] / 2 + 0.5), 0, 1) * 255))
                    );
                    pixelIndex++;
                }
            }

            var imageName = $"sd_image_{DateTime.Now.ToString("yyyyMMddHHmmssfff")}.png";
            var imagePath = Path.Combine(Directory.GetCurrentDirectory(), config.ImageOutputPath, imageName);

            result.Save(imagePath);

            Console.WriteLine($"Image saved to: {imagePath}");

            return result;
        }
    }
}
