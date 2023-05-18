using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System.Diagnostics;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class UNet
    {
        private InferenceSession _session;
        private TextProcessing _textProcessing;
        private VaeDecoder _decoder;
        private LMSDiscreteScheduler _scheduler;
        private SafetyChecker _safetyChecker;
        private float[] _timesteps;

        public UNet(StableDiffusionConfig config)
        {
            var sessionOptions = config.GetSessionOptionsForEp();

            // Only has an effect if the models are converted with Olive, since other versions of the models
            // may not use the same symbolic dimension names. This isn't strictly required, but it helps DML EP
            // further optimize runtime performance.
            var batchSize = 1;
            sessionOptions.AddFreeDimensionOverrideByName("unet_sample_batch", 2 * batchSize);
            sessionOptions.AddFreeDimensionOverrideByName("unet_sample_channels", 4);
            sessionOptions.AddFreeDimensionOverrideByName("unet_sample_height", 64);
            sessionOptions.AddFreeDimensionOverrideByName("unet_sample_width", 64);
            sessionOptions.AddFreeDimensionOverrideByName("unet_time_batch", batchSize);
            sessionOptions.AddFreeDimensionOverrideByName("unet_hidden_batch", 2 * batchSize);
            sessionOptions.AddFreeDimensionOverrideByName("unet_hidden_sequence", 77);

            _session = new InferenceSession(config.UnetOnnxPath, sessionOptions);
            _textProcessing = new TextProcessing(config);
            _decoder = new VaeDecoder(config);
            _scheduler = new LMSDiscreteScheduler();
            // _scheduler = new EulerAncestralDiscreteScheduler();
            _timesteps = _scheduler.SetTimesteps(config.NumInferenceSteps);
            _safetyChecker = new SafetyChecker(config);
        }

        public static List<NamedOnnxValue> CreateUnetModelInput<T>(DenseTensor<T> encoderHiddenStates, DenseTensor<float> sample, float timeStep)
        {
            var sampleFloat16 = TensorHelper.ConvertFloatToFloat16(sample);

            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor<T>("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor<Float16>("sample", sampleFloat16),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<Float16>(new Float16[] { BitConverter.HalfToUInt16Bits((Half)timeStep) }, new int[] { 1 }))
            };
            
            return input;

        }

        public static DenseTensor<float> GenerateLatentSample(StableDiffusionConfig config, int seed, float initNoiseSigma)
        {
            return GenerateLatentSample(config.Height, config.Width, seed, initNoiseSigma);
        }
        public static DenseTensor<float> GenerateLatentSample(int height, int width, int seed, float initNoiseSigma)
        {
            var random = new Random(seed);
            var batchSize = 1;
            var channels = 4;
            var latents = new DenseTensor<float>(new[] { batchSize, channels, height / 8, width / 8 });
            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number

                // add noise to latents with * scheduler.init_noise_sigma
                // generate randoms that are negative and positive
                latentsArray[i] = (float)(standardNormalRand * initNoiseSigma);
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;

        }

        private static Tensor<float> performGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
        {
            for (int i = 0; i < noisePred.Length; i++)
            {
                noisePred.SetValue(i, (float)(noisePred.GetValue(i) + (float)guidanceScale * (noisePredText.GetValue(i) - noisePred.GetValue(i))));
            }
            return noisePred;
        }

        private static void performGuidance(DenseTensor<Float16> unetOutput, float guidanceScale)
        {
            // Assumes batch size == 1 so that first half of unetOutput is noisePred and second half is noisePredText
            var halfSize = (int)(unetOutput.Length / 2);

            Span<Float16> noisePred = unetOutput.Buffer.Span.Slice(0, halfSize);
            Span<Float16> noisePredText = unetOutput.Buffer.Span.Slice(halfSize, halfSize);

            for (int i = 0; i < halfSize; i++)
            {
                var noisePredVal = (float)BitConverter.UInt16BitsToHalf(noisePred[i]);
                var noisePredTextVal = (float)BitConverter.UInt16BitsToHalf(noisePredText[i]);
                noisePred[i] = BitConverter.HalfToUInt16Bits((Half)(noisePredVal + guidanceScale * (noisePredTextVal - noisePredVal)));
            };
        }

        public struct InferenceOutput
        {
            public SixLabors.ImageSharp.Image Image;
            public double IterationsPerSecond;
        }

        public InferenceOutput Inference(String prompt, StableDiffusionConfig config, Action<int>? stepCallback = null)
        {
            // Preprocess text
            var textEmbeddings = _textProcessing.PreprocessText<Float16>(prompt);

            _scheduler.ResetDerivatives();
            //  If you use the same seed, you will get the same image result.
            var seed = new Random().Next();
            //var seed = 329922609;
            Console.WriteLine($"Seed generated: {seed}");

            // create latent tensor
            var latents = GenerateLatentSample(config, seed, _scheduler.InitNoiseSigma);

            Stopwatch stopwatch = new Stopwatch();

            var input = new List<NamedOnnxValue>();

            stopwatch.Start();
            for (int t = 0; t < _timesteps.Length; t++)
            {
                // torch.cat([latents] * 2)
                var latentModelInput = TensorHelper.Duplicate(latents.ToArray(), new[] { 2, 4, config.Height / 8, config.Width / 8 });

                // latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)
                latentModelInput = _scheduler.ScaleInput(latentModelInput, _timesteps[t]);

                Console.WriteLine($"scaled model input {latentModelInput[0]} at step {t}.");
                input = CreateUnetModelInput<Float16>(textEmbeddings, latentModelInput, _timesteps[t]);

                // Run Inference
                var output = _session.Run(input);
                var outputTensor = (output.ToList().First().Value as DenseTensor<Float16>);

                // Perform guidance
                performGuidance(outputTensor, (float)config.GuidanceScale);

                // LMS Scheduler Step
                latents = _scheduler.Step(outputTensor, _timesteps[t], latents);
                //Console.WriteLine($"latents result after step {t} min {latents.Min()} max {latents.Max()}");

                if (stepCallback != null)
                {
                    stepCallback(t);
                }
            }
            stopwatch.Stop();

            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            latents = TensorHelper.MultipleTensorByFloat(latents.ToArray(), (float)(1.0f / 0.18215f), latents.Dimensions.ToArray());
            var latentsFloat16 = TensorHelper.ConvertFloatToFloat16(latents);
            var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("latent_sample", latentsFloat16) };

            // Decode image
            var imageResultTensor = _decoder.Decode(decoderInput);

            var isNotSafe = _safetyChecker.IsNotSafe(imageResultTensor, config);

            if (isNotSafe)
            {
                return new InferenceOutput { Image = null, IterationsPerSecond = config.NumInferenceSteps / (stopwatch.ElapsedMilliseconds / 1000.0) };

            }
            var image = VaeDecoder.ConvertToImage(imageResultTensor, config);
            return new InferenceOutput { Image = image, IterationsPerSecond = config.NumInferenceSteps / (stopwatch.ElapsedMilliseconds / 1000.0) };

        }

    }
}
