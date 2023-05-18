using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using StableDiffusion.ML.OnnxRuntime;
using System.IO;
using System.Windows.Threading;
using System.Configuration;
using StableDiffusionGeneration.Common;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;

namespace StableDiffusionGeneration.Model
{
    public class GeneratorModel: ModelBase
    {
        private double _percentage;
        private long _lastTimeMilliseconds;
        private double _iterationsPerSecond;
        private StableDiffusionConfig _config;
        private UNet _unet;

        public double Percentage
        {
            get => _percentage;
            set => SetProperty(ref _percentage, value);
        }

        public long LastTimeMilliseconds
        {
            get => _lastTimeMilliseconds;
        }

        public long NumInferenceSteps
        {
            get => _config.NumInferenceSteps;
        }

        public double IterationsPerSecond
        {
            get => _iterationsPerSecond;
        }

        public GeneratorModel()
        {
            var modelsPath = Path.Combine(Directory.GetCurrentDirectory(), "Onnx", "fp16");

            _config = new StableDiffusionConfig
            {
                // Number of denoising steps
                NumInferenceSteps = 50,
                // Scale for classifier-free guidance
                GuidanceScale = 7.5,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cpu,
                // Set GPU Device ID.
                DeviceId = 1,
                // Update paths to your models
                TextEncoderOnnxPath = $@"{modelsPath}\text_encoder\model.onnx",
                UnetOnnxPath = $@"{modelsPath}\unet\model.onnx",
                VaeDecoderOnnxPath = $@"{modelsPath}\vae_decoder\model.onnx",
                SafetyModelPath = $@"{modelsPath}\safety_checker\model.onnx",
                ImageOutputPath = "NONE",
            };

            _unet = new UNet(_config);
        }

        //Generate Fake Data
        public ObservableCollection<ImageSource> GenerateFakeData(string description, int numberOfImages)
        {
            string imagePath = "pack://application:,,,/Assets/biaafpwi.png"; // Replace with the actual file path for generated images. 
            var tempImages = new ObservableCollection<ImageSource>();
            for (int i = 0; i < numberOfImages; i++)
            {
                var bitmapImage = new BitmapImage(new Uri(imagePath, UriKind.Absolute));
                ImageSource imageSource = bitmapImage;
                tempImages.Add(imageSource);
            }

            return tempImages;
        }

        public async Task<string[]> GenerateImages(string description, int numberOfImages, Action<int>? stepCallback = null)
        {
            Percentage = 0;
            // No iterations specified.
            if (numberOfImages == 0)
            {
                return new string[0];
            }

            var result = await Task.Run(string[] () =>
            {
                Stopwatch timer = new Stopwatch();
                timer.Start();

                var imageDestination = Path.Combine(Directory.GetCurrentDirectory(), $"Images-{DateTime.Now.Ticks}");

                var config = _config;
                config.ImageOutputPath = imageDestination;

                int totalSteps = numberOfImages * config.NumInferenceSteps;

                // Create the folder if it does not exist
                if (!Directory.Exists(imageDestination))
                {
                    Directory.CreateDirectory(imageDestination);
                }

                for (int i = 1; i <= numberOfImages; i++) 
                {
                    // Inference Stable Diff
                    var output = _unet.Inference(
                        description, 
                        config, 
                        (stepIndex) => { Percentage = ((double)((stepIndex + 1) + (i - 1) * config.NumInferenceSteps) / (double)totalSteps) * 100.0; }
                    );

                    _iterationsPerSecond = output.IterationsPerSecond;

                    // If image failed or was unsafe it will return null.
                    if (output.Image == null)
                    {
                        Console.WriteLine($"There was an error generating image {i}.");
                    }
                }

                // Output image path
                var path = Path.Combine(imageDestination);

                string[] imagePaths = Directory.GetFiles(imageDestination, "*.*", SearchOption.AllDirectories)
                                        .Where(path => new string[] { ".jpg", ".jpeg", ".png", ".bmp", ".gif" }
                                        .Contains(Path.GetExtension(path).ToLower()))
                                        .ToArray();


                timer.Stop();
                _lastTimeMilliseconds = timer.ElapsedMilliseconds;

                return imagePaths;
                
            });

            return result;
        }
    }
}
