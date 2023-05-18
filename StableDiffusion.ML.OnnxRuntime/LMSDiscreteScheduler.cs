using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics;
using NumSharp;
using System.Runtime.Serialization.Formatters;
using System.Diagnostics;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class LMSDiscreteScheduler : SchedulerBase
    {
        private int _numTrainTimesteps;

        private string _predictionType;

        private double[][] _lmsCoefficients;

        public override Tensor<float> Sigmas { get; set; }
        public override List<float> Timesteps { get; set; }
        public List<Tensor<float>> Derivatives;
        public override float InitNoiseSigma { get; set; }

        public LMSDiscreteScheduler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, string beta_schedule = "scaled_linear", string prediction_type = "epsilon", List<float> trained_betas = null)
        {
            _numTrainTimesteps = num_train_timesteps;
            _predictionType = prediction_type;
            Derivatives = new List<Tensor<float>>();
            Timesteps = new List<float>();

            var alphas = new List<float>();
            var betas = new List<float>();

            if (trained_betas != null)
            {
                betas = trained_betas;
            }
            else if (beta_schedule == "linear")
            {
                for(int i = 0; i < num_train_timesteps; i++)
                {
                    betas.Add((float)(beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1)));
                }
                //betas = Enumerable.Range(0, num_train_timesteps).Select(i => beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1)).ToList();
            }
            else if (beta_schedule == "scaled_linear")
            {
                var start =(float)Math.Sqrt(beta_start);
                var end =(float)Math.Sqrt(beta_end);
                var balh = np.linspace(start, end, num_train_timesteps).ToArray<float>().ToList();
                //conver blah to list of float
                for(int i=0; i < balh.Count(); i++)
                {
                    betas.Add((float)balh[i]);
                }
                //betas = np.linspace(start, end, num_train_timesteps).ToArray<float>().ToList();
                for(int i = 0; i < betas.Count(); i++)
                {
                    betas[i] = (float)(betas[i] * betas[i]);
                }
                //.Select(x => x * x).ToList();

            }
            else
            {
                throw new Exception("beta_schedule must be one of 'linear' or 'scaled_linear'");
            }

            for(int i = 0; i < betas.Count(); i++)
            {
                alphas.Add((float)(1 - betas[i]));
                //alphas = betas.Select(beta => 1 - beta).ToList();
            }
            
 
            this._alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate<float>((a, b) => (a * b))).ToList();
            // Create sigmas as a list and reverse it
            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

            // standard deviation of the initial noise distrubution
            this.InitNoiseSigma =(float)sigmas.Max();

        }

        //python line 135 of scheduling_lms_discrete.py
        public double GetLmsCoefficient(int order, int t, int currentOrder)
        {
            // Compute a linear multistep coefficient.

            double LmsDerivative(double tau)
            {
                double prod = 1.0;
                for (int k = 0; k < order; k++)
                {
                    if (currentOrder == k)
                    {
                        continue;
                    }
                    prod *= (tau - this.Sigmas[t - k]) / (this.Sigmas[t - currentOrder] - this.Sigmas[t - k]);
                }
                return prod;
            }

            double integratedCoeff = Integrate.OnClosedInterval(LmsDerivative, this.Sigmas[t], this.Sigmas[t + 1], 1e-4);

            return integratedCoeff;
        }

        private double[] ComputeLinearMultistepCoefficients(int stepIndex, int order)
        {
            order = Math.Min(stepIndex + 1, order);
            return Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder)).ToArray();
        }

        private void PrecomputeLinearMultistepCoefficients(int stepCount, int order = 4)
        {
            _lmsCoefficients = new double[stepCount][];
            for (int stepIndex = 0; stepIndex < stepCount; stepIndex++)
            {
                var result = ComputeLinearMultistepCoefficients(stepIndex, order);
                _lmsCoefficients[stepIndex] = result;
            }
        }

        public void ResetDerivatives()
        {
            this.Derivatives.Clear();
        }

        // Line 157 of scheduling_lms_discrete.py from HuggingFace diffusers
        public override float[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            this.Timesteps = timesteps.Select(x => (float)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            this.Sigmas = new DenseTensor<float>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                this.Sigmas[i] =(float)sigmas[i];
            }

            // precompute LMS coefficients
            PrecomputeLinearMultistepCoefficients(num_inference_steps);

            return this.Timesteps.ToArray();
        }

        public override DenseTensor<float> Step(
           DenseTensor<float> modelOutput,
           float timestep,
           DenseTensor<float> sample,
           int order = 4)
        {
            throw new NotImplementedException();
        }

        public override DenseTensor<float> Step(
               DenseTensor<Float16> modelOutput,
               float timestep,
               DenseTensor<float> sample,
               int order = 4)
        {
            var sws = new Stopwatch[22].Select(h => new Stopwatch()).ToArray();

            ReadOnlySpan<Float16> modelOutputSpan = modelOutput.Buffer.Span;
            ReadOnlySpan<float> sampleSpan = sample.Buffer.Span;

            // modelOutput has noisePred/noisePredText concatenated along first dim
            var dims = modelOutput.Dimensions.ToArray();
            dims[0] /= 2;
            var noisePredLength = modelOutput.Length / 2;

            int stepIndex = this.Timesteps.IndexOf(timestep);
            var sigma = this.Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<float> predOriginalSample;

            // Create array of type float length modelOutput.length
            float[] predOriginalSampleArray = new float[noisePredLength];

            if (this._predictionType == "epsilon")
            {

                for (int i=0; i < noisePredLength; i++)
                {
                    predOriginalSampleArray[i] = (float)(sampleSpan[i] - sigma * (float)BitConverter.UInt16BitsToHalf(modelOutputSpan[i]));
                }

                predOriginalSample = TensorHelper.CreateTensor(predOriginalSampleArray, dims);

            }
            else if (this._predictionType == "v_prediction")
            {
                //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
                throw new Exception($"prediction_type given as {this._predictionType} not implemented yet.");
            }
            else
            {
                throw new Exception($"prediction_type given as {this._predictionType} must be one of `epsilon`, or `v_prediction`");
            }

            // 2. Convert to an ODE derivative
            var derivativeItems = new DenseTensor<float>(sample.Dimensions.ToArray());

            var derivativeItemsArray = new float[derivativeItems.Length];

            for (int i = 0; i < noisePredLength; i++)
            {
                //predOriginalSample = (sample - predOriginalSample) / sigma;
                derivativeItemsArray[i] = (float)((sampleSpan[i] - predOriginalSampleArray[i]) / sigma);
            };
            derivativeItems =  TensorHelper.CreateTensor(derivativeItemsArray, derivativeItems.Dimensions.ToArray());

            this.Derivatives?.Add(derivativeItems);

            if (this.Derivatives?.Count() > order)
            {
                // remove first element
                this.Derivatives?.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            var lmsCoeffs = _lmsCoefficients[stepIndex];

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = Enumerable.Reverse(this.Derivatives).ToList();

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs.Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative));

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new Tensor<float>[this.Derivatives.Count()];

            for(int m = 0; m < lmsCoeffsAndDerivatives.Count(); m++)
            {
                var item = lmsCoeffsAndDerivatives.ElementAt(m);
                // Multiply to coeff by each derivatives to create the new tensors
                lmsDerProduct[m] = TensorHelper.MultipleTensorByFloat(item.derivative.ToArray(),(float)item.lmsCoeff, item.derivative.Dimensions.ToArray());
            }
            // Sum the tensors
            var sumTensor = TensorHelper.SumTensors(lmsDerProduct, new[] { 1, 4, 64, 64 });

            // Add the sumed tensor to the sample
            var prevSample = TensorHelper.AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());

            Console.WriteLine(prevSample[0]);
            return prevSample;

        }
    }
}
