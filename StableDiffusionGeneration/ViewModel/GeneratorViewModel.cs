using StableDiffusionGeneration.Common;
using StableDiffusionGeneration.Model;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Threading;

namespace StableDiffusionGeneration.ViewModel
{
    public class GeneratorViewModel : ModelBase
    {
        private bool _executingInference = false;

        public GeneratorViewModel(GeneratorModel model) 
        {
            _model = model;
            _inputDescription = "landscape, painting, rolling hills, windmill, clouds";
            _numberOfImages = 1;
            _images = new ObservableCollection<ImageSource>();
            _generateCommand = new CommandBase<object>(ExecuteGenerateCommand, CanExecuteGenerateCommand);

            _model.PropertyChanged += _model_PropertyChanged;
            StatusString = "Idle";
   
        }

        private void _model_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            switch(e.PropertyName)
            {
                case nameof(_model.Percentage):
                    Dispatcher.CurrentDispatcher.Invoke(() =>
                    {
                        ProcessPrecentageChange(_model.Percentage);
                    });
                    break;
                default:
                    break;
            }
        }

        private void ProcessPrecentageChange(double percentage)
        {
            ProgressPercentage = percentage;
        }

        private CommandBase<object> _generateCommand;

        public CommandBase<object> GenerateCommand
        {
            get => _generateCommand;
            set => SetProperty<CommandBase<object>>(ref _generateCommand, value);
        }

        private async void ExecuteGenerateCommand(object args)
        {
            _executingInference = true;
            Dispatcher.CurrentDispatcher.Invoke(() =>
            {
                StatusString = "Generating ...";
                _generateCommand.RaiseCanExecute();
            });

            var imagePaths = await _model.GenerateImages(_inputDescription, _numberOfImages)
                .ConfigureAwait(true);

            StatusString = String.Format("{0} iterations ({1:F1} it/sec); {2:F1} sec total", _model.NumInferenceSteps, _model.IterationsPerSecond, _model.LastTimeMilliseconds / 1000.0);
            _executingInference = false;
            _generateCommand.RaiseCanExecute();

            // create images and bind them.
            Dispatcher.CurrentDispatcher.Invoke(() => 
            {
                ObservableCollection<ImageSource> imageSources = new ObservableCollection<ImageSource>();
                foreach (string imagePath in imagePaths)
                {
                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.UriSource = new Uri(imagePath);
                    bitmap.EndInit();

                    imageSources.Add(bitmap);
                }
              
                Images = imageSources; 
            });
        }

        private bool CanExecuteGenerateCommand(object args)
        {
            return !_executingInference;
        }
        private int _numberOfImages;
        public int NumberOfImages
        {
            get => _numberOfImages;
            set => SetProperty(ref _numberOfImages, value);
        }

        private string _inputDescription;
        public string InputDescription
        {
            get => _inputDescription;
            set => SetProperty(ref _inputDescription, value);
        }

        private ICollection<ImageSource> _images;
        public ICollection<ImageSource> Images
        {
            get => _images;
            private set => SetProperty(ref _images, value);
        }

        private GeneratorModel? _model;
        public GeneratorModel? Model 
        { 
            get => _model;
            private set => SetProperty(ref _model, value); 
        }

        private double _progressPercentage;

        public double ProgressPercentage
        {
            get => _progressPercentage;
            set => SetProperty(ref _progressPercentage, value);
        }

        public string _statusString;
        public string StatusString
        {
            get => _statusString;
            set => SetProperty(ref _statusString, value);   
        }
        
    }
}
