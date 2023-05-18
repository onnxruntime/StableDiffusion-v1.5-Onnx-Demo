using StableDiffusionGeneration.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StableDiffusionGeneration.ViewModel
{
    public class ViewModelLocator
    {
        public static ViewModelLocator Instance { get; } = new ViewModelLocator();

        public static GeneratorViewModel GeneratorViewModel { get; } = new GeneratorViewModel(new GeneratorModel());
    }
}
