using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace StableDiffusionGeneration.Common
{
    public class CommandBase<T> : ICommand where T : class
    {
        private readonly Action<T?> _execute;
        private readonly Predicate<T?> _canExecute;

        public CommandBase(Action<T?> execute, Predicate<T?> canExecute = null)
        {
            _execute = execute ?? throw new ArgumentNullException(nameof(execute));
            _canExecute = canExecute;
        }

        public event EventHandler? CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }

        public bool CanExecute(object? parameter)
        {
            return _canExecute?.Invoke(parameter as T) ?? true;
        }

        public void Execute(object? parameter)
        {
            _execute.Invoke(parameter as T);
        }

        public void RaiseCanExecute()
        {
            CommandManager.InvalidateRequerySuggested();
        }
    }
}
