# File: SandboxTwitterModule/core/decorators.py
import os
import inspect

def function_call_logger(func):
    """
    A decorator that logs the call and return of the function, 
    showing the relative or absolute path of the source file where the function is called.
    
    Args:
        func (Callable): The function to be decorated.
        
    Returns:
        Callable: The wrapper function that adds logging to the original function.
    """
    def log_message(phase, filename):
        """ Helper function to print log messages. """
        if filename.startswith(os.getcwd()):
            path = os.path.relpath(filename, os.getcwd())
        else:
            path = os.path.basename(filename)
        print(f"=====[func_call_logger:]源文件 [{path}] 中的函数 [{func.__name__}] {phase}=====")

    def wrapper(*args, **kwargs):
        caller_filename = inspect.stack()[1].filename
        log_message("即将被调用", caller_filename)
        result = func(*args, **kwargs)
        log_message("调用结束", caller_filename)
        return result
    return wrapper
