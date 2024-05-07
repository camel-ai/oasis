import os
import inspect

def function_call_logger(func):
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        caller_frame = inspect.stack()[1]
        caller_filename = caller_frame.filename
        current_dir = os.getcwd()
        if caller_filename.startswith(current_dir):
            relative_path = os.path.relpath(caller_filename, current_dir)
            print(f"=====[func_call_logger:]源文件 [{relative_path}] 中的函数 [{function_name}] 即将被调用=====")
        else:
            basename = os.path.basename(caller_filename)
            print(f"=====[func_call_logger:]源文件 [{basename}] 中的函数 [{function_name}] 即将被调用=====")
        result = func(*args, **kwargs)
        if caller_filename.startswith(current_dir):
            relative_path = os.path.relpath(caller_filename, current_dir)
            print(f"=====[func_call_logger:]源文件 [{relative_path}] 中的函数 [{function_name}] 调用结束=====")
        else:
            basename = os.path.basename(caller_filename)
            print(f"=====[func_call_logger:]源文件 [{basename}] 中的函数 [{function_name}] 调用结束=====")
        return result
    return wrapper



