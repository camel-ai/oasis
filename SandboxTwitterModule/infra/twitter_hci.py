# File: SandboxTwitterModule/infra/twitter_hci.py

from SandboxTwitterModule.core.decorators import function_call_logger

import argparse
def load_config():
    return {
        "api_key": "12345",
        "database_url": "http://example.com",
        "start_mode": "cold_start"
    }


class TwitterHCI:
    def __init__(self, signal_callback):
        self.args = None
        self.config = None
        self.start_mode = None
        self.signal_callback = signal_callback  # 保存回调函数
        self.get_init_info()
    
    def get_init_info(self):
        self.get_user_start_input(1)

    @function_call_logger
    def get_user_start_input(self, flag):
        if (1 != flag):
            print("-----[hci:]Invalid flag provided.")
            self.signal_callback("-----[hci:]Initialization failed: invalid flag")  # 调用回调函数
            pass

        # 模拟用户输入
        try:
            self.args = {'example_arg': 'value'}
            self.config = {'example_config': 'setting'}
            self.start_mode = 'COLD_START'  # 或 'WARM_START' 根据需要更改
            print("-----[hci:]用户输入 [args, config, start_mode] 信息 获取成功。")
            self.signal_callback("HCI initialization succeeded")  # 成功时的信号
        except Exception as e:
            print(f"-----[hci:]获取用户输入失败: {e}")
            self.signal_callback(f"-----[hci:]Initialization failed: {e}")  # 失败时的信号
            exit(1)  # 如果获取失败，退出程序


