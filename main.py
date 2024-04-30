# main.py
from SandboxTwitterModule.core.decorators import function_call_logger

from SandboxTwitterModule.sandbox_twitter import SandboxTwitter as ST

@function_call_logger
def main():
    # 实例化并启动 Twitter 沙盒
    my_twitter = ST()
    # my_twitter.run()


if __name__ == '__main__':
    main()
