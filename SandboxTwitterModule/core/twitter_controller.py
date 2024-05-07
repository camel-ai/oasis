# File: SandboxTwitterModule/core/twitter_controller.py

from enum import Enum

class Signal(Enum):
    END = 0
    PRE_START = 1
    COLD_START = 2
    WARM_START = 3
    AABS = 4  # All Agents Build Succeeded
    AGBS = 5  # Agents Graph Build Succeeded
    ATUDBBS = 6  # All Twitter Users DB Build Succeeded
    ATTDBBS = 7  # All Twitter Tweets DB Build Succeeded
    ATTraDBBS = 8  # All Twitter Traces DB Build Succeeded
    RecPoolDBBS = 9 # RecPoolDBBuildSucceeded
    DBsBS = 10  # four DataBases Build Succeeded
    RecSysConfigBS = 11 # RecSys Config Build Succeeded
    RecSysBS = 12 # RecSys Build Succeeded
    TApiBS = 15 # Twitter Api Build Succeeded
    InfraBS = 20 # infra build succeeded
    AgentRunning = 30 # Agent are Running
    
    FREEZE = 100

class TwitterController:
    def __init__(self):
        self.signal = Signal.END  # 初始化时信号设置为 END
        self.first_start()
        

    def first_start(self):
        print("-----[ctrl_signal:]TwitterController 初次启动提示, 更新 signal: END -> PRE_START-----")
        self.update_signal(Signal.PRE_START)  # 更新信号为 PRE_START

    def update_signal(self, new_signal):
        old_signal = self.signal
        self.signal = new_signal
        print(f"-----[ctrl_signal:]信号从 {old_signal.name} 更新为 {self.signal.name}-----")

    def set_signal(self, signal):
        self.signal = signal
        print(f"-----[ctrl_signal:]Signal set to {signal}-----")