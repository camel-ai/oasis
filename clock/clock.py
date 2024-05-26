from datetime import datetime


class clock:

    def __init__(self, K: int):
        # 时间放大系数
        self.k = K

    def time_transfer(self, now_time: datetime,
                      start_time: datetime) -> datetime:
        # 计算时间差
        time_diff = now_time - start_time
        # 根据系数K调整时间差
        adjusted_diff = self.k * time_diff
        # 将调整后的时间差加到现在的时间上
        adjusted_time = start_time + adjusted_diff
        return adjusted_time
