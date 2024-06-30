from datetime import datetime

from clock.clock import Clock


def test_time_transfer():
    sandbox_clock = Clock(k=60)
    start_date_string = '2024-05-26 22:31:42.098815'

    now_date_string_1 = '2024-05-26 22:31:43.098815'
    now_date_string_2 = '2024-05-26 22:33:42.098815'

    date_format = '%Y-%m-%d %H:%M:%S.%f'
    start_time = datetime.strptime(start_date_string, date_format)

    now_time_1 = datetime.strptime(now_date_string_1, date_format)
    now_time_2 = datetime.strptime(now_date_string_2, date_format)

    adjust_time_1 = sandbox_clock.time_transfer(now_time_1, start_time)
    adjust_time_2 = sandbox_clock.time_transfer(now_time_2, start_time)

    # K = 60时，增加1秒换算成增加1分钟
    expected_date_1 = datetime.strptime('2024-05-26 22:32:42.098815',
                                        date_format)
    # K = 60时，增加2分钟换算成增加2小时
    expected_date_2 = datetime.strptime('2024-05-27 0:31:42.098815',
                                        date_format)

    assert adjust_time_1 == expected_date_1
    assert adjust_time_2 == expected_date_2
