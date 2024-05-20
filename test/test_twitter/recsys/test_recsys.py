'''测试recsys'''
from twitter.recsys import rec_sys_random  # 确保替换为包含 rec_sys_random 函数的模块名


def test_rec_sys_random_all_tweets():
    # 测试当推文数量小于等于最大推荐长度时的情况
    user_table = [{'user_id': 1}, {'user_id': 2}]
    tweet_table = [{'tweet_id': '1'}, {'tweet_id': '2'}]
    trace_table = []
    rec_matrix = [[None], [], []]
    max_rec_tweet_len = 2  # 最大推荐长度设置为2

    expected = [None, ['1', '2'], ['1', '2']]
    result = rec_sys_random(
        user_table, tweet_table, trace_table, rec_matrix, max_rec_tweet_len)
    assert result == expected


def test_rec_sys_random_sample_tweets():
    # 测试当推文数量大于最大推荐长度时的情况
    user_table = [{'user_id': 1}, {'user_id': 2}]
    tweet_table = [{'tweet_id': '1'}, {'tweet_id': '2'}, {'tweet_id': '3'}]
    trace_table = []  # 在这个测试中未使用，但是为了完整性加入
    rec_matrix = [[None], [], []]  # 假设有两个用户
    max_rec_tweet_len = 2  # 最大推荐长度设置为2

    result = rec_sys_random(
        user_table, tweet_table, trace_table, rec_matrix, max_rec_tweet_len)
    print(result)
    # 验证第一个元素是None
    assert result[0] is None
    # 验证每个用户获得了2个推文ID
    for rec in result[1:]:
        assert len(rec) == max_rec_tweet_len
        # 验证推荐的推文ID确实存在于原始推文ID列表中
        for tweet_id in rec:
            assert tweet_id in ['1', '2', '3']
