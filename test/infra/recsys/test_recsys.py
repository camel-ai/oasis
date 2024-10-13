from social_simulation.social_platform.recsys import (
    rec_sys_personalized, rec_sys_personalized_with_trace, rec_sys_random,
    rec_sys_reddit, rec_sys_personalized_twh, reset_globals)


def test_rec_sys_random_all_posts():
    # 测试当推文数量小于等于最大推荐长度时的情况
    user_table = [{'user_id': 1}, {'user_id': 2}]
    post_table = [{'post_id': '1'}, {'post_id': '2'}]
    trace_table = []
    rec_matrix = [[None], [], []]
    max_rec_post_len = 2  # 最大推荐长度设置为2

    expected = [None, ['1', '2'], ['1', '2']]
    result = rec_sys_random(user_table, post_table, trace_table, rec_matrix,
                            max_rec_post_len)
    assert result == expected


def test_rec_sys_reddit_all_posts():
    # 测试当推文数量小于等于最大推荐长度时的情况
    post_table = [{'post_id': '1'}, {'post_id': '2'}]
    rec_matrix = [[None], [], []]
    max_rec_post_len = 2  # 最大推荐长度设置为2

    expected = [None, ['1', '2'], ['1', '2']]
    result = rec_sys_reddit(post_table, rec_matrix, max_rec_post_len)
    assert result == expected


def test_rec_sys_personalized_all_posts():
    # 测试当推文数量小于等于最大推荐长度时的情况
    user_table = [{
        'user_id': 1,
        'bio': 'I like cats'
    }, {
        'user_id': 2,
        'bio': 'I like dogs'
    }]
    post_table = [{
        'post_id': '1',
        'user_id': 3,
        'content': 'I like dogs'
    }, {
        'post_id': '2',
        'user_id': 4,
        'content': 'I like cats'
    }]
    trace_table = []
    rec_matrix = [[None], [], []]
    max_rec_post_len = 2  # 最大推荐长度设置为2

    expected = [None, ['1', '2'], ['1', '2']]
    result = rec_sys_personalized(user_table, post_table, trace_table,
                                  rec_matrix, max_rec_post_len)
    assert result == expected

def test_rec_sys_personalized_twhin():
    # 测试当推文数量小于等于最大推荐长度时的情况
    user_table = [{
        'user_id': 1,
        'bio': 'I like cats',
        'num_followers': 3
    }, {
        'user_id': 2,
        'bio': 'I like dogs',
        'num_followers': 5
    }, {
        'user_id': 3,
        'bio': '',
        'num_followers': 5
    }, {
        'user_id': 4,
        'bio': '',
        'num_followers': 5
    }]
    post_table = [{
        'post_id': '1',
        'user_id': 3,
        'content': 'I like dogs',
        "created_at": "0"
    }, {
        'post_id': '2',
        'user_id': 4,
        'content': 'I like cats',
        "created_at": "0"
    }]
    trace_table = []
    rec_matrix = [[None], [], [], [], []]
    max_rec_post_len = 2  # 最大推荐长度设置为2
    latest_post_count = len(post_table)
    expected = [None, ['1', '2'], ['1', '2'], ['1', '2'], ['1', '2']]

    reset_globals()
    result = rec_sys_personalized_twh(user_table,post_table,latest_post_count, trace_table,
                                  rec_matrix, max_rec_post_len)
    assert result == expected


def test_rec_sys_personalized_with_trace_all_posts():
    # 测试当推文数量小于等于最大推荐长度时的情况
    user_table = [{
        'user_id': 1,
        'bio': 'I like cats'
    }, {
        'user_id': 2,
        'bio': 'I like dogs'
    }]
    post_table = [{
        'post_id': '1',
        'user_id': 3,
        'content': 'I like dogs'
    }, {
        'post_id': '2',
        'user_id': 4,
        'content': 'I like cats'
    }]
    trace_table = []
    rec_matrix = [[None], [], []]
    max_rec_post_len = 2  # 最大推荐长度设置为2

    expected = [None, ['1', '2'], ['1', '2']]
    result = rec_sys_personalized_with_trace(user_table, post_table,
                                             trace_table, rec_matrix,
                                             max_rec_post_len)
    assert result == expected


def test_rec_sys_random_sample_posts():
    # 测试当推文数量大于最大推荐长度时的情况
    user_table = [{'user_id': 1}, {'user_id': 2}]
    post_table = [{'post_id': '1'}, {'post_id': '2'}, {'post_id': '3'}]
    trace_table = []  # 在这个测试中未使用，但是为了完整性加入
    rec_matrix = [[None], [], []]  # 假设有两个用户
    max_rec_post_len = 2  # 最大推荐长度设置为2

    result = rec_sys_random(user_table, post_table, trace_table, rec_matrix,
                            max_rec_post_len)
    print(result)
    # 验证第一个元素是None
    assert result[0] is None
    # 验证每个用户获得了2个推文ID
    for rec in result[1:]:
        assert len(rec) == max_rec_post_len
        # 验证推荐的推文ID确实存在于原始推文ID列表中
        for post_id in rec:
            assert post_id in ['1', '2', '3']


def test_rec_sys_reddit_sample_posts():
    # 测试当推文数量大于最大推荐长度时的情况
    post_table = [{
        'post_id': '1',
        'num_likes': 100000,
        'num_dislikes': 25,
        'created_at': "2024-06-25 12:00:00.222000"
    }, {
        'post_id': '2',
        'num_likes': 90,
        'num_dislikes': 30,
        'created_at': "2024-06-26 12:00:00.321009"
    }, {
        'post_id': '3',
        'num_likes': 75,
        'num_dislikes': 50,
        'created_at': "2024-06-27 12:00:00.123009"
    }, {
        'post_id': '4',
        'num_likes': 70,
        'num_dislikes': 50,
        'created_at': "2024-06-27 13:00:00.321009"
    }]
    rec_matrix = [[None], [], []]  # 假设有两个用户
    max_rec_post_len = 3  # 最大推荐长度设置为2

    result = rec_sys_reddit(post_table, rec_matrix, max_rec_post_len)
    print(result)
    # 验证每个用户获得了2个推文ID
    for rec in result[1:]:
        assert len(rec) == max_rec_post_len
        # 验证推荐的推文ID确实存在于原始推文ID列表中
        for post_id in rec:
            assert post_id in ['3', '4', '1']


def test_rec_sys_personalized_sample_posts():
    # 测试当推文数量大于最大推荐长度时的情况
    user_table = [{
        'user_id': 1,
        'bio': 'I like cats'
    }, {
        'user_id': 2,
        'bio': 'I like dogs'
    }]
    post_table = [{
        'post_id': '1',
        'user_id': 3,
        'content': 'I like dogs'
    }, {
        'post_id': '2',
        'user_id': 4,
        'content': 'I like cats'
    }, {
        'post_id': '3',
        'user_id': 5,
        'content': 'I like birds'
    }]
    trace_table = []  # 在这个测试中未使用，但是为了完整性加入
    rec_matrix = [[None], [], []]  # 假设有两个用户
    max_rec_post_len = 2  # 最大推荐长度设置为2

    result = rec_sys_personalized(user_table, post_table, trace_table,
                                  rec_matrix, max_rec_post_len)
    print(result)
    # 验证第一个元素是None
    assert result[0] is None
    # 验证每个用户获得了2个推文ID
    for rec in result[1:]:
        assert len(rec) == max_rec_post_len
        # 验证推荐的推文ID确实存在于原始推文ID列表中
        for post_id in rec:
            assert post_id in ['1', '2', '3']

    # personalized 推荐应该是根据用户的bio进行推荐
    for i in range(1, len(result)):
        if i == 1:
            assert result[i] == ['2', '1']

        if i == 2:
            assert result[i] == ['1', '2']

def test_rec_sys_personalized_twhin_sample_posts():
    # 测试当推文数量大于最大推荐长度时的情况
    user_table = [{
        'user_id': 1,
        'bio': 'I like cats',
        'num_followers': 3
    }, {
        'user_id': 2,
        'bio': 'I like dogs',
        'num_followers': 3
    }, {
        'user_id': 3,
        'bio': '',
        'num_followers': 3
    }, {
        'user_id': 4,
        'bio': '',
        'num_followers': 3
    }, {
        'user_id': 5,
        'bio': '',
        'num_followers': 3
    }]
    post_table = [{
        'post_id': '1',
        'user_id': 3,
        'content': 'I like dogs',
        "created_at": "0"
    }, {
        'post_id': '2',
        'user_id': 4,
        'content': 'I like cats',
        "created_at": "0"
    }, {
        'post_id': '3',
        'user_id': 5,
        'content': 'I like birds',
        "created_at": "0"
    }]
    trace_table = []  # 在这个测试中未使用，但是为了完整性加入
    rec_matrix = [[None], [], [],[], [],[]]  # 假设有5个用户
    max_rec_post_len = 2  # 最大推荐长度设置为2
    latest_post_count = len(post_table)
    reset_globals()    
    result = rec_sys_personalized_twh(user_table, post_table, latest_post_count, trace_table,
                                  rec_matrix, max_rec_post_len)
    print(result)
    # 验证第一个元素是None
    assert result[0] is None
    # 验证每个用户获得了2个推文ID
    for rec in result[1:]:
        assert len(rec) == max_rec_post_len
        # 验证推荐的推文ID确实存在于原始推文ID列表中
        for post_id in rec:
            assert post_id in ['1', '2', '3']

    # personalized 推荐应该是根据用户的bio进行推荐
    for i in range(1, len(result)):
        if i == 1:
            assert result[i] == ['2', '1']

        if i == 2:
            assert result[i] == ['1', '2']

def test_rec_sys_personalized_with_trace_sample_posts():
    # 测试当推文数量大于最大推荐长度时的情况
    user_table = [{
        'user_id': 1,
        'bio': 'I like cats'
    }, {
        'user_id': 2,
        'bio': 'I like dogs'
    }]
    post_table = [{
        'post_id': '1',
        'user_id': 3,
        'content': 'I like dogs'
    }, {
        'post_id': '2',
        'user_id': 4,
        'content': 'I like cats'
    }, {
        'post_id': '3',
        'user_id': 5,
        'content': 'I like birds'
    }]
    trace_table = [{
        'user_id': 1,
        'post_id': '3',
        'action': 'like'
    }, {
        'user_id': 2,
        'post_id': '2',
        'action': 'like'
    }]
    rec_matrix = [[None], [], []]  # 假设有两个用户
    max_rec_post_len = 2  # 最大推荐长度设置为2

    result = rec_sys_personalized_with_trace(user_table, post_table,
                                             trace_table, rec_matrix,
                                             max_rec_post_len)

    print(result)
    # 验证第一个元素是None
    assert result[0] is None

    # 验证每个用户获得了2个推文ID
    for rec in result[1:]:
        assert len(rec) == max_rec_post_len
        # 验证推荐的推文ID确实存在于原始推文ID列表中
        for post_id in rec:
            assert post_id in ['1', '2', '3']

    # personalized 推荐应该是根据用户的bio进行推荐
