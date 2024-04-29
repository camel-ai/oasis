# File: recsys/code/test/test.py
import unittest
from recsys import RecSys


class TestRecSys(unittest.TestCase):
    def setUp(self):
        # 使用测试数据库初始化 RecSys 对象
        self.rec_sys = RecSys('testDB.db')

    def test_fetch_data(self):
        # 测试获取数据功能
        users, tweets = self.rec_sys.fetch_data()
        self.assertTrue(len(users) > 0, "未获取到用户数据")
        self.assertTrue(len(tweets) > 0, "未获取到推文数据")

    def test_clean_data(self):
        # 假设我们有预定义的测试推文
        tweets = [
            (1, "这是一条安全的推文。"),
            (2, "这条推文包含违法二字。")
        ]
        cleaned_tweets = self.rec_sys.clean_data(tweets)
        self.assertEqual(len(cleaned_tweets), 1, "推文清洗失败")
        self.assertNotIn("违法", cleaned_tweets[0][1], "包含违法内容的推文未被移除")

    # 可以添加更多测试覆盖其他方法，如 store_recommendations 和 calculate_similarity

if __name__ == '__main__':
    unittest.main()
