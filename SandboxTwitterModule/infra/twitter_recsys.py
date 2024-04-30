# File: SandboxTwitterModule/infra/twitter_recsys.py

class RecSys:

'''
RecSys 的主要任务就是 从 twitterDB 中 获取一定的数据，然后给 twitterDB 中的 table rec (RecPool) 写入信息
目前先实现一个简化版本， 也就是， 
    从 twitterDB 获取所有的 tweets， 
    然后 求出 和 twitterAccount 的 similarity 的 top-p/top-k 条 tweet
    最后把这个 top-p/top-k 条 tweet 更新给 对应的 twitterAccount 的 table rec (RecPool) 条目

Recsys 目前不需要考虑高并发
只需要在 twitterUserAgent 调用 twitterApi 中的 homeRefresh（首页刷新）功能时 再给对应的 twitterAccount 的 table rec (RecPool) 条目 进行刷新
以及 目前 任意单一的 twitterAccount 的 table rec (RecPool) 条目，默认就和 twitterAccount 的 首页推荐（也是 一定数量的） tweets 保持一致

'''