import asyncio
import threading
from social_agent.agents_generator import AgentsGenerator
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter


twitter_channel = Twitter_Channel()
twitter = Twitter(db_path='/db', channel=twitter_channel)
twitter_thread = threading.Thread(target=twitter.run)
twitter_thread.start()


print('test')

async def g():
    print('test g')

g_thread = threading.Thread(target=asyncio.run(g()))
g_thread.start()

async def f():
    print('test f')

f_thread = threading.Thread(target=asyncio.run(f()))
f_thread.start()
    