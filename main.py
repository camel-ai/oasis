<<<<<<< Updated upstream
# main.py
from SandboxTwitterModule.core.decorators import function_call_logger

from SandboxTwitterModule.sandbox_twitter import SandboxTwitter as ST

@function_call_logger
def main():
    # 实例化并启动 Twitter 沙盒
    my_twitter = ST()
    my_twitter.run()


if __name__ == '__main__':
    main()
=======

import asyncio
import threading
from social_agent.agents_generator import generate_agents
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter


def main():
    twitter_channel = Twitter_Channel()
    twitter = Twitter(db_path='/db', channel=twitter_channel)
    twitter_thread = threading.Thread(target=twitter.run)
    twitter_thread.start()

    print('Twitter is running...')
    agents = asyncio.run(generate_agents('user_all_id.csv', twitter_channel))
    tasks = []

    for agent in agents.values():
        tasks.append(agent.running())

    asyncio.gather(*tasks)


if __name__ == '__main__':
    main()
>>>>>>> Stashed changes
