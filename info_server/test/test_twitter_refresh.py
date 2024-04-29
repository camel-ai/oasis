import asyncio
import os
import os.path as osp
import pathlib
import sys

import pytest
from communication.channel import Channel

from core.create_database import create_db
from core.twitter import Twitter

parent_folder = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_folder))

parent_folder = osp.dirname(osp.abspath(__file__))
db_filepath = osp.join(parent_folder, "mock_twitter_refresh.db")


class MockReceiverChannel(Channel):
    """
    A mock channel to receive tweets
    Assume the receiver channel has already connected to the RecSys channel
    """

    def __init__(self, name):
        super().__init__(name)
        self.input_queues["RecSys"] = asyncio.Queue()
        self.output_queues["RecSys"] = asyncio.Queue()

    async def receive_from(self, name, user_id):
        # Mock recommended tweets
        return [{
            "user_id": 1,
            "tweet_id": i,
            "content": f"Tweet {i}",
            "created_at": f"2024-04-{21 + i} 22:02:42.510232",
            "num_likes": i
        } for i in range(10)]


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    # Create a mock database:
    create_db(db_filepath)
    yield
    if osp.exists(db_filepath):
        os.remove(db_filepath)


@pytest.mark.asyncio
async def test_twitter_refresh():
    receiver_channel = MockReceiverChannel("Receiver")
    twitter = Twitter(db_path=db_filepath)
    content = await twitter.refresh(channel=receiver_channel,
                                    user_id=1,
                                    created_at='2024-05-01 20:42:01.910175')
    # assert we can get the tweets from the channel
    assert content == [{
        "user_id": 1,
        "tweet_id": i,
        "content": f"Tweet {i}",
        "created_at": f"2024-04-{21 + i} 22:02:42.510232",
        "num_likes": i
    } for i in range(10)]
    twitter.db_cursor.execute(
        "SELECT user_id, created_at, action, info FROM trace WHERE user_id = 1"
    )
    trace = twitter.db_cursor.fetchall()
    # assert the record in trace table is correct
    assert trace == [(1, '2024-05-01 20:42:01.910175', 'refresh', None)]
