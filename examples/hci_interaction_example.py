# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import logging
import signal
import sys
import traceback

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from oasis import ActionType, generate_reddit_agent_graph
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 输入超时处理
class TimeoutExpired(Exception):
    pass


def input_with_timeout(prompt, timeout=60):

    def handler(signum, frame):
        raise TimeoutExpired()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        value = input(prompt)
        signal.alarm(0)
        return value
    except TimeoutExpired:
        print(f"\n[Error] {timeout}秒内未输入，程序自动退出。")
        sys.exit(1)
    except Exception as e:
        signal.alarm(0)
        raise e


async def main():
    # try:
    print("[INFO] 设置 OpenAI API Key...")

    print("[INFO] 初始化模型...")
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    print("[INFO] 初始化通道和平台...")
    channel = Channel()
    platform = Platform(
        db_path="./reddit_simulation.db",  # 使用现有数据库
        channel=channel,
    )
    platform_task = asyncio.create_task(platform.running())

    # 定义可用的动作
    available_actions = [
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
        ActionType.SEARCH_POSTS,
        ActionType.SEARCH_USER,
        ActionType.TREND,
        ActionType.REFRESH,
        ActionType.DO_NOTHING,
        ActionType.FOLLOW,
        ActionType.MUTE,
    ]

    print("[INFO] 生成智能体图谱...")
    try:
        agent_graph = await generate_reddit_agent_graph(
            profile_path="./data/reddit/user_data_36.json",
            model=openai_model,
            available_actions=available_actions,
            channel=channel,
        )
    except Exception as e:
        logger.error(f"[ERROR] 生成智能体图谱失败: {e}")
        logger.error(traceback.format_exc())
        return

    print("[INFO] 获取智能体实例...")
    try:
        agent = agent_graph.get_agent(0)
    except Exception as e:
        logger.error(f"[ERROR] 获取智能体失败: {e}")
        logger.error(traceback.format_exc())
        return

    print("[INFO] 开始人机交互...")

    # try:
    result = await agent.perform_action_by_hci()
    if result is None:
        logger.error("[ERROR] 操作执行失败，返回值为 None")
        return
    print("[INFO] 操作已完成，结果如下：")
    print(result)
    # except Exception as e:
    #     logger.error(f"[ERROR] 智能体交互过程中发生异常: {e}")
    #     logger.error(traceback.format_exc())

    # except Exception as e:
    #     logger.error(f"[FATAL] 程序运行异常: {e}")
    #     logger.error(traceback.format_exc())
    # finally:
    #     # 发送退出消息给平台以停止其运行
    await channel.write_to_receive_queue((0, None, ActionType.EXIT.value))
    await platform_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"[FATAL] 程序运行异常: {e}")
        logger.error(traceback.format_exc())
