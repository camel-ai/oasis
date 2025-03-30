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
import os
import json

import redis as redis_client
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL")

redis = redis_client.from_url(redis_url)

# redis.publish(
#     "predict",
#     json.dumps({
#         "action": "start",
#         "predict_id": 1000,
#         "content": """我们做了一个支持上传knowledge base 的知识问答类型的eigentbot，现在想要提高付费意愿，大家有什么好的观点吗？"""
#     }),
# )

redis.publish("predict_new_1000", json.dumps("或许这样的chat bot 产品并不会是llm 产品的最终形态，我认为多模态产品或者游戏类型的大型模拟的内容产品会是更好的产品形态。"))
