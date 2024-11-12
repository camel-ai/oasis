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
import json
import random

import pandas as pd
from tqdm import tqdm

data_path = './users'

with open(f'{data_path}.json', 'r') as json_file:
    users_1 = json.load(json_file)

users = users_1

users = users[:1_000_0]
users = random.sample(users, 800)

df = pd.read_csv('new_stars.csv')


def generate_random_number(thresh):
    return 1 if random.random() <= thresh else 0


for user in tqdm(users, desc="Processing users"):
    following = []
    for topic in user['topics']:
        # print(topic)
        topic_users = df[df['category'] == topic]
        topic_user_ids = topic_users['user_id'].tolist()
        # print(topic_user_ids)
        for user_id in topic_user_ids:
            if generate_random_number(0.2):
                following.append(user_id)
        # print(following)

    user['following_list'] = following
    user['activity_level'] = ['active'] * 24
    user['activity_level_frequency'] = [100] * 24
    user['previous_tweets'] = []
    user['tweets_id'] = 0

print("finish generating")

user_df = pd.DataFrame(users)
print(user_df.head())

user_df.rename(columns={
    "realname": "name",
    "username": "username",
    "bio": "description",
    "persona": "user_char",
    "following_list": "following_agentid_list"
},
               inplace=True)

origin_data = pd.read_csv('new_stars.csv')
origin_data["following_agentid_list"] = origin_data["following_list"]

new_data = pd.concat([origin_data, user_df])
# new_data
new_data = new_data.drop(['Unnamed: 0.1', 'Unnamed: 0', 'user_id'], axis=1)
new_data = new_data.drop(
    ['created_at', 'followers_count', 'following_count', 'following_list'],
    axis=1)
new_data['user_id'] = range(0, len(new_data))
new_data = new_data.reset_index(drop=True)


def my_function(x):
    if isinstance(x, str):  # Make sure you only encode strings.
        return x.encode('utf-8', 'replace').decode('utf-8')
    return x


new_data = new_data.applymap(my_function)
new_data.to_csv("1k_0.2.csv")
print(len(new_data))
