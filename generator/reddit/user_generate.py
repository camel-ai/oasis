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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(api_key='sk-xxx')

# Gender ratio
gender_ratio = [0.351, 0.636]
genders = ['female', 'male']

# Age ratio
age_ratio = [0.44, 0.31, 0.11, 0.03, 0.11]
age_groups = ['18-29', '30-49', '50-64', '65-100', 'underage']

# MBTI ratio
p_mbti = [
    0.12625, 0.11625, 0.02125, 0.03125, 0.05125, 0.07125, 0.04625, 0.04125,
    0.04625, 0.06625, 0.07125, 0.03625, 0.10125, 0.11125, 0.03125, 0.03125
]
mbti_types = [
    "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP",
    "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

# Country ratio
country_ratio = [0.4833, 0.0733, 0.0697, 0.0416, 0.0306, 0.3016]
countries = ["US", "UK", "Canada", "Australia", "Germany", "Other"]

# Profession ratio
p_professions = [1 / 16] * 16
professions = [
    "Agriculture, Food & Natural Resources", "Architecture & Construction",
    "Arts, Audio/Video Technology & Communications",
    "Business Management & Administration", "Education & Training", "Finance",
    "Government & Public Administration", "Health Science",
    "Hospitality & Tourism", "Human Services", "Information Technology",
    "Law, Public Safety, Corrections & Security", "Manufacturing", "Marketing",
    "Science, Technology, Engineering & Mathematics",
    "Transportation, Distribution & Logistics"
]


def get_random_gender():
    return random.choices(genders, gender_ratio)[0]


def get_random_age():
    group = random.choices(age_groups, age_ratio)[0]
    if group == 'underage':
        return random.randint(10, 17)
    elif group == '18-29':
        return random.randint(18, 29)
    elif group == '30-49':
        return random.randint(30, 49)
    elif group == '50-64':
        return random.randint(50, 64)
    else:
        return random.randint(65, 100)


def get_random_mbti():
    return random.choices(mbti_types, p_mbti)[0]


def get_random_country():
    country = random.choices(countries, country_ratio)[0]
    if country == "Other":
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Select a real country name randomly:"
            }])
        return response.choices[0].message.content.strip()
    return country


def get_random_profession():
    return random.choices(professions, p_professions)[0]


def get_interested_topics(mbti, age, gender, country, profession):
    prompt = f"""Based on the provided personality traits, age, gender and profession, please select 2-3 topics of interest from the given list.
    Input:
        Personality Traits: {mbti}
        Age: {age}
        Gender: {gender}
        Country: {country}
        Profession: {profession}
    Available Topics:
        1. Economics: The study and management of production, distribution, and consumption of goods and services. Economics focuses on how individuals, businesses, governments, and nations make choices about allocating resources to satisfy their wants and needs, and tries to determine how these groups should organize and coordinate efforts to achieve maximum output.
        2. IT (Information Technology): The use of computers, networking, and other physical devices, infrastructure, and processes to create, process, store, secure, and exchange all forms of electronic data. IT is commonly used within the context of business operations as opposed to personal or entertainment technologies.
        3. Culture & Society: The way of life for an entire society, including codes of manners, dress, language, religion, rituals, norms of behavior, and systems of belief. This topic explores how cultural expressions and societal structures influence human behavior, relationships, and social norms.
        4. General News: A broad category that includes current events, happenings, and trends across a wide range of areas such as politics, business, science, technology, and entertainment. General news provides a comprehensive overview of the latest developments affecting the world at large.
        5. Politics: The activities associated with the governance of a country or other area, especially the debate or conflict among individuals or parties having or hoping to achieve power. Politics is often a battle over control of resources, policy decisions, and the direction of societal norms.
        6. Business: The practice of making one's living through commerce, trade, or services. This topic encompasses the entrepreneurial, managerial, and administrative processes involved in starting, managing, and growing a business entity.
        7. Fun: Activities or ideas that are light-hearted or amusing. This topic covers a wide range of entertainment choices and leisure activities that bring joy, laughter, and enjoyment to individuals and groups.
    Output:
    [list of topic numbers]
    Ensure your output could be parsed to **list**, don't output anything else."""  # noqa: E501

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    topics = response.choices[0].message.content.strip()
    return json.loads(topics)


def generate_user_profile(age, gender, mbti, profession, topics):
    prompt = f"""Please generate a social media user profile based on the provided personal information, including a real name, username, user bio, and a new user persona. The focus should be on creating a fictional background story and detailed interests based on their hobbies and profession.
    Input:
        age: {age}
        gender: {gender}
        mbti: {mbti}
        profession: {profession}
        interested topics: {topics}
    Output:
    {{
        "realname": "str",
        "username": "str",
        "bio": "str",
        "persona": "str"
    }}
    Ensure the output can be directly parsed to **JSON**, do not output anything else."""  # noqa: E501

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    profile = response.choices[0].message.content.strip()
    return json.loads(profile)


def index_to_topics(index_lst):
    topic_dict = {
        '1': 'Economics',
        '2': 'Information Technology',
        '3': 'Culture & Society',
        '4': 'General News',
        '5': 'Politics',
        '6': 'Business',
        '7': 'Fun'
    }
    result = []
    for index in index_lst:
        topic = topic_dict[str(index)]
        result.append(topic)
    return result


def create_user_profile():
    while True:
        try:
            gender = get_random_gender()
            age = get_random_age()
            mbti = get_random_mbti()
            country = get_random_country()
            profession = get_random_profession()
            topic_index_lst = get_interested_topics(mbti, age, gender, country,
                                                    profession)
            topics = index_to_topics(topic_index_lst)
            profile = generate_user_profile(age, gender, mbti, profession,
                                            topics)
            profile['age'] = age
            profile['gender'] = gender
            profile['mbti'] = mbti
            profile['country'] = country
            profile['profession'] = profession
            profile['interested_topics'] = topics
            return profile
        except Exception as e:
            print(f"Profile generation failed: {e}. Retrying...")


def generate_user_data(n):
    user_data = []
    start_time = datetime.now()
    max_workers = 100  # Adjust according to your system capability
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_user_profile) for _ in range(n)]
        for i, future in enumerate(as_completed(futures)):
            profile = future.result()
            user_data.append(profile)
            elapsed_time = datetime.now() - start_time
            print(f"Generated {i+1}/{n} user profiles. Time elapsed: "
                  f"{elapsed_time}")
    return user_data


def save_user_data(user_data, filename):
    with open(filename, 'w') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    N = 10000  # Target user number
    user_data = generate_user_data(N)
    output_path = 'experiment_dataset/user_data/user_data_10000.json'
    save_user_data(user_data, output_path)
    print(f"Generated {N} user profiles and saved to {output_path}")
