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
import itertools
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rag import generate_user_profile

total = 60_000
model = "8b"


def weighted_random_age(ages, probabilities):
    ranges = []
    for age_range in ages:
        if '+' in age_range:
            start = int(age_range[:-1])
            end = start + 20  # Assume 50+ means 50-70
        else:
            start, end = map(int, age_range.split('-'))
        ranges.append((start, end))

    total_weight = sum(probabilities)
    rnd = random.uniform(0, total_weight)
    cumulative_weight = 0
    for i, weight in enumerate(probabilities):
        cumulative_weight += weight
        if rnd < cumulative_weight:
            start, end = ranges[i]
            return random.randint(start, end)
    return None


def gen_topics():
    elements = list(range(8))
    combinations = list(itertools.combinations(elements, 2))
    expanded_combinations = []
    while len(expanded_combinations) < total:
        expanded_combinations.extend(combinations)
    # Take the first 10,000
    expanded_combinations = expanded_combinations[:total]
    # Step 3: Shuffle the order
    random.shuffle(expanded_combinations)
    return expanded_combinations


# Exampls
ages = ["13-17", "18-24", "25-34", "35-49", "50+"]
probabilities = [0.066, 0.171, 0.385, 0.207, 0.171]

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

topics = [
    "Politics", "Urban Legends", "Business", "Terrorism & War",
    "Science & Technology", "Entertainment", "Natural Disasters", "Health",
    "Education"
]

mbtis = [
    "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP",
    "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

genders = ["male", "female", "other"]

p_mbti = [
    0.12625, 0.11625, 0.02125, 0.03125, 0.05125, 0.07125, 0.04625, 0.04125,
    0.04625, 0.06625, 0.07125, 0.03625, 0.10125, 0.11125, 0.03125, 0.03125
]
p_ages = [0.066, 0.171, 0.385, 0.207, 0.171]
p_genders = [0.4, 0.4, 0.2]
p_professions = [1 / 16] * 16

mbti_index = np.random.choice(len(p_mbti), size=total, p=p_mbti)
age_index = np.random.choice(len(p_ages), size=total, p=p_ages)
gender_index = np.random.choice(len(p_genders), size=total, p=p_genders)
profession_index = np.random.choice(len(p_professions),
                                    size=total,
                                    p=p_professions)
topic_index = gen_topics()


def create_user_profile(i):
    age = weighted_random_age(ages, probabilities)
    print(f"Person {i + 1}: Age={age}, MBTI={mbtis[mbti_index[i]]}, Gender="
          f"{genders[gender_index[i]]}, "
          f"Profession={professions[profession_index[i]]}")
    try:
        return generate_user_profile(age, mbtis[mbti_index[i]],
                                     genders[gender_index[i]],
                                     professions[profession_index[i]],
                                     [topics[x] for x in topic_index[i]])
    except Exception as e:
        print(e)
        retry = 5
        while retry > 0:
            try:
                return generate_user_profile(
                    age, mbtis[mbti_index[i]], genders[gender_index[i]],
                    professions[profession_index[i]],
                    [topics[x] for x in topic_index[i]])
            except Exception as e:
                print(f"{retry} times", e)
                retry -= 1
        return None


user_dict = []
start_time = time.time()

with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(create_user_profile, i) for i in range(total)]
    for future in as_completed(futures):
        user = future.result()
        if user:
            user_dict.append(user)
        if len(user_dict) % 5000 == 0:
            print(f"finish {len(user_dict)}")
            with open(f"./large_3/{model}_{len(user_dict)}_agents.json",
                      "w") as f:
                json.dump(user_dict, f, indent=4)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")
print(f"Total users generated: {len(user_dict)}")

with open(f'./large_3/{model}_{total}_agents.json', 'w') as f:
    json.dump(user_dict, f, indent=4)
