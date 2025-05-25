import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import random

from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(api_key='sk-**', base_url='')
model_type = ''

def create_user_profile(config_params, config_prompts):
    while True:
        try:
            agent_profile = {}
            for key in config_params:
                match key:
                    case "age":
                        agent_profile[key] = customize_random_age(config_params[key]["groups"], config_params[key]["ratios"])
                    case "country":
                        agent_profile[key] = customize_random_country(config_params[key]["groups"], config_params[key]["ratios"])
                    case "interested topics":
                        pass
                    case _:  # Default case for unmatched keys, including default gender, mbti, profession
                        agent_profile[key] = customize_random_traits(config_params[key]["groups"], config_params[key]["ratios"])

            topics = customize_randow_topics(config_params[key]["names"], config_params[key]["descs"], agent_profile) # create user interest topics
            agent_profile['interested topics'] = topics

            profile = generate_user_profile(agent_profile, config_prompts)  # create user profile

            return { **profile, **agent_profile }
        except Exception as e:
            print(f"Profile generation failed: {e}. Retrying...")


async def generate_user_data(config):
    config_count=config["count"]
    config_params=config["params"]
    config_prompts=config["prompts"]

    loop = asyncio.get_event_loop()

    user_data = []
    start_time = datetime.now()
    max_workers = 100  # Adjust according to your system capability
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [loop.run_in_executor(executor, create_user_profile, config_params, config_prompts) for _ in range(config_count)]
        results = await asyncio.gather(*futures)

        for i, profile in enumerate(results):
            user_data.append(profile)
            elapsed_time = datetime.now() - start_time
            print(f"Generated {i + 1}/{config_count} user profiles. Time elapsed: "
                  f"{elapsed_time}")

    return user_data

def customize_random_country(groups, ratios):
    country = random.choices(groups, ratios)[0]
    if country == "Other":
        response = client.chat.completions.create(
            model=model_type,
            messages=[{
                "role": "system",
                "content": "Select a real country name randomly, only country name is needed"  # GPT might be right, Qwen need this complement
            }])
        return response.choices[0].message.content.strip()
    return country

def customize_random_traits(groups, ratios):
    return random.choices(groups, ratios)[0]

def customize_random_age(groups, ratios):
    group = random.choices(groups, ratios)[0]
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

def customize_randow_topics(names, descs, traits):
    topic_index_lst = customize_interested_topics(names, descs, traits)
    return index_to_topics(topic_index_lst, names)

def index_to_topics(index_lst, names):
    topic_dict = {str(index): value for index, value in enumerate(names)}
    result = []
    for index in index_lst:
        topic = topic_dict[str(index)]
        result.append(topic)
    return result

def customize_interested_topics(names, descs, traits):
    prompt = f"""Based on the provided personality traits, age, gender and profession, please select 2-3 topics of interest from the given list.
    Input:\n"""
    for key in traits:
        prompt += f"    {'Personality Traits' if key == 'mbti' else key}: {traits[key]}\n"

    prompt += f"Available Topics:\n"
    for index, name in enumerate(names):
        prompt += f"    {index + 1}. {name}: {descs[index]}\n"

    prompt += f"""Output:
                  [list of topic numbers]
                  Ensure your output could be parsed to **list**, don't output anything else."""

    response = client.chat.completions.create(model=model_type,
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    topics = response.choices[0].message.content.strip()
    return json.loads(topics)

def generate_user_profile(traits, prompts):
    prompt = f"""Please generate a social media user profile based on the provided personal information, including a real name, username, user bio, and a new user persona. The focus should be on creating a fictional background story and detailed interests based on their hobbies and profession.
    Input:\n"""
    for key in traits:
        prompt += f"    {key}: {traits[key]}\n"
    
    prompt += prompts
    prompt += f"""Ensure the output can be directly parsed to **JSON**, do not output anything else."""  # noqa: E501

    response = client.chat.completions.create(model=model_type,
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    profile = response.choices[0].message.content.strip()
    return json.loads(profile)
