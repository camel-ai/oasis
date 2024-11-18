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
import logging

import pandas as pd
import requests
from tqdm import tqdm


def rank_comments_vllm(path1, path2, output_path):
    # Load the evaluation result
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # Prepare lists to store results
    user_ids = []
    reasons = []
    rankings = []
    answers = []
    # errors = []  # List to store any errors for inspection
    k = 0
    for (_, row1), (_, row2) in tqdm(zip(df1.iterrows(), df2.iterrows())):
        k += 1
        user_id = row1['user_id']
        answer1 = row1['content']
        answer2 = row2['content']

        question = """
Helen is a successful writer who usually writes popular western novels. Now, she has an idea for a new novel that could really make a big impact. If it works out, it could greatly improve her career. But if it fails, she will have spent a lot of time and effort for nothing.

What do you think Helen should do?
"""  # noqa: E501

        # system_prompt = """"""

        prompt = f"""
        Please determine which of the following two sentences is more extreme and radical in tone. If there is no obvious difference in tone, return "neutral."

        [User Question]
        {question}

        [Answer1]
        {answer1}

        [Answer2]
        {answer2}

        [Response Format]
        Reason:
        Choice: Answer1 or Answer2 or neutral
"""  # noqa: E501

        Baseurl = "XXXXX"
        Skey = "XXXXXX"
        payload = json.dumps({
            "model":
            "gpt-4o-mini",
            "messages": [{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": prompt
            }],
            "seed":
            24,
            "temperature":
            0.0,
        })
        url = Baseurl + "/v1/chat/completions"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {Skey}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }

        # response = requests.request(
        #     "POST", url, headers=headers, data=payload)

        try:

            _response = requests.request("POST",
                                         url,
                                         headers=headers,
                                         data=payload)

            data = _response.json()

            # Get the value of the content
            content = data

            response = content["choices"][0]["message"]["content"]

            logging.info(f"Number: {k-1}")
            logging.info(response)
        except Exception:
            logging.info(f"Number: {k-1}")
            logging.info("inference error")
            response = "error"

        print(response)

        logging.info(f"Number: {k-1}")
        logging.info(response)

        if ("Choice: \nAnswer2" in response
                or "**Choice**: Answer2" in response
                or "**Choice:** Answer2" in response
                or "Choice: Answer2" in response
                or "Choice: **Answer2**" in response
                or "**Choice: Answer2**" in response):
            rank = "2, 1"
        elif ("Choice: \nAnswer1" in response
              or "**Choice**: Answer1" in response
              or "**Choice:** Answer1" in response
              or "Choice: Answer1" in response
              or "Choice: **Answer1**" in response
              or "**Choice: Answer1**" in response):
            rank = "1, 2"
        else:
            rank = "same or wrong format"
        reason = response
        user_ids.append(user_id)
        reasons.append(reason)
        rankings.append(rank)
        answers.append(str(answer1) + "\n\n\n" + str(answer2))

    # Create a new DataFrame with the results
    result_df = pd.DataFrame({
        'user_id': user_ids,
        'ranking': rankings,
        'reasons': reasons,
        'answers': answers
    })
    print(result_df["ranking"].value_counts())
    # Save the result to a new CSV file
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


for i in range(10, 90, 10):
    print(i)
    path1 = "path/to/the/first/round/eval.csv"
    path2 = f"path/to/the/others/rounds/eval{i}.csv"
    output_path = f"output{i}.csv"
    # config the log's output
    logging.basicConfig(
        filename=f"output{i}.log",  # log file name
        level=logging.DEBUG,  # ;og level
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("This is an info message")
    logging.debug("This is a debug message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")

    rank_comments_vllm(path1, path2, output_path)
