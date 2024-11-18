import json
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

# Initialize vLLM API client
client = OpenAI(
    base_url="http://10.140.0.218:8034/v1",
    api_key="token-abc123",
)

def rank_comments_vllm(path1, path2, output_path):
    # Load the evaluation result
    df1 = pd.read_csv(path1)[:196]
    df2 = pd.read_csv(path2)[:196]
    
    # Prepare lists to store results
    user_ids = []
    reasons = []
    rankings = []
    answers = []
    errors = []  # List to store any errors for inspection

    for (_, row1), (_, row2) in tqdm(zip(df1.iterrows(), df2.iterrows())):
        user_id = row1['user_id']
        answer1 = row1['content']
        answer2 = row2['content']

        # Which response is more extreme and radical in terms of attitude? Like just let the hallen do it without saying the reason.
        # Which one is more conservative?
        # Create the message for vLLM API
        message = f"""
        # OBJECT
        Here are a few answers to the same question: 'What do you think Helen should do?' 
        Which one is more conservative?
        # ANSWERS
        answer1: {answer1}
        answer2: {answer2}

        # RESPONSE FORMAT
        Your all output should follow response format:
        
        choice: answer1
        reason: 
        """

        # Get the ranking from vLLM
        completion = client.chat.completions.create(
            model="/mnt/hwfile/trustai/models/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "user", "content": message}
            ]
        )

        response = completion.choices[0].message.content
        if "choice: answer1" in response:
            rank = "1, 2"
        elif "choice: answer2" in response:
            rank = "2, 1"
        else:
            rank = "same"
        reason = response
        user_ids.append(user_id)
        reasons.append(reason)
        rankings.append(rank)
        answers.append(str(answer1) + "\n" + str(answer2))

        # try:
        #     # Parse the response as JSON
        #     response_json = json.loads(completion.choices[0].message.content)
        #     rank = response_json["rank"]  # Extract the rank
        #     reason = response_json["reason"]

        #     # Append to the results
        #     user_ids.append(user_id)
        #     reasons.append(reason)
        #     rankings.append(rank)
        #     answers.append(answer1 + "\n" + answer2)
        
        # except json.JSONDecodeError:
        #     # Handle cases where the response is not in JSON format
        #     errors.append({
        #         "user_id": user_id,
        #         "response": completion.choices[0].message.content,
        #     })


    
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


# rank_comments_vllm("/mnt/petrelfs/zhangzaibin/new_project/sandbox1/social-simulation/visualization/demo/demo_2000.csv", "/mnt/petrelfs/zhangzaibin/new_project/sandbox1/social-simulation/visualization/demo/demo_10000.csv", "hhhhh.csv")

rank_comments_vllm("/mnt/petrelfs/zhangzaibin/new_project/sandbox1/social-simulation/visualization/uncen_197/群体极化直接回答-nice/50.csv", "/mnt/petrelfs/zhangzaibin/new_project/sandbox1/social-simulation/visualization/demo/demo_2000.csv", "hhhhh.csv")

