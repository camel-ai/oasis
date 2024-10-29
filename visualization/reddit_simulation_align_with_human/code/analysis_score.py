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
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class Database:

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def get_score_comment_id(self, comment_id):
        cursor = self.conn.cursor()
        """
        Query the score of a specific comment by comment ID (likes minus
        dislikes).
        """
        # Prepare SQL query to calculate the score (likes - dislikes)
        query = """
        SELECT (num_likes - num_dislikes) AS score
        FROM comment
        WHERE comment_id = ?
        """
        # Execute the query
        cursor.execute(query, (comment_id, ))
        # Fetch the result
        result = cursor.fetchone()
        # If there's a result, return the score; otherwise, return None
        if result:
            return result[0]
        else:
            return None


def get_result(comment_id_lst, db_path):
    db = Database(db_path)

    result_lst = []

    for track_comment_id in comment_id_lst:
        result = db.get_score_comment_id(track_comment_id)
        if result is None:
            print(f"Comment with id:{track_comment_id} not found.")
            result_lst.append(result)
        else:
            result_lst.append(result)
    return result_lst


# Function to calculate mean and 95% confidence interval
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def visualization(up_result, down_result, control_result, exp_name,
                  folder_path):
    # Calculate the mean and confidence interval for each group
    up_mean, up_ci_low, up_ci_high = mean_confidence_interval(up_result)
    down_mean, down_ci_low, down_ci_high = mean_confidence_interval(
        down_result)
    control_mean, control_ci_low, control_ci_high = mean_confidence_interval(
        control_result)

    # Plotting
    labels = ["Down", "Control", "Up"]
    means = [down_mean, control_mean, up_mean]
    conf_intervals = [
        (down_ci_low, down_ci_high),
        (control_ci_low, control_ci_high),
        (up_ci_low, up_ci_high),
    ]

    x_pos = range(len(labels))  # x positions

    fig, ax = plt.subplots()

    # Plot the bar chart
    # Note: The calculation of yerr needs adjustment to ensure the error
    # bars correspond to the respective means
    ax.bar(
        labels,
        means,
        color="skyblue",
        yerr=np.transpose([[mean - ci_low, ci_high - mean]
                           for mean, (ci_low,
                                      ci_high) in zip(means, conf_intervals)]),
        capsize=10,
    )

    # Add dots on top of the bars to represent the mean
    for i, mean in enumerate(means):
        ax.plot(x_pos[i], mean, "ro")  # 'ro' means red circle

    ax.set_ylabel("Scores")
    ax.set_title("Mean Scores with 95% Confidence Intervals")

    # Save the image, ensure the directory exists or adjust to the correct path
    plt.savefig(f"{folder_path}/"
                f"score_{exp_name}.png")

    plt.show()


def main(exp_info_file_path, db_path, exp_name, folder_path):
    with open(exp_info_file_path, "r") as file:
        exp_info = json.load(file)

    up_result = get_result(exp_info["up_comment_id"], db_path)
    down_result = get_result(exp_info["down_comment_id"], db_path)
    control_result = get_result(exp_info["control_comment_id"], db_path)
    print(
        "up_result:",
        up_result,
        "down_result:",
        down_result,
        "control_result",
        control_result,
    )
    visualization(up_result, down_result, control_result, exp_name,
                  folder_path)


if __name__ == "__main__":
    main(
        exp_info_file_path=(
            "./experiments/reddit_herding_effect/results_analysis/"
            "result_data/exp_info.json"),
        db_path=("./experiments/reddit_herding_effect/results_analysis/"
                 "result_data/mock_reddit_06-30_06-33-29.db"),
    )
