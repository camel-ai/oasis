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
import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def load_misinfo_data():
    """Load misinfo.csv data"""
    df = pd.read_csv('./data/hci/misinfo.csv')
    # Create mapping from post_id to label
    misinfo_map = {}
    for _, row in df.iterrows():
        post_id = row['id']
        label = row['label']
        # Convert post_id to string for matching
        misinfo_map[str(post_id)] = ('Official News'
                                     if label else 'Misinformation')
    return misinfo_map


def find_root_post(post_id, posts_dict):
    """Trace back to find the root post"""
    current_post = posts_dict.get(post_id)
    if current_post is None:
        return None

    # If original_post_id is None, this is the root post
    if current_post['original_post_id'] is None:
        return post_id

    # Recursively find the root post
    return find_root_post(current_post['original_post_id'], posts_dict)


def analyze_posts():
    """Analyze posts and aggregate by time step"""
    # Connect to database
    conn = sqlite3.connect('./data/db/misinfo_hci_at3.db')
    cursor = conn.cursor()

    # Read all posts
    cursor.execute(
        "SELECT post_id, user_id, original_post_id, content, created_at "
        "FROM post")
    posts_data = cursor.fetchall()

    # Create posts dictionary
    posts_dict = {}
    for post_id, user_id, original_post_id, content, created_at in posts_data:
        posts_dict[post_id] = {
            'user_id': user_id,
            'original_post_id': original_post_id,
            'content': content,
            'created_at': created_at
        }

    # Find misinfo data by content matching
    content_to_label = {}
    df = pd.read_csv('./data/hci/misinfo.csv')
    for _, row in df.iterrows():
        content = row['source_tweet']
        label = 'Official News' if row['label'] else 'Misinformation'
        content_to_label[content] = label

    # Count data for each time step
    time_step_stats = defaultdict(lambda: {
        'Official News': 0,
        'Misinformation': 0
    })

    for post_id, post_info in posts_dict.items():
        # Find the root post
        root_post_id = find_root_post(post_id, posts_dict)

        if root_post_id:
            root_post = posts_dict[root_post_id]
            root_content = root_post['content']

            # Find label by content matching
            if root_content in content_to_label:
                time_step = post_info['created_at']
                label = content_to_label[root_content]
                time_step_stats[time_step][label] += 1

    conn.close()
    return time_step_stats


def plot_results(time_step_stats):
    """Plot line chart"""
    # Prepare data
    time_steps = sorted(time_step_stats.keys())
    official_news_counts = [
        time_step_stats[ts]['Official News'] for ts in time_steps
    ]
    misinformation_counts = [
        time_step_stats[ts]['Misinformation'] for ts in time_steps
    ]

    # Create figure
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps,
             official_news_counts,
             'b-',
             label='Official News',
             linewidth=2,
             marker='o')
    plt.plot(time_steps,
             misinformation_counts,
             'r-',
             label='Misinformation',
             linewidth=2,
             marker='s')

    # Set chart properties
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.title('Posts Distribution Over Time Steps',
              fontsize=14,
              fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Set x-axis labels
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(
        './visualization/twitter_simulation/human_interect/'
        'posts_distribution_at3.png',
        dpi=300,
        bbox_inches='tight')

    # Show figure
    plt.show()

    # Print statistics
    print("Time step statistics:")
    print("=" * 50)
    for ts in time_steps:
        print(f"Time step {ts}:")
        print(f"  Official News: {time_step_stats[ts]['Official News']}")
        print(f"  Misinformation: {time_step_stats[ts]['Misinformation']}")
        print()


if __name__ == "__main__":
    print("Start analyzing posts data...")
    time_step_stats = analyze_posts()
    print("Analysis complete, start plotting chart...")
    plot_results(time_step_stats)
    print("Chart has been saved as posts_distribution.png")
