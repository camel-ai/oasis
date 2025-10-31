# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""
Visualize OASIS simulation results from the database.

This script provides functions to:
1. View and analyze the trace table
2. Create visualizations of agent interactions
3. Generate network graphs of social interactions
"""

import json
import sqlite3
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def view_trace_table(
    db_path: str = "./data/twitter_simulation.db",
    limit: int | None = None
) -> pd.DataFrame:
    """
    View the trace table from the simulation database.

    Args:
        db_path: Path to the simulation database
        limit: Maximum number of rows to return (None for all rows)

    Returns:
        DataFrame containing the trace data
    """
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM trace"
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def get_action_summary(db_path: str = "./data/twitter_simulation.db") -> pd.DataFrame:
    """
    Get a summary of all actions performed in the simulation.

    Args:
        db_path: Path to the simulation database

    Returns:
        DataFrame with action counts and statistics
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT 
            action,
            COUNT(*) as total_actions,
            COUNT(DISTINCT user_id) as unique_users
        FROM trace
        WHERE action != 'sign_up'
        GROUP BY action
        ORDER BY total_actions DESC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def create_interaction_network(
    db_path: str = "./data/twitter_simulation.db",
    action_types: list[str] | None = None
) -> nx.DiGraph:
    """
    Create a network graph of agent interactions.

    Args:
        db_path: Path to the simulation database
        action_types: List of action types to include (None for all)

    Returns:
        NetworkX directed graph of interactions
    """
    conn = sqlite3.connect(db_path)

    # Get all posts and their creators
    posts_df = pd.read_sql("SELECT post_id, user_id, content FROM post", conn)

    # Get all interactions (likes, reposts, comments)
    likes_df = pd.read_sql(
        "SELECT user_id, post_id FROM like",
        conn
    )
    reposts_df = pd.read_sql(
        "SELECT user_id, original_post_id as post_id FROM post WHERE original_post_id IS NOT NULL AND quote_content IS NULL",
        conn
    )
    quotes_df = pd.read_sql(
        "SELECT user_id, original_post_id as post_id FROM post WHERE original_post_id IS NOT NULL AND quote_content IS NOT NULL",
        conn
    )
    comments_df = pd.read_sql(
        "SELECT user_id, post_id FROM comment",
        conn
    )

    conn.close()

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes (all users)
    all_users = set(posts_df['user_id'].unique())
    all_users.update(likes_df['user_id'].unique())
    all_users.update(reposts_df['user_id'].unique())
    all_users.update(quotes_df['user_id'].unique())
    all_users.update(comments_df['user_id'].unique())

    for user in all_users:
        G.add_node(user)

    # Add edges for likes
    for _, row in likes_df.iterrows():
        liker = row['user_id']
        post_id = row['post_id']
        # Find who created the post
        post_creator = posts_df[posts_df['post_id']
                                == post_id]['user_id'].values
        if len(post_creator) > 0 and liker != post_creator[0]:
            if G.has_edge(liker, post_creator[0]):
                G[liker][post_creator[0]]['weight'] += 1
            else:
                G.add_edge(liker, post_creator[0],
                           weight=1, interaction='like')

    # Add edges for reposts
    for _, row in reposts_df.iterrows():
        reposter = row['user_id']
        post_id = row['post_id']
        if pd.notna(post_id):
            post_creator = posts_df[posts_df['post_id']
                                    == post_id]['user_id'].values
            if len(post_creator) > 0 and reposter != post_creator[0]:
                if G.has_edge(reposter, post_creator[0]):
                    G[reposter][post_creator[0]]['weight'] += 2
                else:
                    G.add_edge(
                        reposter, post_creator[0], weight=2, interaction='repost')

    # Add edges for quote posts
    for _, row in quotes_df.iterrows():
        quoter = row['user_id']
        post_id = row['post_id']
        if pd.notna(post_id):
            post_creator = posts_df[posts_df['post_id']
                                    == post_id]['user_id'].values
            if len(post_creator) > 0 and quoter != post_creator[0]:
                if G.has_edge(quoter, post_creator[0]):
                    G[quoter][post_creator[0]]['weight'] += 1.5
                else:
                    G.add_edge(
                        quoter, post_creator[0], weight=1.5, interaction='quote')

    # Add edges for comments
    for _, row in comments_df.iterrows():
        commenter = row['user_id']
        post_id = row['post_id']
        post_creator = posts_df[posts_df['post_id']
                                == post_id]['user_id'].values
        if len(post_creator) > 0 and commenter != post_creator[0]:
            if G.has_edge(commenter, post_creator[0]):
                G[commenter][post_creator[0]]['weight'] += 1.5
            else:
                G.add_edge(
                    commenter, post_creator[0], weight=1.5, interaction='comment')

    return G


def visualize_network(
    G: nx.DiGraph,
    output_path: str | None = None,
    figsize: tuple[int, int] = (16, 12),
    min_degree: int = 0
) -> None:
    """
    Visualize the interaction network.

    Args:
        G: NetworkX graph to visualize
        output_path: Path to save the figure (None to display)
        figsize: Figure size as (width, height)
        min_degree: Minimum degree for nodes to display (filter isolated nodes)
    """
    # Filter nodes by degree if specified
    if min_degree > 0:
        nodes_to_keep = [n for n in G.nodes() if G.degree(n) >= min_degree]
        G = G.subgraph(nodes_to_keep).copy()

    plt.figure(figsize=figsize)

    # Calculate node sizes based on degree (influence)
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 100 + 100 for node in G.nodes()]

    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Draw the network
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        alpha=0.7
    )

    # Draw edges with varying thickness based on weight
    edges = G.edges()
    weights = [G[u][v].get('weight', 1) for u, v in edges]

    nx.draw_networkx_edges(
        G, pos,
        width=[w * 0.5 for w in weights],
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        edge_color='gray'
    )

    # Draw labels for high-degree nodes only
    high_degree_nodes = {n: n for n in G.nodes() if degrees[n] >= 3}
    nx.draw_networkx_labels(
        G, pos,
        labels=high_degree_nodes,
        font_size=8
    )

    plt.title(
        "Agent Interaction Network\n(Node size = activity level, Edge width = interaction strength)")
    plt.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved to: {output_path}")
    else:
        plt.show()


def plot_action_timeline(
    db_path: str = "./data/twitter_simulation.db",
    output_path: str | None = None
) -> None:
    """
    Plot a timeline of actions in the simulation.

    Args:
        db_path: Path to the simulation database
        output_path: Path to save the figure (None to display)
    """
    df = view_trace_table(db_path)

    # Filter out sign_up actions
    df = df[df['action'] != 'sign_up'].copy()

    # Count actions by type
    action_counts = df['action'].value_counts()

    # Create bar plot
    plt.figure(figsize=(12, 6))
    action_counts.plot(kind='bar', color='steelblue')
    plt.title('Action Distribution in Simulation',
              fontsize=14, fontweight='bold')
    plt.xlabel('Action Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Action timeline saved to: {output_path}")
    else:
        plt.show()


def analyze_simulation(db_path: str = "./data/twitter_simulation.db") -> dict[str, Any]:
    """
    Perform comprehensive analysis of the simulation.

    Args:
        db_path: Path to the simulation database

    Returns:
        Dictionary with analysis results
    """
    conn = sqlite3.connect(db_path)

    # Get basic stats
    trace_df = pd.read_sql(
        "SELECT * FROM trace WHERE action != 'sign_up'", conn)
    users_df = pd.read_sql("SELECT COUNT(*) as total_users FROM user", conn)
    posts_df = pd.read_sql("SELECT COUNT(*) as total_posts FROM post", conn)

    conn.close()

    results = {
        'total_users': int(users_df['total_users'].iloc[0]),
        'total_posts': int(posts_df['total_posts'].iloc[0]),
        'total_actions': len(trace_df),
        'action_breakdown': trace_df['action'].value_counts().to_dict(),
        'active_users': int(trace_df['user_id'].nunique()),
        'avg_actions_per_user': len(trace_df) / trace_df['user_id'].nunique()
    }

    return results


def main() -> None:
    """Main function to demonstrate visualization capabilities."""
    db_path = "./data/twitter_simulation.db"

    print("=" * 60)
    print("OASIS SIMULATION ANALYSIS")
    print("=" * 60)

    # 1. View trace table sample
    print("\n1. TRACE TABLE SAMPLE (First 10 rows)")
    print("-" * 60)
    trace_df = view_trace_table(db_path, limit=10)
    print(trace_df.to_string(index=False))

    # 2. Get action summary
    print("\n\n2. ACTION SUMMARY")
    print("-" * 60)
    summary_df = get_action_summary(db_path)
    print(summary_df.to_string(index=False))

    # 3. Comprehensive analysis
    print("\n\n3. SIMULATION STATISTICS")
    print("-" * 60)
    analysis = analyze_simulation(db_path)
    for key, value in analysis.items():
        if key == 'action_breakdown':
            print(f"\n{key}:")
            for action, count in value.items():
                print(f"  - {action}: {count}")
        else:
            print(f"{key}: {value}")

    # 4. Create visualizations
    print("\n\n4. CREATING VISUALIZATIONS")
    print("-" * 60)

    # Plot action distribution
    print("Creating action timeline...")
    plot_action_timeline(db_path, output_path="./data/action_timeline.png")

    # Create and visualize network
    print("Creating interaction network...")
    G = create_interaction_network(db_path)
    print(
        f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Visualize (filtering for nodes with at least 1 connection)
    visualize_network(
        G, output_path="./data/interaction_network.png", min_degree=1)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - ./data/action_timeline.png")
    print("  - ./data/interaction_network.png")


if __name__ == "__main__":
    main()
