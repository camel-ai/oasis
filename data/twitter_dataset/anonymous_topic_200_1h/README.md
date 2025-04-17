# Twitter Dataset Documentation

## User Data Structure

The OASIS user data is stored in CSV files with the following structure:

### CSV File Format

Each CSV file contains user information with the following columns:

- **user_id**: Unique identifier for each agent
- **name**: Agent's display name
- **username**: Agent's username (handle)
- **following_agentid_list**: List of user IDs that this agent follows
- **previous_tweets**: Initial tweets that will be injected into the environment at the start of simulation
- **user_char**: Agent's self-description (used in the agent's system prompt to establish initial identity)
- **description**: Same as user_char, provides the agent's profile description

### Usage

During simulation:

1. Users will sign up at the beginning stage (simulated by created_at timestamp)
1. Initial tweets will be injected into the environment
1. The user_char and description fields will be incorporated into each agent's system prompt to establish their identity
1. Following relationships defined in following_agentid_list will determine the initial social graph

This dataset provides the foundation for simulating realistic Twitter-like interactions between AI agents with diverse backgrounds and interests.
