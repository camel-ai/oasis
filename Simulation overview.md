Simulation overview
1. DATABASE SCHEMA
The system uses SQLite with 15 core tables defined in oasis/social_platform/schema/:
Core Tables:
user.sql - Agent profiles
user_id, agent_id, user_name, name, bio, created_at, 
num_followings, num_followers
post.sql - Content with engagement metrics
post_id, user_id, original_post_id, content, quote_content,
num_likes, num_dislikes, num_shares, num_reports
trace.sql - Critical: Action history log
user_id, created_at, action, info (JSON)
-- Records EVERY agent action for analysis
rec.sql - Recommendation buffer
user_id, post_id
-- Stores personalized feed for each user
Additional tables: follow, like, dislike, comment, mute, report, product (Reddit), chat_group, group_member, group_message
2. PERSONA CARDS
Data Structure
Personas are defined in oasis/social_platform/config/user.py:
@dataclass
class UserInfo:
    user_name: str
    name: str
    description: str
    profile: dict  # Detailed persona data
    recsys_type: str = "twitter"
    is_controllable: bool = False
Example Persona Format
From data/reddit/user_data_36.json:
{
  "realname": "James Miller",
  "username": "millerhospitality",
  "bio": "Passionate about hospitality & tourism...",
  "persona": "James is a seasoned professional...",
  "age": 40,
  "gender": "male",
  "mbti": "ESTJ",
  "country": "UK",
  "profession": "Hospitality & Tourism",
  "interested_topics": ["Economics", "Business"]
}
System Prompt Generation
Personas are converted to system prompts in oasis/social_platform/config/user.py:
system_content = f"""
# OBJECTIVE
You're a Twitter user, and I'll present you with posts. 
Choose actions from the available functions.

# SELF-DESCRIPTION
Your actions should match your personality:
{description}

# RESPONSE METHOD
Perform actions by tool calling.
"""
3. AGENT INPUT FLOW
Input Components
Agents receive environment state from oasis/social_agent/agent_environment.py:
async def to_text_prompt(self) -> str:
    # 1. Recommended posts from rec table
    posts = await self.action.refresh()
    
    # 2. Social stats
    num_followers = fetch_from_db(agent_id)
    num_followings = ...
    
    # 3. Group messages (if applicable)
    groups = await self.action.listen_from_group()
    
    return formatted_environment_string
Environment Prompt Example:
You have 42 followers and are following 15 users.
After refreshing, you see these posts:
[
  {
    "post_id": 1,
    "user_id": 23,
    "content": "This is an example post...",
    "num_likes": 5,
    "created_at": "2024-05-14T12:00:00Z"
  },
  ...
]
Pick one action that best reflects your current inclination.
4. AGENT LOGIC & ARCHITECTURE
SocialAgent Class
Core implementation in oasis/social_agent/agent.py:
class SocialAgent(ChatAgent):  # Inherits CAMEL ChatAgent
    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,  # Persona
        model: BaseModelBackend,  # LLM
        channel: Channel,  # Platform communication
        available_actions: list[ActionType],  # Allowed behaviors
        agent_graph: AgentGraph  # Social network
    )
Decision-Making Process
From oasis/social_agent/agent.py:
async def perform_action_by_llm(self):
    # 1. Get current environment
    env_prompt = await self.env.to_text_prompt()
    
    # 2. Create observation message
    user_msg = BaseMessage.make_user_message(
        content=f"Observe platform: {env_prompt}"
    )
    
    # 3. Send to LLM with memory context + tools
    response = await self.astep(user_msg)
    
    # 4. Execute tool calls
    for tool_call in response.info['tool_calls']:
        action_name = tool_call.tool_name
        args = tool_call.args
        # Executes via SocialAction → Channel → Platform
Memory System
Type: CAMEL MemoryRecord (conversation history)
Storage: System prompt + observations + actions
Context: Retrieved via memory.get_context() for LLM calls
Persistence: In-memory during simulation
5. AVAILABLE ACTIONS (23 Total)
Defined in oasis/social_agent/agent_action.py:
Content Actions
create_post, create_comment
repost, quote_post
Engagement Actions
like_post, unlike_post, dislike_post
like_comment, unlike_comment, dislike_comment
Social Actions
follow, unfollow, mute, unmute
Discovery Actions
search_posts, search_user, trend, refresh
Special Actions
report_post (moderation)
purchase_product (Reddit e-commerce)
create_group, join_group, leave_group, send_to_group (chat)
do_nothing, interview (research)
6. SIMULATION DESIGN
OasisEnv Orchestration
Core loop in oasis/environment/env.py:
class OasisEnv:
    async def reset(self):
        # 1. Start platform event loop
        self.platform_task = asyncio.create_task(
            self.platform.running()
        )
        
        # 2. Sign up all agents
        self.agent_graph = await generate_custom_agents(
            channel=self.channel, 
            agent_graph=self.agent_graph
        )
    
    async def step(self, actions: dict):
        # 1. Update recommendation system
        await self.platform.update_rec_table()
        
        # 2. Execute all agent actions in parallel
        tasks = []
        for agent, action in actions.items():
            if isinstance(action, LLMAction):
                tasks.append(agent.perform_action_by_llm())
            elif isinstance(action, ManualAction):
                tasks.append(agent.perform_action_by_data(...))
        
        await asyncio.gather(*tasks)
        
        # 3. Advance time
        self.platform.sandbox_clock.time_step += 1
    
    async def close(self):
        # Stop platform and save database
Time Management
From oasis/clock/clock.py:
class Clock:
    def __init__(self, k: int = 1):  # k = acceleration factor
        self.real_start_time = datetime.now()
        self.k = k
        self.time_step = 0  # Discrete steps (Twitter)
    
    def time_transfer(self, now_time, start_time) -> datetime:
        # Accelerated time: k × real_time_diff
        time_diff = now_time - self.real_start_time
        return start_time + (self.k * time_diff)
7. ACTION EXECUTION PIPELINE
Platform Action Handler
From oasis/social_platform/platform.py:
async def running(self):
    while True:
        # 1. Receive action from agent via channel
        message_id, (agent_id, message, action) = \
            await self.channel.receive_from()
        
        # 2. Get action handler method
        action_function = getattr(self, action.value)
        
        # 3. Execute with validation
        result = await action_function(agent_id, message)
        
        # 4. Update database tables
        # 5. Log to trace table
        # 6. Return result to agent
        await self.channel.send_to((message_id, agent_id, result))
Example: like_post Implementation
From oasis/social_platform/platform.py:
async def like_post(self, agent_id: int, post_id: int):
    # 1. Check for duplicate like
    if like_exists(post_id, user_id):
        return {"success": False, "error": "Already liked"}
    
    # 2. Validate self-rating (optional)
    if not self.allow_self_rating:
        if is_own_post(post_id, user_id):
            return {"success": False}
    
    # 3. Update post.num_likes
    UPDATE post SET num_likes = num_likes + 1 WHERE post_id = ?
    
    # 4. Insert like record
    INSERT INTO 'like' (post_id, user_id, created_at) VALUES (...)
    
    # 5. Log action to trace table
    INSERT INTO trace (user_id, action, info, created_at) VALUES (...)
    
    # 6. Return success
    return {"success": True, "like_id": like_id}
8. SOCIAL NETWORK STRUCTURE
AgentGraph Class
From oasis/social_agent/agent_graph.py:
class AgentGraph:
    def __init__(
        self,
        backend: Literal["igraph", "neo4j"] = "igraph"
    ):
        # igraph: In-memory (< 100K agents)
        # neo4j: Persistent (> 100K agents)
        
        self.graph = ig.Graph(directed=True)
        self.agent_mappings: dict[int, SocialAgent] = {}
    
    def add_agent(self, agent: SocialAgent):
        self.graph.add_vertex(agent.social_agent_id)
    
    def add_edge(self, agent_id_0: int, agent_id_1: int):
        # Directed edge = follow relationship
        self.graph.add_edge(agent_id_0, agent_id_1)
Relationship Impact
Follow → Affects content visibility in feed
Mute → Filters content without unfollowing
Network structure → Influences recommendation algorithm
Database → follow table stores persistent relationships
Graph → In-memory for network analysis
9. DATA FLOW ARCHITECTURE
┌─────────────────────────────────────────────────┐
│                  SIMULATION STEP                 │
└─────────────────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Agent 1 │      │ Agent 2 │      │ Agent N │
└────┬────┘      └────┬────┘      └────┬────┘
     │                │                │
     │ 1. Get Environment               │
     │ ─────────────────────────────────┤
     │                                  │
     ├──► SocialEnvironment.to_text_prompt()
     │           │
     │           ├─► Query rec table (recommended posts)
     │           ├─► Get follower/following counts
     │           └─► Format as text
     │
     │ 2. LLM Decision
     │ ─────────────────────────────────┤
     │                                  │
     ├──► CAMEL ChatAgent.astep()
     │           │
     │           ├─► Retrieve memory (persona + history)
     │           ├─► Send to LLM with tools
     │           └─► LLM returns tool calls
     │
     │ 3. Execute Action
     │ ─────────────────────────────────┤
     │                                  │
     ├──► SocialAction.like_post(post_id=1)
     │           │
     │           └─► Channel.write_to_receive_queue()
     │                      │
     │                      ▼
     │              ┌──────────────┐
     │              │   Platform   │
     │              │   running()  │
     │              └──────┬───────┘
     │                     │
     │                     ├─► Validate action
     │                     ├─► Update database
     │                     │   (post, like, trace tables)
     │                     └─► Return result
     │                      │
     │           ┌──────────┘
     │           ▼
     │    Channel.send_to()
     │           │
     │           ▼
     └──► Update agent memory
10. KEY FILE LOCATIONS
Core Implementation
Agent Logic: oasis/social_agent/agent.py
Actions: oasis/social_agent/agent_action.py
Environment: oasis/social_agent/agent_environment.py
Platform: oasis/social_platform/platform.py
Orchestration: oasis/environment/env.py
Configuration
Persona: oasis/social_platform/config/user.py
Database: oasis/social_platform/database.py
Schemas: oasis/social_platform/schema/*.sql
Examples
Quick Start: examples/quick_start.py
Twitter: examples/twitter_quick_start.py
Reddit: examples/reddit_simulation_openai.py
11. SCALABILITY CONSIDERATIONS
Scale	Agents	Backend	Database	Notes
Small	< 1K	igraph	In-memory	Standard setup
Medium	1K-100K	igraph	SQLite disk	TwHIN-BERT recsys
Large	100K-1M	Neo4j	Optimized queries	List-based agent storage
Note: At 1M+ agents, the codebase bypasses AgentGraph class for performance (oasis/social_agent/agents_generator.py)
This architecture enables realistic, scalable social media simulations with LLM-driven agents that exhibit complex social behaviors based on detailed persona configurations. The async-first design allows massive parallelization, while the trace table provides complete action history for analysis.