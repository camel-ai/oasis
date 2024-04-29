import queue
import time
from log import Log
# class Agent(ChatAgent):
from collections.abc import Iterable
import time
import json
def try_except(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            Log.error(f"An error occurred in {func.__name__}: {e}")
    return wrapper


class Memory:
    def __init__(self,agent_id, max_retrieval_limit=None) -> None:
        self._memory = []
        self._agent_id = agent_id
        self.max_retrieval_limit = max_retrieval_limit
      
    def get_total_memory(self):
        return self._memory.copy()

    def get_memory(self):
        if self.max_retrieval_limit is not None:
            if len(self._memory) > self.max_retrieval_limit:
                return self._memory[-self.max_retrieval_limit:].copy()
        return self.get_total_memory() 
    
    def get_agent_id(self):
        return self._agent_id

    def update(self, memory_batch):
        if type(memory_batch) == list:
            self._memory.extend(memory_batch)
        elif type(memory_batch) == dict:
            self._memory.append(memory_batch)
        elif isinstance(memory_batch, Iterable):
            for c in memory_batch:
                self._memory.append(c)
        else:
            self._memory.append(memory_batch)
        

class Profile:
    def __init__(self,agent_id) -> None:
        self._agent_id = agent_id

    def get_agent_id(self):
        return self._agent_id
    
    # def update(self):
    #     pass


class Relationship:
    def __init__(self,agent_id) -> None:
        self._agent_id = agent_id
    
    def get_agent_id(self):
        return self._agent_id
    
    def get_relation_info(self):
        return f"Agent {self._agent_id} relationship"

    def update(self,relation_map):
        pass
  

class Agent:
    @try_except
    def __init__(self, id, args, kargs):
        self.id=id
        self.memory=Memory(agent_id=id)
        self.profile=Profile(agent_id=id)
        self.relationship=Relationship(agent_id=id)
        self.system_prompt = self.init_system_prompt()
        self.tools = []
        self.input_flag=False
        self.output_flag=False
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.function_dict
    
    class GPT_Assistant:
        def __init__(self) -> None:
            self.tools = []
            pass

        def start_chat(self,instruction:str, model='gpt-3.5-turbo-0613'):
            """
            use the gpt assistant
            """
            fake_responce = {'role': 'assistant',
                            'content': None,
                            'function_call': {'name': 'like',
                            'arguments': '{\n  "post_id": 2223\n}'}}
            return fake_responce


    def init_system_prompt(self):
        """
        need to build the prompt based on self.profile
        """
        init_system_prompt = "You are a personal math tutor. Write and run code to answer math questions."
        Log.info(f"Agent {self.id} system prompt created.")
        Log.info(f"Agent {self.id} system prompt: {init_system_prompt}")
        return init_system_prompt
    
    async def process_input(self,content):
        self.input_queue.put(content)
        self.input_flag=True

    async def process_output(self, channel):
        while self.output_flag:
            channel.send_to("Twitter", self.output_queue.get())
        self.output_flag=False

    @try_except
    def update_memory(self, memory_batch):
        self.memory.update(memory_batch)
        Log.info(f'Agent {self.id} memory has been successfully updated')
    
    @try_except
    def update_relation(self, relation_map):
        self.relationship.update(relation_map)
        Log.info(f'Agent {self.id} relation has been successfully updated')

    async def refresh(self):
        data = dict(id=self.id, message='refresh', action='refresh')
        self.output_queue.put(data)
        self.output_flag=True

    
    # def generator_input(self):
    #     current_twitter_post = self.input_queue.get()
    #     system_prompt = self.system_prompt
    #     # past_read_post = self.memory.get_memory()

    #     self.memory.update(memory_cache=current_twitter_post)
    #     assistant_content = dict(instruction=system_prompt,
    #                              tools=[])
    #     return assistant_content

    def think(self):
        """
        temporal remove generator_input() and direct send args
        """
        assistant = self.GPT_Assistant()
        current_twitter_post = self.input_queue.get()
        assistant_message = assistant.start_chat(instruction=current_twitter_post)
        self.memory.update(memory_cache=current_twitter_post)
        if assistant_message["function_call"]:
            function_name = assistant_message["function_call"]["name"]
            function_args = json.loads(assistant_message["function_call"]["arguments"])
            self.action_queue.put((function_name,function_args))


    async def running(self):
        while True:
            self.get_twitter()
            if self.input_flag:
                self.think()
                while self.action_queue.qsize() >0:
                    function_name, function_args = self.action_queue.get()
                    getattr(self, function_name)(*function_args)
                if self.output_flag == False:
                    self.input_flag = False
            time.sleep(time)

    async def search_tweet(self, query:str):
        data = dict(id=self.id, message=query, action='search_tweet')
        self.output_queue.put(data)
        self.output_flag=True

    async def search_user(self, query:str):
        data = dict(id=self.id,  message=query, action='search_user')
        self.output_queue.put(data)
        self.output_flag=True


    async def search_tweet(self, query:str):
        data = dict(id=self.id, message=query, action='search_tweet')
        self.output_queue.put(data)
        self.output_flag=True

    async def create_tweet(self, content:str):
        data = dict(id=self.id, message=content, action='create_tweet')
        self.output_queue.put(data)
        self.output_flag=True


    async def like(self, tweet_id:str):
        data = dict(id=id, message=tweet_id, action='like')
        self.output_queue.put(data)
        self.output_flag=True


    async def unlike(self, tweet_id:str):
        data = dict(id=self.id, message=tweet_id, action='like')
        self.output_queue.put(data)
        self.output_flag=True


    async def follow(self, user_id:int):
        data = dict(id=self.id, message=str(user_id), action='follow')
        self.output_queue.put(data)
        self.output_flag=True


    async def unfollow(self, user_id:int):
        data = dict(id=self.id, message=str(user_id), action='unfollow')
        self.output_queue.put(data)
        self.output_flag=True


    async def mute(self, user_id:int):
        data = dict(id=self.id, message=str(user_id), action='mute')
        self.output_queue.put(data)
        self.output_flag=True


    async def unmute(self, user_id:int):
        data = dict(id=self.id, message=str(user_id), action='unmute')
        self.output_queue.put(data)
        self.output_flag=True


    async def trend(self):
        data = dict(id=self.id, message='trend', action='trend')
        self.output_queue.put(data)
        self.output_flag=True

