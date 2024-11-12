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
import os

from langchain import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


class RAG:

    def __init__(
        self,
        llm,
        retriever,
        parser,
        prompt_template,
        format_func,
    ) -> None:
        self.rag_chain = ({
            "examples": retriever | format_func,
            "prompt": RunnablePassthrough()
        }
                          | prompt_template
                          | llm
                          | parser)

    def gen(self, prompt):
        return self.rag_chain.invoke(prompt)


# retriever
file_path = './complete_user_char.csv'

loader = CSVLoader(file_path=file_path, encoding='utf-8')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
splits = text_splitter.split_documents(data)
model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cuda:0"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name=model_name,
                              encode_kwargs=encode_kwargs)

vectorstore = Chroma(persist_directory='./bge', embedding_function=hf)

retriever = vectorstore.as_retriever(top_k=3)


def format_docs(docs):
    formatted_docs = ""
    for i, doc in enumerate(docs, start=1):
        formatted_docs += f"Example {i}:\n{doc.page_content}\n\n"
    # print(Fore.RED + "retriever result: ",
    #       formatted_docs + '\n' + "-"*10 +'\n' + Fore.RESET)
    return formatted_docs.strip().replace("character", "persona")


class User(BaseModel):
    realname: str = Field(description="")
    username: str = Field(description="name of a social media user")
    bio: str = Field(description="bio of a social media user")
    persona: str = Field(description="description of a social media user")


parser = PydanticOutputParser(pydantic_object=User)

topic_template = """Based on the provided personality traits, age, gender and profession, please select 2-3 topics of interest from the given list.
    Input:
        Personality Traits: {mbti}
        Age: {age}
        Gender: {gender}
        Profession: Your profession is Education & Training: Planning, managing and providing education and training services, and related learning support services.
    Available Topics:
        0. **Politics**: The activities related to the governance of a country or area, often involving the struggle for power and influence among various groups and individuals.
        1. **Urban Legends**: Popular stories or myths that are widely circulated as true, especially through word of mouth or the internet, but are often based more on imagination than on fact.
        2. **Business**: The activity of making one's living or forming a livelihood by engaging in commerce, trade, or industry. It encompasses all aspects of managing and operating a company.
        3. **Terrorism & War**: Topics related to acts of violence and intimidation intended to achieve political aims (Terrorism) and large-scale armed conflict between states or nations (War).
        4. **Science & Technology**: The systematic study of the physical and natural world (Science) and the application of scientific knowledge for practical purposes (Technology).
        5. **Entertainment**: Activities that hold the attention and interest of an audience, giving pleasure and delight. It includes a wide range of activities from watching movies to playing games.
        6. **Natural Disasters**: Catastrophic events occurring in nature that cause widespread destruction and loss. Examples include earthquakes, hurricanes, tsunamis, and volcanic eruptions.
        7. **Health**: The state of being free from illness or injury. It also refers to the general well-being of an individual or group, often influenced by factors such as diet, exercise, and mental health.
        8. **Education**: The process of receiving or giving systematic instruction, especially at a school or university. It also encompasses the acquisition of knowledge, skills, values, and habits.
    Output:
    [list of topic number]

    Ensure your output could be parsed to **list**, don't output anything else.
    """  # noqa: E501

topic_system = PromptTemplate(input_variables=["mbti", "age", "gender"],
                              template=topic_template)

# llm

llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=os.environ["OPENAI_API_KEY"],
)

topic_chain = (topic_system | RunnablePassthrough() | llm
               | StrOutputParser(output_type=list))

all_topics = [
    'Politics', 'Urban Legends', 'Business', 'Terrorism & War',
    'Science & Technology', 'Entertainment', 'Natural Disasters', 'Health',
    'Education'
]

rag = RAG(llm=llm,
          retriever=retriever,
          parser=parser,
          prompt_template=PromptTemplate(
              template="RAG: {examples}\n\nPrompt: {prompt}"),
          format_func=format_docs)

prompt_tem = """Please generate a social media user profile based on the provided personal information, including a realname, username, user bio, and a new user persona. The focus should be on creating a fictional background story and detailed interests based on their hobbies and profession.
Input:
    age: {age}
    gender: {gender}
    mbti: {mbti}
    profession: {profession}
    interested topics: {topics}
Output:
{{
    "realname": str, realname,
    "username": str, username,
    "bio": str, bio,
    "persona": str, user persona,
}}
Ensure the output can be directly parsed to **JSON**, do not output anything else.
"""  # noqa: E501


def generate_user_profile(age, gender, mbti, profession, topics):
    # topic_index = topic_chain.invoke(
    #     {"age": age, "gender": gender,
    #      "mbti": mbti, "profession": profession})
    # print(topic_index)
    # topic_index = json.loads(topic_index)
    # topics = [all_topics[i] for i in topic_index]
    prompt = prompt_tem.format(age=age,
                               gender=gender,
                               mbti=mbti,
                               profession=profession,
                               topics=topics)
    # print("retrieved from:\n"+ prompt + '\n')
    user = rag.gen(prompt)
    user_dict = user.dict()
    user_dict["topics"] = topics
    return user_dict


if __name__ == '__main__':
    age = [17, 76, 45]
    gender = ['Female', 'Male', 'Other']
    mbti = ['INTJ', 'ESFP', 'ENFP']
    profession = ['Education & Training', 'Healthcare', 'Business & Finance']
    topics = [['Politics', 'Urban Legends'], ['Business'],
              ['Science & Technology', 'Entertainment']]

    for a in age:
        for g in gender:
            for m in mbti:
                for pro in profession:
                    for top in topics:
                        user = generate_user_profile(a, g, m, pro, top)
                        print(user)
