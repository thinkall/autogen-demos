import os
import time

import autogen
import chromadb
import openai
import panel as pn
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from panel.chat import ChatInterface

pn.extension(design="material")

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
)

llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}


# autogen.ChatCompletion.start_logging()
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    system_message="The boss who ask questions and give tasks.",
    code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
)

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

coder = AssistantAgent(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

pm = autogen.AssistantAgent(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

reviewer = autogen.AssistantAgent(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)


def send_messages(recipient, messages, sender, config):
    chat_interface.send(messages[-1]["content"], user=messages[-1]["name"], respond=False)
    print(
        f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}"
    )
    return False, None  # required to ensure the agent communication flow continues


boss.register_reply(
    [autogen.Agent, None],
    reply_func=send_messages,
    config={"callback": None},
)
boss_aid.register_reply(
    [autogen.Agent, None],
    reply_func=send_messages,
    config={"callback": None},
)
coder.register_reply(
    [autogen.Agent, None],
    reply_func=send_messages,
    config={"callback": None},
)
pm.register_reply(
    [autogen.Agent, None],
    reply_func=send_messages,
    config={"callback": None},
)
reviewer.register_reply(
    [autogen.Agent, None],
    reply_func=send_messages,
    config={"callback": None},
)

groupchat = autogen.GroupChat(
    agents=[boss, coder, pm, reviewer], messages=[], max_round=12, speaker_selection_method="round_robin"
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    # Start chatting with boss_aid as this is the user proxy agent.
    # boss_aid.initiate_chat(manager, problem=contents, n_results=3)
    boss.initiate_chat(manager, message=contents)


chat_interface = pn.chat.ChatInterface(callback=callback)

chat_interface.send(
    "Enter a message in the TextInput below to start chat with AutoGen!",
    user="System",
    respond=False,
)
chat_interface.servable()
