import sys
import threading
from ast import literal_eval

import autogen
import chromadb
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.compressible_agent import CompressibleAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.agentchat.contrib.llava_agent import LLaVAAgent
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen.code_utils import extract_code


def get_retrieve_config(docs_path, model_name, collection_name):
    return {
        "docs_path": literal_eval(docs_path),
        "chunk_token_size": 1000,
        "model": model_name,
        "embedding_model": "all-mpnet-base-v2",
        "get_or_create": True,
        "client": chromadb.PersistentClient(path=".chromadb"),
        "collection_name": collection_name,
    }


# autogen.ChatCompletion.start_logging()
def termination_msg(x):
    """Check if a message is a termination message."""
    _msg = str(x.get("content", "")).upper().strip().strip("\n").strip(".")
    return isinstance(x, dict) and (_msg.endswith("TERMINATE") or _msg.startswith("TERMINATE"))


def _is_termination_msg(message):
    """Check if a message is a termination message.
    Terminate when no code block is detected. Currently only detect python code blocks.
    """
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        # todo: support more languages
        if c[0] == "python":
            contain_code = True
            break
    return not contain_code


def new_generate_oai_reply(
    cls,
    messages=None,
    sender=None,
    config=None,
):
    """Generate a reply using autogen.oai."""
    client = cls.client if config is None else config
    if client is None:
        return False, None
    if messages is None:
        messages = cls._oai_messages[sender]

    # handle 336006 https://cloud.baidu.com/doc/WENXINWORKSHOP/s/tlmyncueh
    _context = messages[-1].pop("context", None)
    _messages = cls._oai_system_message + messages
    for idx, msg in enumerate(_messages):
        if idx == 0:
            continue
        if idx % 2 == 1:
            msg["role"] = "user" if msg.get("role") != "function" else "function"
        else:
            msg["role"] = "assistant"
    print(f"messages: {_messages}")
    response = client.create(context=_context, messages=_messages)
    print(f"{response=}")
    return True, client.extract_text_or_function_call(response)[0]


def initialize_agents(
    llm_config, agent_name, system_msg, agent_type, retrieve_config=None, code_execution_config=False
):
    if "RetrieveUserProxyAgent" == agent_type:
        agent = RetrieveUserProxyAgent(
            name=agent_name,
            is_termination_msg=termination_msg,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            retrieve_config=retrieve_config,
            code_execution_config=code_execution_config,  # set to False if you don't want to execute the code
            default_auto_reply="Please reply exactly `TERMINATE` to me if the task is done.",
        )
    elif "GPTAssistantAgent" == agent_type:
        agent = GPTAssistantAgent(
            name=agent_name,
            instructions=system_msg,
            llm_config=llm_config,
            is_termination_msg=termination_msg,
        )
    elif "CompressibleAgent" == agent_type:
        compress_config = {
            "mode": "COMPRESS",
            "trigger_count": 600,  # set this to a large number for less frequent compression
            "verbose": True,  # to allow printing of compression information: contex before and after compression
            "leave_last_n": 2,
        }
        agent = CompressibleAgent(
            name=agent_name,
            system_message=system_msg,
            llm_config=llm_config,
            compress_config=compress_config,
            is_termination_msg=termination_msg,
        )
    elif "UserProxy" in agent_type:
        agent = globals()[agent_type](
            name=agent_name,
            is_termination_msg=termination_msg,
            human_input_mode="NEVER",
            system_message=system_msg,
            default_auto_reply="Please reply exactly `TERMINATE` to me if the task is done.",
            max_consecutive_auto_reply=5,
            code_execution_config=code_execution_config,
        )
    else:
        agent = globals()[agent_type](
            name=agent_name,
            is_termination_msg=termination_msg,
            human_input_mode="NEVER",
            system_message=system_msg,
            llm_config=llm_config,
        )

    # agent.generate_oai_reply = new_generate_oai_reply.__get__(agent)
    return agent
