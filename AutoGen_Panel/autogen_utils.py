import asyncio
import sys
import textwrap
import threading
import time
from ast import literal_eval

import autogen
import chromadb
import isort
import panel as pn
from autogen import Agent, AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.compressible_agent import CompressibleAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.agentchat.contrib.llava_agent import LLaVAAgent
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen.code_utils import extract_code
from configs import (
    DEFAULT_AUTO_REPLY,
    DEFAULT_SYSTEM_MESSAGE,
    Q1,
    Q2,
    Q3,
    TIMEOUT,
    TITLE,
)

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


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
    self,
    messages=None,
    sender=None,
    config=None,
):
    """Generate a reply using autogen.oai."""
    client = self.client if config is None else config
    if client is None:
        return False, None
    if messages is None:
        messages = self._oai_messages[sender]

    # handle 336006 https://cloud.baidu.com/doc/WENXINWORKSHOP/s/tlmyncueh
    _context = messages[-1].pop("context", None)
    _messages = self._oai_system_message + messages
    for idx, msg in enumerate(_messages):
        if idx == 0:
            continue
        if idx % 2 == 1:
            msg["role"] = "user" if msg.get("role") != "function" else "function"
        else:
            msg["role"] = "assistant"
    if len(_messages) % 2 == 1:
        _messages.append({"content": DEFAULT_AUTO_REPLY, "role": "user"})
    # print(f"messages: {_messages}")
    response = client.create(context=_context, messages=_messages)
    # print(f"{response=}")
    return True, client.extract_text_or_function_call(response)[0]


def initialize_agents(
    llm_config, agent_name, system_msg, agent_type, retrieve_config=None, code_execution_config=False
):
    agent_name = agent_name.strip()
    system_msg = system_msg.strip()

    if "RetrieveUserProxyAgent" == agent_type:
        agent = RetrieveUserProxyAgent(
            name=agent_name,
            system_message=system_msg,
            is_termination_msg=_is_termination_msg,
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=5,
            retrieve_config=retrieve_config,
            code_execution_config=code_execution_config,  # set to False if you don't want to execute the code
            default_auto_reply=DEFAULT_AUTO_REPLY,
        )
    elif "GPTAssistantAgent" == agent_type:
        agent = GPTAssistantAgent(
            name=agent_name,
            instructions=system_msg if system_msg else DEFAULT_SYSTEM_MESSAGE,
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
            system_message=system_msg if system_msg else DEFAULT_SYSTEM_MESSAGE,
            llm_config=llm_config,
            compress_config=compress_config,
            is_termination_msg=termination_msg,
        )
    elif "UserProxy" in agent_type:
        agent = globals()[agent_type](
            name=agent_name,
            is_termination_msg=termination_msg,
            human_input_mode="TERMINATE",
            system_message=system_msg,
            default_auto_reply=DEFAULT_AUTO_REPLY,
            max_consecutive_auto_reply=5,
            code_execution_config=code_execution_config,
        )
    else:
        agent = globals()[agent_type](
            name=agent_name,
            is_termination_msg=termination_msg,
            human_input_mode="NEVER",
            system_message=system_msg if system_msg else DEFAULT_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )
    # if any(["ernie" in cfg["model"].lower() for cfg in llm_config["config_list"]]):
    if "ernie" in llm_config["config_list"][0]["model"].lower():
        # Hack for ERNIE Bot models
        # print("Hack for ERNIE Bot models.")
        agent._reply_func_list.pop(-1)
        agent.register_reply([Agent, None], new_generate_oai_reply, -1)
    return agent


async def get_human_input(name, prompt: str, instance=None) -> str:
    """Get human input."""
    if instance is None:
        return input(prompt)
    get_input_widget = pn.widgets.TextAreaInput(placeholder=prompt, name="", sizing_mode="stretch_width")
    get_input_checkbox = pn.widgets.Checkbox(name="Check to Submit Feedback")
    instance.send(pn.Row(get_input_widget, get_input_checkbox), user=name, respond=False)
    ts = time.time()
    while True:
        if time.time() - ts > TIMEOUT:
            instance.send(
                f"You didn't provide your feedback in {TIMEOUT} seconds, skip and use auto-reply.",
                user=name,
                respond=False,
            )
            reply = ""
            break
        if get_input_widget.value != "" and get_input_checkbox.value is True:
            get_input_widget.disabled = True
            reply = get_input_widget.value
            break
        await asyncio.sleep(0.1)
    return reply


async def check_termination_and_human_reply(
    self,
    messages=None,
    sender=None,
    config=None,
    instance=None,
):
    """Check if the conversation should be terminated, and if human reply is provided."""
    if config is None:
        config = self
    if messages is None:
        messages = self._oai_messages[sender]
    message = messages[-1]
    reply = ""
    no_human_input_msg = ""
    if self.human_input_mode == "ALWAYS":
        reply = await get_human_input(
            self.name,
            f"Provide feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: ",
            instance,
        )
        no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
        # if the human input is empty, and the message is a termination message, then we will terminate the conversation
        reply = reply if reply or not self._is_termination_msg(message) else "exit"
    else:
        if self._consecutive_auto_reply_counter[sender] >= self._max_consecutive_auto_reply_dict[sender]:
            if self.human_input_mode == "NEVER":
                reply = "exit"
            else:
                # self.human_input_mode == "TERMINATE":
                terminate = self._is_termination_msg(message)
                reply = await get_human_input(
                    self.name,
                    f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                    if terminate
                    else f"Please give feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: ",
                    instance,
                )
                no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                reply = reply if reply or not terminate else "exit"
        elif self._is_termination_msg(message):
            if self.human_input_mode == "NEVER":
                reply = "exit"
            else:
                # self.human_input_mode == "TERMINATE":
                reply = await get_human_input(
                    self.name,
                    f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: ",
                    instance,
                )
                no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                reply = reply or "exit"

    # print the no_human_input_msg
    if no_human_input_msg:
        print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

    # stop the conversation
    if reply == "exit":
        # reset the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] = 0
        return True, None

    # send the human reply
    if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
        # reset the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] = 0
        return True, reply

    # increment the consecutive_auto_reply_counter
    self._consecutive_auto_reply_counter[sender] += 1
    if self.human_input_mode != "NEVER":
        print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)

    return False, None


async def generate_code(agents, manager, contents, code_editor):
    code = """import autogen
import os
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
)
if not config_list:
    os.environ["MODEL"] = "<your model name>"
    os.environ["OPENAI_API_KEY"] = "<your openai api key>"
    os.environ["OPENAI_BASE_URL"] = "<your openai base url>" # optional

    config_list = autogen.config_list_from_models(
        model_list=[os.environ.get("MODEL", "gpt-35-turbo")],
    )

llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

def termination_msg(x):
    _msg = str(x.get("content", "")).upper().strip().strip("\\n").strip(".")
    return isinstance(x, dict) and (_msg.endswith("TERMINATE") or _msg.startswith("TERMINATE"))

def _is_termination_msg(message):
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

agents = []

"""

    for agent in agents:
        if isinstance(agent, RetrieveUserProxyAgent):
            _code = f"""from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

agent = RetrieveUserProxyAgent(
    name="{agent.name}",
    system_message=\"\"\"{agent.system_message}\"\"\",
    is_termination_msg=_is_termination_msg,
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=5,
    retrieve_config={agent._retrieve_config},
    code_execution_config={agent._code_execution_config},  # set to False if you don't want to execute the code
    default_auto_reply="{DEFAULT_AUTO_REPLY}",
)

"""
        elif isinstance(agent, GPTAssistantAgent):
            _code = f"""from auotgen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

agent = GPTAssistantAgent(
    name="{agent.name}",
    instructions=\"\"\"{agent.system_message}\"\"\",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
)

"""
        elif isinstance(agent, CompressibleAgent):
            _code = f"""from autogen.agentchat.contrib.compressible_agent import CompressibleAgent

compress_config = {{
    "mode": "COMPRESS",
    "trigger_count": 600,  # set this to a large number for less frequent compression
    "verbose": True,  # to allow printing of compression information: contex before and after compression
    "leave_last_n": 2,
}}

agent = CompressibleAgent(
    name="{agent.name}",
    system_message=\"\"\"{agent.system_message}\"\"\",
    llm_config=llm_config,
    compress_config=compress_config,
    is_termination_msg=termination_msg,
)

"""
        elif isinstance(agent, UserProxyAgent):
            _code = f"""from autogen import UserProxyAgent

agent = UserProxyAgent(
    name="{agent.name}",
    is_termination_msg=termination_msg,
    human_input_mode="TERMINATE",
    default_auto_reply="{DEFAULT_AUTO_REPLY}",
    max_consecutive_auto_reply=5,
    code_execution_config={agent._code_execution_config},
)

"""
        elif isinstance(agent, RetrieveAssistantAgent):
            _code = f"""from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent

agent = RetrieveAssistantAgent(
    name="{agent.name}",
    system_message=\"\"\"{agent.system_message}\"\"\",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    retrieve_config={agent._retrieve_config},
)

"""
        elif isinstance(agent, AssistantAgent):
            _code = f"""from autogen import AssistantAgent

agent = AssistantAgent(
    name="{agent.name}",
    system_message=\"\"\"{agent.system_message}\"\"\",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
)

"""
        code += _code + "\n" + "agents.append(agent)\n\n"

    _code = """
for agent in agents:
    if "UserProxy" in str(type(agent)):
        init_sender = agent
        break

if not init_sender:
    init_sender = agents[0]

"""
    code += _code

    if manager:
        _code = """
groupchat = autogen.GroupChat(
    agents=agents, messages=[], max_round=12, speaker_selection_method="round_robin", allow_repeat_speaker=False
)  # todo: auto, sometimes message has no name
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

recipient = manager
"""
    else:
        _code = """
recipient = agents[1] if agents[1] != init_sender else agents[0]
"""
    code += _code

    _code = f"""
if isinstance(init_sender, (RetrieveUserProxyAgent, MathUserProxyAgent)):
    init_sender.initiate_chat(recipient, problem="{contents}")
else:
    init_sender.initiate_chat(recipient, message="{contents}")
"""
    code += _code
    code = textwrap.dedent(code)
    code_editor.value = isort.code(code)
