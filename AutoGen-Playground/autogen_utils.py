import atexit
import sys
import threading
import time
from itertools import chain

import anyio
import cloudpickle
import redis
from autogen import Agent, AssistantAgent, OpenAIWrapper, UserProxyAgent
from autogen.code_utils import extract_code
from diskcache import Cache
from gradio import ChatInterface, Request
from gradio.helpers import special_args
from pydantic.dataclasses import dataclass

LOG_LEVEL = "INFO"
TIMEOUT = 10  # seconds
CACHE_EXPIRE_TIME = 7200  # 2 hours
_cache = Cache(".cache/gradio")

# r = redis.Redis(host="localhost", port=6379, db=0)


def close_cache():
    _cache.close()


atexit.register(close_cache)


class myChatInterface(ChatInterface):
    async def _submit_fn(
        self,
        message: str,
        history_with_input: list[list[str | None]],
        request: Request,
        *args,
    ) -> tuple[list[list[str | None]], list[list[str | None]]]:
        history = history_with_input[:-1]
        inputs, _, _ = special_args(self.fn, inputs=[message, history, *args], request=request)

        if self.is_async:
            await self.fn(*inputs)
        else:
            await anyio.to_thread.run_sync(self.fn, *inputs, limiter=self.limiter)

        # history.append([message, response])
        return history, history


@dataclass
class AgentMessage:
    session_hash: str = None
    sender: str = None
    messages: str = None


def get_history(session_hash):
    # msg = r.get(session_hash)
    msg = _cache.get(session_hash)
    if msg is None:
        msg = cloudpickle.dumps([])
    return msg


def save_history(session_hash, agent_message: AgentMessage):
    hist = cloudpickle.loads(get_history(session_hash))
    hist.append(agent_message)
    hist = cloudpickle.dumps(hist)
    # r.set(session_hash, hist, ex=CACHE_EXPIRE_TIME)
    _cache.set(session_hash, hist, expire=CACHE_EXPIRE_TIME)


def delete_history(session_hash):
    # r.delete(session_hash)
    _cache.delete(session_hash)


def flatten_chain(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


class thread_with_trace(threading.Thread):
    # https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
    # https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
        self._return = None

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def globaltrace(self, frame, event, arg):
        if event == "call":
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == "line":
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True

    def join(self, timeout=0):
        threading.Thread.join(self, timeout)
        return self._return


def update_agent_history(recipient, messages, sender, config, session_hash=None):
    if config is None:
        config = recipient
    if messages is None:
        messages = recipient._oai_messages[sender]
    message = messages[-1]
    current_agent_message = AgentMessage(
        session_hash=session_hash, sender=sender.name, messages=message.get("content", "")
    )
    print(f"update_agent_history: {current_agent_message}")
    save_history(session_hash, current_agent_message)
    time.sleep(2)
    return False, None  # required to ensure the agent communication flow continues


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


def initialize_agents(config_list):
    assistant = AssistantAgent(
        name="assistant",
        max_consecutive_auto_reply=5,
        llm_config={
            "cache_seed": 42,
            "timeout": TIMEOUT,
            "config_list": config_list,
        },
    )

    userproxy = UserProxyAgent(
        name="userproxy",
        human_input_mode="NEVER",
        is_termination_msg=_is_termination_msg,
        max_consecutive_auto_reply=5,
        # code_execution_config=False,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # set to True or image name like "python:3" to use docker
        },
    )

    return assistant, userproxy


def chat_to_oai_message(chat_history):
    """Convert chat history to OpenAI message format."""
    messages = []
    if LOG_LEVEL == "DEBUG":
        print(f"chat_to_oai_message: {chat_history}")
    for msg in chat_history:
        messages.append(
            {
                "content": msg[0].split()[0] if msg[0].startswith("exitcode") else msg[0],
                "role": "user",
            }
        )
        messages.append({"content": msg[1], "role": "assistant"})
    return messages


def oai_message_to_chat(oai_messages, sender):
    """Convert OpenAI message format to chat history."""
    chat_history = []
    messages = oai_messages[sender]
    if LOG_LEVEL == "DEBUG":
        print(f"oai_message_to_chat: {messages}")
    for i in range(0, len(messages), 2):
        chat_history.append(
            [
                messages[i]["content"],
                messages[i + 1]["content"] if i + 1 < len(messages) else "",
            ]
        )
    return chat_history


def agent_history_to_chat(agent_history):
    """Convert agent history to chat history."""
    chat_history = []
    for i in range(0, len(agent_history), 2):
        chat_history.append(
            [
                agent_history[i].messages,
                agent_history[i + 1].messages if i + 1 < len(agent_history) else None,
            ]
        )
    return chat_history
