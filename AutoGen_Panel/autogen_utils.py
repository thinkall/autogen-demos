import sys
import threading

import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.compressible_agent import CompressibleAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.agentchat.contrib.llava_agent import LLaVAAgent
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen.code_utils import extract_code


# autogen.ChatCompletion.start_logging()
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


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


def initialize_agents(llm_config, agent_name, system_msg, agent_type):
    if "UserProxy" in agent_type:
        agent = globals()[agent_type](
            name=agent_name,
            is_termination_msg=termination_msg,
            human_input_mode="NEVER",
            system_message=system_msg,
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            max_consecutive_auto_reply=5,
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,  # set to True or image name like "python:3" to use docker
            },
        )
    else:
        agent = globals()[agent_type](
            name=agent_name,
            is_termination_msg=termination_msg,
            human_input_mode="NEVER",
            system_message=system_msg,
            llm_config=llm_config,
        )

    return agent


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
